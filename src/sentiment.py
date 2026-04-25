"""
sentiment.py — FinBERT inference and sentiment aggregation for FinSentinel.

Pipeline:
  1. Load ProsusAI/finbert from HuggingFace (cached locally after first run)
  2. Run batched inference (batch_size from config) over article texts
  3. Return per-article {positive, negative, neutral} probability vectors
  4. Aggregate to daily scores per ticker with recency weighting
  5. Compute 3-day rolling sentiment momentum

Output:
  - Article-level: DataFrame[ticker, date, text, pos, neg, neu, sentiment_label]
  - Daily-level:   DataFrame[ticker, date, daily_score, sentiment_momentum,
                              article_count]
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.utils import AppConfig, chunked, clamp, get_logger, load_config

logger = get_logger(__name__)

# FinBERT label → column name mapping (model outputs these exact strings)
_LABEL_MAP: dict[str, str] = {
    "positive": "pos",
    "negative": "neg",
    "neutral": "neu",
}

# Article-level output schema
_ARTICLE_SCHEMA: dict[str, str] = {
    "ticker": "string",
    "date": "datetime64[ns, UTC]",
    "text": "string",
    "pos": "float64",
    "neg": "float64",
    "neu": "float64",
    "sentiment_label": "string",
}

# Daily aggregated output schema
_DAILY_SCHEMA: dict[str, str] = {
    "ticker": "string",
    "date": "datetime64[ns, UTC]",
    "daily_score": "float64",
    "sentiment_momentum": "float64",
    "article_count": "int64",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_CACHE: Optional[object] = None


def load_finbert(cfg: Optional[AppConfig] = None) -> object:
    """Load the FinBERT sentiment pipeline from HuggingFace.

    Results are module-level cached — the model loads only once per process.
    Automatically uses CUDA if available, falls back to CPU.

    Args:
        cfg: AppConfig with ``sentiment.model_name``; loaded if None.

    Returns:
        HuggingFace ``TextClassificationPipeline`` instance.
    """
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is not None:
        return _PIPELINE_CACHE

    cfg = cfg or load_config()
    model_name = cfg.sentiment.model_name

    device = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA" if device == 0 else "CPU"
    logger.info("Loading FinBERT model '%s' on %s...", model_name, device_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        sentiment_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,          # return all 3 class scores
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        logger.error("Failed to load FinBERT: %s", exc)
        raise

    _PIPELINE_CACHE = sentiment_pipeline
    logger.info("FinBERT loaded successfully.")
    return sentiment_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Batch Inference
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference_batch(
    texts: list[str],
    sentiment_pipeline: object,
) -> list[dict[str, float]]:
    """Run FinBERT on a single batch of texts.

    Args:
        texts: List of cleaned article texts.
        sentiment_pipeline: Loaded HuggingFace pipeline.

    Returns:
        List of dicts with keys ``pos``, ``neg``, ``neu`` (float probabilities
        summing to ~1.0).
    """
    raw_outputs = sentiment_pipeline(texts)  # type: ignore[operator]
    results: list[dict[str, float]] = []

    for output in raw_outputs:
        scores: dict[str, float] = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
        # output is a list of {"label": ..., "score": ...} dicts
        for item in output:
            label_key = _LABEL_MAP.get(item["label"].lower())
            if label_key:
                scores[label_key] = float(item["score"])
        results.append(scores)

    return results


def run_inference(
    df: pd.DataFrame,
    cfg: Optional[AppConfig] = None,
    sentiment_pipeline: Optional[object] = None,
) -> pd.DataFrame:
    """Run FinBERT sentiment inference over all articles in the DataFrame.

    Processes texts in batches of ``cfg.sentiment.batch_size`` for efficiency.
    Adds ``pos``, ``neg``, ``neu``, and ``sentiment_label`` columns.

    Args:
        df: Preprocessed DataFrame with a ``text`` column.
        cfg: AppConfig; loaded if None.
        sentiment_pipeline: Pre-loaded pipeline; loaded if None.

    Returns:
        Input DataFrame with appended sentiment probability columns.

    Raises:
        ValueError: If the ``text`` column is missing.
    """
    if "text" not in df.columns:
        raise ValueError("DataFrame must have a 'text' column for inference.")

    cfg = cfg or load_config()
    pipe = sentiment_pipeline or load_finbert(cfg)

    texts = df["text"].fillna("").tolist()
    batches = chunked(texts, cfg.sentiment.batch_size)
    n_batches = len(batches)

    logger.info(
        "Running FinBERT inference on %d articles in %d batches (batch_size=%d).",
        len(texts),
        n_batches,
        cfg.sentiment.batch_size,
    )

    all_scores: list[dict[str, float]] = []
    for i, batch in enumerate(batches, 1):
        if i % 10 == 0 or i == n_batches:
            logger.info("  Batch %d/%d...", i, n_batches)
        try:
            batch_scores = _run_inference_batch(batch, pipe)
        except Exception as exc:
            logger.error("Inference failed on batch %d: %s. Using neutral.", i, exc)
            batch_scores = [{"pos": 0.0, "neg": 0.0, "neu": 1.0}] * len(batch)
        all_scores.extend(batch_scores)

    result = df.copy()
    result["pos"] = [s["pos"] for s in all_scores]
    result["neg"] = [s["neg"] for s in all_scores]
    result["neu"] = [s["neu"] for s in all_scores]

    # Dominant label
    score_matrix = result[["pos", "neg", "neu"]].to_numpy()
    label_indices = score_matrix.argmax(axis=1)
    idx_to_label = {0: "positive", 1: "negative", 2: "neutral"}
    result["sentiment_label"] = pd.array(
        [idx_to_label[i] for i in label_indices], dtype="string"
    )

    # Enforce dtypes
    for col in ("pos", "neg", "neu"):
        result[col] = result[col].astype("float64")

    logger.info("Inference complete.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Daily Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _recency_weight(dates: pd.Series, cfg: AppConfig) -> pd.Series:
    """Compute per-article recency weights.

    Articles published within the last 24 hours receive
    ``cfg.sentiment.recency_weight_24h`` (default 2×). Older articles get 1×.

    Args:
        dates: UTC-aware datetime Series.
        cfg: AppConfig with recency weight parameter.

    Returns:
        Float Series of weights, one per article.
    """
    now = datetime.now(tz=timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    weights = pd.Series(
        np.where(dates >= cutoff_24h, cfg.sentiment.recency_weight_24h, 1.0),
        index=dates.index,
        dtype="float64",
    )
    return weights


def _compute_daily_score(group: pd.DataFrame, cfg: AppConfig) -> float:
    """Compute the weighted daily sentiment score for one (ticker, date) group.

    Score = weighted average of (pos - neg), where weights are recency-based.
    Range: [-1, 1], then rescaled to [0, 1].

    Args:
        group: Sub-DataFrame for one ticker on one date.
        cfg: AppConfig.

    Returns:
        Float daily sentiment score in [0, 1].
    """
    weights = _recency_weight(group["date"], cfg)
    raw_score = group["pos"] - group["neg"]          # range [-1, 1]
    weighted_avg = float((raw_score * weights).sum() / weights.sum())
    # Rescale from [-1, 1] → [0, 1]
    normalized = (weighted_avg + 1.0) / 2.0
    return clamp(normalized, 0.0, 1.0)


def aggregate_daily(
    df: pd.DataFrame,
    cfg: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """Aggregate article-level sentiment to daily scores per ticker.

    For each (ticker, date) pair:
      - ``daily_score``: Recency-weighted mean of (pos - neg), scaled [0, 1].
        0.5 = neutral, >0.5 = net positive, <0.5 = net negative.
      - ``sentiment_momentum``: 3-day rolling change in daily_score.
      - ``article_count``: Number of articles contributing to the score.

    Args:
        df: Article-level DataFrame with ``pos``, ``neg``, ``neu`` columns.
        cfg: AppConfig; loaded if None.

    Returns:
        DataFrame[ticker, date, daily_score, sentiment_momentum, article_count].
    """
    cfg = cfg or load_config()

    required = {"ticker", "date", "pos", "neg"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns for aggregation: {missing}")

    # Normalise date to UTC date-only (floor to day)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["day"] = df["date"].dt.floor("D")

    records: list[dict] = []
    for (ticker, day), group in df.groupby(["ticker", "day"], sort=True):
        score = _compute_daily_score(group, cfg)
        records.append(
            {
                "ticker": ticker,
                "date": day,
                "daily_score": score,
                "article_count": len(group),
            }
        )

    daily = pd.DataFrame(records)

    if daily.empty:
        logger.warning("No daily sentiment records produced.")
        return _empty_daily_df()

    # Compute momentum: rolling change over momentum_window_days
    window = cfg.sentiment.momentum_window_days
    daily = daily.sort_values(["ticker", "date"])

    daily["sentiment_momentum"] = (
        daily.groupby("ticker")["daily_score"]
        .transform(lambda s: s.diff(periods=window - 1).rolling(window=1).mean())
    )

    # Enforce dtypes
    daily["ticker"] = daily["ticker"].astype("string")
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily["daily_score"] = daily["daily_score"].astype("float64")
    daily["sentiment_momentum"] = daily["sentiment_momentum"].astype("float64")
    daily["article_count"] = daily["article_count"].astype("int64")

    daily.reset_index(drop=True, inplace=True)
    logger.info(
        "Daily aggregation complete: %d ticker-day records.", len(daily)
    )
    return daily[list(_DAILY_SCHEMA.keys())]


def _empty_daily_df() -> pd.DataFrame:
    """Return an empty DataFrame matching the daily sentiment schema.

    Returns:
        Empty typed DataFrame.
    """
    df = pd.DataFrame(columns=list(_DAILY_SCHEMA.keys()))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["daily_score"] = df["daily_score"].astype("float64")
    df["sentiment_momentum"] = df["sentiment_momentum"].astype("float64")
    df["article_count"] = df["article_count"].astype("int64")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze(
    df: pd.DataFrame,
    cfg: Optional[AppConfig] = None,
    sentiment_pipeline: Optional[object] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full sentiment analysis pipeline: inference + daily aggregation.

    Args:
        df: Preprocessed DataFrame from ``preprocessor.preprocess()``.
        cfg: AppConfig; loaded if None.
        sentiment_pipeline: Pre-loaded FinBERT pipeline; loaded if None.

    Returns:
        Tuple of:
          - article_df: Per-article DataFrame with pos/neg/neu/sentiment_label.
          - daily_df: Daily aggregated DataFrame per ticker.
    """
    cfg = cfg or load_config()
    article_df = run_inference(df, cfg=cfg, sentiment_pipeline=sentiment_pipeline)
    daily_df = aggregate_daily(article_df, cfg=cfg)
    return article_df, daily_df
