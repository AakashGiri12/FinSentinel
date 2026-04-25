"""
preprocessor.py — Text cleaning and feature preparation for FinSentinel.

Pipeline:
  1. Remove HTML tags and decode HTML entities
  2. Strip special characters while preserving sentence structure
  3. Drop duplicate articles (exact headline match per ticker)
  4. Normalize all datetimes to UTC
  5. Filter out irrelevant articles via configurable keyword list
  6. Discard articles shorter than min_article_length tokens
  7. Combine headline + summary into a single ``text`` field for FinBERT

Output: DataFrame[ticker, date, text, headline, summary, source]
"""

from __future__ import annotations

import html
import re
from typing import Optional

import pandas as pd

from src.utils import AppConfig, get_logger, load_config

logger = get_logger(__name__)

_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")
_RE_SPECIAL_CHARS = re.compile(r"[^\w\s\.\,\!\?\-\:\;\'\"\(\)\/\%\$\+\=\&\@\#]")
_RE_URL = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")

_OUTPUT_SCHEMA: dict[str, str] = {
    "ticker": "string",
    "date": "datetime64[ns, UTC]",
    "text": "string",
    "headline": "string",
    "summary": "string",
    "source": "string",
}


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning Primitives
# ─────────────────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities from a string.

    Args:
        text: Raw string potentially containing HTML markup.

    Returns:
        Clean plain-text string.
    """
    decoded = html.unescape(text or "")
    return _RE_HTML_TAG.sub(" ", decoded).strip()


def remove_urls(text: str) -> str:
    """Strip URLs from text.

    Args:
        text: Input string.

    Returns:
        String with URLs removed.
    """
    return _RE_URL.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple consecutive whitespace characters into one space.

    Args:
        text: Input string.

    Returns:
        Whitespace-normalized string.
    """
    return _RE_MULTI_SPACE.sub(" ", text).strip()


def remove_special_chars(text: str) -> str:
    """Remove non-alphanumeric characters except common punctuation.

    Args:
        text: Input string.

    Returns:
        Cleaned string with rare special characters removed.
    """
    return _RE_SPECIAL_CHARS.sub(" ", text)


def clean_text(text: str) -> str:
    """Apply the full text cleaning pipeline to a single string.

    Order: HTML strip → URL removal → special chars → whitespace normalise.

    Args:
        text: Raw article text (headline or summary).

    Returns:
        Cleaned string suitable for FinBERT tokenisation.
    """
    text = strip_html(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-Level Cleaning Steps
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``clean_text`` to ``headline`` and ``summary`` columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned text columns.
    """
    df = df.copy()
    df["headline"] = df["headline"].fillna("").apply(clean_text)
    df["summary"] = df["summary"].fillna("").apply(clean_text)
    return df


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the ``date`` column is UTC-aware datetime64.

    Args:
        df: DataFrame with a ``date`` column.

    Returns:
        DataFrame with UTC dates; rows with NaT dropped.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    n_before = len(df)
    df.dropna(subset=["date"], inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d rows with unparseable dates.", n_dropped)
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate articles based on (ticker, headline) pairs.

    Keeps the earliest occurrence when duplicates are found.

    Args:
        df: Input DataFrame.

    Returns:
        De-duplicated DataFrame.
    """
    n_before = len(df)
    df = df.sort_values("date", ascending=True)
    df = df.drop_duplicates(subset=["ticker", "headline"], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Deduplication removed %d duplicate articles.", n_dropped)
    return df


def _filter_irrelevant(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Drop articles whose text contains any configured irrelevant keywords.

    Args:
        df: Input DataFrame with ``headline`` and ``summary`` columns.
        cfg: Application config providing the keyword list.

    Returns:
        Filtered DataFrame.
    """
    keywords = list(cfg.preprocessor.irrelevant_keywords)
    if cfg.preprocessor.keep_press_releases and "press release" in keywords:
        keywords = [k for k in keywords if k != "press release"]

    if not keywords:
        return df

    pattern = "|".join(re.escape(kw) for kw in keywords)
    combined_text = (df["headline"] + " " + df["summary"]).str.lower()
    mask = combined_text.str.contains(pattern, regex=True, na=False)

    n_filtered = int(mask.sum())
    if n_filtered:
        logger.info("Keyword filter removed %d irrelevant articles.", n_filtered)

    return df[~mask].copy()


def _build_text_field(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate headline and summary into a single ``text`` field for FinBERT.

    Args:
        df: DataFrame with ``headline`` and ``summary`` columns.

    Returns:
        DataFrame with added ``text`` column.
    """
    df = df.copy()
    df["text"] = (
        df["headline"].str.strip() + ". " + df["summary"].str.strip()
    ).str.strip(". ")
    df["text"] = df["text"].astype("string")
    return df


def _filter_short_articles(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Remove articles whose combined text is shorter than min_article_length tokens.

    Args:
        df: DataFrame with ``text`` column.
        cfg: Config providing ``min_article_length``.

    Returns:
        DataFrame with short articles removed.
    """
    token_counts = df["text"].str.split().str.len().fillna(0)
    mask = token_counts >= cfg.preprocessor.min_article_length
    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.info("Short-article filter removed %d articles.", n_dropped)
    return df[mask].copy()


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns and enforce explicit dtypes.

    Args:
        df: Input DataFrame.

    Returns:
        Schema-conformant DataFrame.
    """
    for col in _OUTPUT_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[list(_OUTPUT_SCHEMA.keys())].copy()

    for col, dtype in _OUTPUT_SCHEMA.items():
        if dtype == "datetime64[ns, UTC]":
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        elif dtype == "string":
            df[col] = df[col].astype("string")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    cfg: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a raw scraped DataFrame.

    Steps (in order):
      1. Clean HTML / URLs / special chars from headline + summary
      2. Normalise dates to UTC
      3. Deduplicate by (ticker, headline)
      4. Filter irrelevant articles via keyword list
      5. Build combined ``text`` field
      6. Drop articles shorter than ``min_article_length`` tokens
      7. Enforce output schema and dtypes

    Args:
        df: Raw DataFrame from ``scraper.scrape_all()``.
        cfg: AppConfig; loaded from config.yaml if None.

    Returns:
        Cleaned DataFrame[ticker, date, text, headline, summary, source].

    Raises:
        ValueError: If the input DataFrame is missing required columns.
    """
    cfg = cfg or load_config()

    required = {"ticker", "date", "headline", "summary", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    n_input = len(df)
    logger.info("Preprocessing %d raw articles...", n_input)

    df = _clean_text_columns(df)
    df = _normalize_dates(df)
    df = _deduplicate(df)
    df = _filter_irrelevant(df, cfg)
    df = _build_text_field(df)
    df = _filter_short_articles(df, cfg)
    df = _enforce_schema(df)

    df.sort_values(["ticker", "date"], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Preprocessing complete: %d → %d articles (dropped %d).",
        n_input,
        len(df),
        n_input - len(df),
    )
    return df


def preprocess_and_save(
    df: pd.DataFrame,
    output_path: str = "data/processed/cleaned_articles.csv",
    cfg: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """Preprocess and persist the cleaned DataFrame to disk.

    Args:
        df: Raw scraped DataFrame.
        output_path: Relative path for the output CSV.
        cfg: AppConfig; loaded if None.

    Returns:
        Cleaned DataFrame.
    """
    from src.utils import ensure_dir, resolve_data_path

    cfg = cfg or load_config()
    cleaned = preprocess(df, cfg)

    out = resolve_data_path(output_path)
    ensure_dir(out.parent)
    cleaned.to_csv(out, index=False)
    logger.info("Saved processed data → %s", out)
    return cleaned
