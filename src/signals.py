"""
signals.py — Trading signal generation from FinBERT sentiment scores.

Logic:
  BUY  → daily_score > buy_threshold  AND sentiment_momentum > momentum_threshold
  SELL → daily_score < sell_threshold AND sentiment_momentum < -momentum_threshold
  HOLD → all other cases

Additional features:
  - Confidence score (0–1) proportional to score distance from thresholds
  - Signal smoothing: only emit a signal if it holds for ``consistency_days``
  - Raw score column preserved for downstream inspection

Output: DataFrame[ticker, date, signal, confidence, raw_score]
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.utils import AppConfig, clamp, get_logger, load_config

logger = get_logger(__name__)

# ── Signal labels ─────────────────────────────────────────────────────────────
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

# ── Output schema ─────────────────────────────────────────────────────────────
_SIGNAL_SCHEMA: dict[str, str] = {
    "ticker": "string",
    "date": "datetime64[ns, UTC]",
    "signal": "string",
    "confidence": "float64",
    "raw_score": "float64",
}


# ─────────────────────────────────────────────────────────────────────────────
# Signal Classification (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_raw(
    daily_score: pd.Series,
    momentum: pd.Series,
    cfg: AppConfig,
) -> pd.Series:
    """Vectorized classification of raw BUY/SELL/HOLD signals.

    Args:
        daily_score: Series of daily sentiment scores in [0, 1].
        momentum: Series of sentiment momentum values.
        cfg: AppConfig with signal thresholds.

    Returns:
        String Series with values ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
    """
    buy_thresh = cfg.sentiment.buy_threshold
    sell_thresh = cfg.sentiment.sell_threshold
    mom_thresh = cfg.sentiment.momentum_threshold

    buy_mask = (daily_score > buy_thresh) & (momentum > mom_thresh)
    sell_mask = (daily_score < sell_thresh) & (momentum < -mom_thresh)

    raw_signal = pd.Series(HOLD, index=daily_score.index, dtype="string")
    raw_signal = raw_signal.where(~buy_mask, other=BUY)
    raw_signal = raw_signal.where(~sell_mask, other=SELL)
    return raw_signal


def _compute_confidence(
    daily_score: pd.Series,
    momentum: pd.Series,
    raw_signal: pd.Series,
    cfg: AppConfig,
) -> pd.Series:
    """Compute a 0–1 confidence score per signal based on threshold distance.

    For BUY:  confidence ∝ (score − buy_threshold) + (momentum − mom_threshold)
    For SELL: confidence ∝ (sell_threshold − score) + (mom_threshold + |momentum|)
    For HOLD: fixed at 0.5 (uncertain)

    All values are clamped and normalised to [0, 1].

    Args:
        daily_score: Daily sentiment score Series.
        momentum: Sentiment momentum Series.
        raw_signal: Signal label Series (BUY/SELL/HOLD).
        cfg: AppConfig.

    Returns:
        Float Series of confidence values in [0, 1].
    """
    buy_thresh = cfg.sentiment.buy_threshold
    sell_thresh = cfg.sentiment.sell_threshold
    mom_thresh = cfg.sentiment.momentum_threshold

    # Normalise distance contributions independently
    score_range = 1.0 - buy_thresh  # max possible BUY distance from threshold
    mom_range = 1.0 - mom_thresh

    buy_conf = (
        (daily_score - buy_thresh).clip(lower=0) / score_range
        + (momentum - mom_thresh).clip(lower=0) / mom_range
    ) / 2.0

    sell_conf = (
        (sell_thresh - daily_score).clip(lower=0) / sell_thresh
        + (-momentum - mom_thresh).clip(lower=0) / mom_range
    ) / 2.0

    confidence = pd.Series(0.5, index=daily_score.index, dtype="float64")
    confidence = confidence.where(raw_signal != BUY, other=buy_conf.clip(0, 1))
    confidence = confidence.where(raw_signal != SELL, other=sell_conf.clip(0, 1))

    return confidence.apply(lambda x: clamp(float(x), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Signal Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_signals(
    df: pd.DataFrame,
    consistency_days: int,
) -> pd.DataFrame:
    """Apply consistency-based signal smoothing per ticker.

    A signal is only emitted if it equals the signal on the previous
    ``consistency_days - 1`` trading days. Otherwise it reverts to HOLD.

    This prevents overtrading on noisy one-day sentiment spikes.

    Args:
        df: DataFrame with ``ticker``, ``date``, ``signal`` columns, sorted by
            (ticker, date) ascending.
        consistency_days: Minimum number of consecutive days a signal must
            persist before it is kept (1 = no smoothing).

    Returns:
        DataFrame with smoothed ``signal`` column (HOLD replaces inconsistent signals).
    """
    if consistency_days <= 1:
        return df

    df = df.copy()

    def _smooth_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        signals = group["signal"].tolist()
        smoothed: list[str] = []
        for i, sig in enumerate(signals):
            if i < consistency_days - 1:
                smoothed.append(HOLD)
                continue
            window = signals[i - (consistency_days - 1): i + 1]
            # All signals in window must match
            smoothed.append(sig if all(s == sig for s in window) else HOLD)
        group["signal"] = pd.array(smoothed, dtype="string")
        return group

    result = (
        df.groupby("ticker", group_keys=False)
        .apply(_smooth_group)
        .reset_index(drop=True)
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Schema Enforcement
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column order and explicit dtypes for the signal output.

    Args:
        df: Input DataFrame.

    Returns:
        Schema-conformant DataFrame.
    """
    for col in _SIGNAL_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[list(_SIGNAL_SCHEMA.keys())].copy()

    for col, dtype in _SIGNAL_SCHEMA.items():
        if dtype == "datetime64[ns, UTC]":
            df[col] = pd.to_datetime(df[col], utc=True)
        elif dtype == "float64":
            df[col] = df[col].astype("float64")
        elif dtype == "string":
            df[col] = df[col].astype("string")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(
    daily_df: pd.DataFrame,
    cfg: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """Generate BUY/SELL/HOLD trading signals from daily sentiment scores.

    Full pipeline:
      1. Vectorized signal classification using sentiment thresholds
      2. Confidence score computation based on distance from thresholds
      3. Signal smoothing (consistency filter over N days)

    Args:
        daily_df: Output of ``sentiment.aggregate_daily()`` with columns
                  [ticker, date, daily_score, sentiment_momentum, article_count].
        cfg: AppConfig; loaded from config.yaml if None.

    Returns:
        DataFrame[ticker, date, signal, confidence, raw_score] with explicit dtypes.

    Raises:
        ValueError: If required columns are missing from ``daily_df``.
    """
    cfg = cfg or load_config()

    required = {"ticker", "date", "daily_score", "sentiment_momentum"}
    if missing := required - set(daily_df.columns):
        raise ValueError(f"Missing columns in daily_df: {missing}")

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Fill NaN momentum with 0 (no information → no signal)
    df["sentiment_momentum"] = df["sentiment_momentum"].fillna(0.0)

    logger.info(
        "Generating signals for %d ticker-day rows. "
        "Thresholds: BUY>%.2f, SELL<%.2f, MOM±%.2f, consistency=%d days.",
        len(df),
        cfg.sentiment.buy_threshold,
        cfg.sentiment.sell_threshold,
        cfg.sentiment.momentum_threshold,
        cfg.signals.consistency_days,
    )

    # Step 1: Raw signal classification (vectorized)
    raw_signal = _classify_raw(df["daily_score"], df["sentiment_momentum"], cfg)
    df["signal"] = raw_signal

    # Step 2: Confidence scores
    df["confidence"] = _compute_confidence(
        df["daily_score"], df["sentiment_momentum"], raw_signal, cfg
    )

    # Step 3: Preserve raw score before smoothing
    df["raw_score"] = df["daily_score"].astype("float64")

    # Step 4: Smooth signals (consistency filter)
    df = _smooth_signals(df, cfg.signals.consistency_days)

    # Log distribution
    counts = df["signal"].value_counts().to_dict()
    logger.info(
        "Signal distribution: BUY=%d, SELL=%d, HOLD=%d",
        counts.get(BUY, 0),
        counts.get(SELL, 0),
        counts.get(HOLD, 0),
    )

    return _enforce_schema(df)


def get_latest_signals(
    signal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return only the most recent signal for each ticker.

    Useful for the Live Signals page of the dashboard.

    Args:
        signal_df: Full signal DataFrame from ``generate_signals()``.

    Returns:
        DataFrame with one row per ticker (the most recent signal date).
    """
    latest = (
        signal_df.sort_values("date", ascending=False)
        .groupby("ticker", as_index=False)
        .first()
    )
    return latest.reset_index(drop=True)
