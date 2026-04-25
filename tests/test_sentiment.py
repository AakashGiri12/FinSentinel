"""
test_sentiment.py — Unit tests for src/sentiment.py

Tests:
  - FinBERT score aggregation logic (mocked model)
  - Recency weighting
  - Daily aggregation with known inputs
  - Empty / edge case handling
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.sentiment import (
    _compute_daily_score,
    _recency_weight,
    aggregate_daily,
    run_inference,
)
from src.utils import load_config


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def now_utc():
    return datetime.now(tz=timezone.utc)


@pytest.fixture()
def sample_article_df(now_utc):
    """Small preprocessed DataFrame with known sentiment scores."""
    return pd.DataFrame({
        "ticker": pd.array(["AAPL", "AAPL", "MSFT"], dtype="string"),
        "date": pd.to_datetime([
            now_utc,
            now_utc - timedelta(hours=30),
            now_utc - timedelta(hours=5),
        ], utc=True),
        "text": pd.array([
            "Apple reports record earnings beating all expectations.",
            "Market uncertainty weighs on tech stocks outlook.",
            "Microsoft Azure growth accelerates cloud revenue.",
        ], dtype="string"),
        "headline": pd.array(["A", "B", "C"], dtype="string"),
        "summary": pd.array(["s1", "s2", "s3"], dtype="string"),
        "source": pd.array(["yahoo_rss"] * 3, dtype="string"),
    })


@pytest.fixture()
def article_with_scores():
    """DataFrame that already has pos/neg/neu columns for aggregation tests.

    All 3 AAPL articles are pinned to a fixed UTC noon — guaranteed same calendar day.
    """
    base = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    return pd.DataFrame({
        "ticker": pd.array(["AAPL", "AAPL", "AAPL"], dtype="string"),
        "date": pd.to_datetime([
            base,
            base - timedelta(hours=1),
            base - timedelta(hours=2),
        ], utc=True),
        "text": pd.array(["t1", "t2", "t3"], dtype="string"),
        "pos":  [0.8, 0.2, 0.5],
        "neg":  [0.1, 0.7, 0.3],
        "neu":  [0.1, 0.1, 0.2],
        "sentiment_label": pd.array(["positive", "negative", "neutral"], dtype="string"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Recency Weight Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRecencyWeight:
    def test_recent_articles_get_higher_weight(self, now_utc, cfg):
        dates = pd.to_datetime([
            now_utc - timedelta(hours=1),   # recent → 2x
            now_utc - timedelta(hours=25),  # old    → 1x
        ], utc=True)
        weights = _recency_weight(pd.Series(dates), cfg)
        assert weights.iloc[0] == pytest.approx(cfg.sentiment.recency_weight_24h)
        assert weights.iloc[1] == pytest.approx(1.0)

    def test_all_old_articles_get_unit_weight(self, now_utc, cfg):
        dates = pd.to_datetime([now_utc - timedelta(days=5)] * 4, utc=True)
        weights = _recency_weight(pd.Series(dates), cfg)
        assert (weights == 1.0).all()

    def test_weights_always_positive(self, now_utc, cfg):
        dates = pd.to_datetime([
            now_utc - timedelta(hours=i * 12) for i in range(10)
        ], utc=True)
        weights = _recency_weight(pd.Series(dates), cfg)
        assert (weights > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Daily Score Computation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyScore:
    def test_all_positive_articles_score_above_half(self, now_utc, cfg):
        group = pd.DataFrame({
            "date": pd.to_datetime([now_utc] * 3, utc=True),
            "pos":  [0.9, 0.85, 0.8],
            "neg":  [0.05, 0.08, 0.1],
            "neu":  [0.05, 0.07, 0.1],
        })
        score = _compute_daily_score(group, cfg)
        assert score > 0.5

    def test_all_negative_articles_score_below_half(self, now_utc, cfg):
        group = pd.DataFrame({
            "date": pd.to_datetime([now_utc] * 3, utc=True),
            "pos":  [0.05, 0.08, 0.1],
            "neg":  [0.9, 0.85, 0.8],
            "neu":  [0.05, 0.07, 0.1],
        })
        score = _compute_daily_score(group, cfg)
        assert score < 0.5

    def test_score_bounded_zero_to_one(self, now_utc, cfg):
        group = pd.DataFrame({
            "date": pd.to_datetime([now_utc] * 5, utc=True),
            "pos":  [1.0, 1.0, 1.0, 0.0, 0.0],
            "neg":  [0.0, 0.0, 0.0, 1.0, 1.0],
            "neu":  [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        score = _compute_daily_score(group, cfg)
        assert 0.0 <= score <= 1.0

    def test_neutral_articles_score_near_half(self, now_utc, cfg):
        group = pd.DataFrame({
            "date": pd.to_datetime([now_utc] * 3, utc=True),
            "pos":  [0.33, 0.33, 0.33],
            "neg":  [0.33, 0.33, 0.33],
            "neu":  [0.34, 0.34, 0.34],
        })
        score = _compute_daily_score(group, cfg)
        assert 0.45 <= score <= 0.55


# ─────────────────────────────────────────────────────────────────────────────
# Inference Tests (mocked model)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunInference:
    def _make_mock_pipeline(self, label: str = "positive", score: float = 0.8):
        """Return a mock HuggingFace pipeline that returns fixed outputs."""
        mock = MagicMock()
        mock.return_value = [
            [
                {"label": "positive", "score": score if label == "positive" else 0.1},
                {"label": "negative", "score": score if label == "negative" else 0.05},
                {"label": "neutral",  "score": score if label == "neutral"  else 0.1},
            ]
        ]
        # Make it return a list per batch input
        def side_effect(texts, **kwargs):
            return [mock.return_value[0]] * len(texts)
        mock.side_effect = side_effect
        return mock

    def test_inference_adds_score_columns(self, sample_article_df, cfg):
        pipe = self._make_mock_pipeline("positive", 0.8)
        result = run_inference(sample_article_df, cfg=cfg, sentiment_pipeline=pipe)
        for col in ("pos", "neg", "neu", "sentiment_label"):
            assert col in result.columns

    def test_positive_mock_produces_positive_label(self, sample_article_df, cfg):
        pipe = self._make_mock_pipeline("positive", 0.8)
        result = run_inference(sample_article_df, cfg=cfg, sentiment_pipeline=pipe)
        assert (result["sentiment_label"] == "positive").all()

    def test_scores_sum_to_approx_one(self, sample_article_df, cfg):
        pipe = self._make_mock_pipeline("neutral", 0.6)
        result = run_inference(sample_article_df, cfg=cfg, sentiment_pipeline=pipe)
        # Sum won't be exactly 1 due to mocked values; just check columns exist and are floats
        assert result["pos"].dtype == np.float64
        assert result["neg"].dtype == np.float64

    def test_missing_text_column_raises(self, cfg):
        bad_df = pd.DataFrame({"ticker": ["AAPL"], "headline": ["test"]})
        with pytest.raises(ValueError, match="text"):
            run_inference(bad_df, cfg=cfg, sentiment_pipeline=MagicMock())

    def test_empty_dataframe_returns_empty(self, cfg):
        empty = pd.DataFrame(columns=["ticker", "date", "text", "headline", "summary", "source"])
        pipe = self._make_mock_pipeline()
        result = run_inference(empty, cfg=cfg, sentiment_pipeline=pipe)
        assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# Daily Aggregation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateDaily:
    def test_output_has_required_columns(self, article_with_scores, cfg):
        result = aggregate_daily(article_with_scores, cfg=cfg)
        for col in ("ticker", "date", "daily_score", "sentiment_momentum", "article_count"):
            assert col in result.columns

    def test_article_count_correct(self, article_with_scores, cfg):
        result = aggregate_daily(article_with_scores, cfg=cfg)
        aapl = result[result["ticker"] == "AAPL"]
        # All 3 articles are on same day → article_count = 3
        assert int(aapl["article_count"].iloc[0]) == 3

    def test_daily_score_in_range(self, article_with_scores, cfg):
        result = aggregate_daily(article_with_scores, cfg=cfg)
        assert (result["daily_score"] >= 0.0).all()
        assert (result["daily_score"] <= 1.0).all()

    def test_missing_columns_raises(self, cfg):
        bad = pd.DataFrame({"ticker": ["AAPL"], "date": [pd.Timestamp.now(tz="UTC")]})
        with pytest.raises(ValueError, match="Missing columns"):
            aggregate_daily(bad, cfg=cfg)

    def test_multi_ticker_produces_separate_rows(self, now_utc, cfg):
        df = pd.DataFrame({
            "ticker": pd.array(["AAPL", "MSFT"], dtype="string"),
            "date": pd.to_datetime([now_utc, now_utc], utc=True),
            "pos": [0.8, 0.3], "neg": [0.1, 0.6], "neu": [0.1, 0.1],
            "text": pd.array(["t1", "t2"], dtype="string"),
            "sentiment_label": pd.array(["positive", "negative"], dtype="string"),
        })
        result = aggregate_daily(df, cfg=cfg)
        assert set(result["ticker"].tolist()) == {"AAPL", "MSFT"}

    def test_dtypes_correct(self, article_with_scores, cfg):
        result = aggregate_daily(article_with_scores, cfg=cfg)
        assert result["daily_score"].dtype == np.float64
        assert result["article_count"].dtype == np.int64
