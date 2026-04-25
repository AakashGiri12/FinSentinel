"""
test_signals.py — Unit tests for src/signals.py
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest
from src.signals import (
    BUY, HOLD, SELL,
    _classify_raw, _compute_confidence, _smooth_signals,
    generate_signals, get_latest_signals,
)
from src.utils import load_config


@pytest.fixture()
def cfg():
    return load_config()


def _multi_day(ticker, scores, moms):
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    return pd.DataFrame({
        "ticker": pd.array([ticker] * len(scores), dtype="string"),
        "date": pd.to_datetime([base + timedelta(days=i) for i in range(len(scores))], utc=True),
        "daily_score": scores, "sentiment_momentum": moms,
        "article_count": [10] * len(scores),
    })


def _single_day(tickers, scores, moms):
    now = datetime.now(tz=timezone.utc)
    return pd.DataFrame({
        "ticker": pd.array(tickers, dtype="string"),
        "date": pd.to_datetime([now] * len(tickers), utc=True),
        "daily_score": scores, "sentiment_momentum": moms,
        "article_count": [5] * len(tickers),
    })


class TestClassifyRaw:
    def test_buy_signal(self, cfg):
        r = _classify_raw(pd.Series([0.75]), pd.Series([0.15]), cfg)
        assert r.iloc[0] == BUY

    def test_sell_signal(self, cfg):
        r = _classify_raw(pd.Series([0.25]), pd.Series([-0.15]), cfg)
        assert r.iloc[0] == SELL

    def test_hold_neutral(self, cfg):
        r = _classify_raw(pd.Series([0.5]), pd.Series([0.0]), cfg)
        assert r.iloc[0] == HOLD

    def test_high_score_low_momentum_is_hold(self, cfg):
        # momentum=0.01 is below momentum_threshold=0.02 → HOLD
        r = _classify_raw(pd.Series([0.75]), pd.Series([0.01]), cfg)
        assert r.iloc[0] == HOLD

    def test_low_score_weak_neg_momentum_is_hold(self, cfg):
        # momentum=-0.01 is above -momentum_threshold=-0.02 → HOLD
        r = _classify_raw(pd.Series([0.25]), pd.Series([-0.01]), cfg)
        assert r.iloc[0] == HOLD

    def test_at_exact_buy_threshold_is_hold(self, cfg):
        # score == buy_threshold (0.55) uses strict > so it is HOLD
        r = _classify_raw(pd.Series([0.55]), pd.Series([0.15]), cfg)
        assert r.iloc[0] == HOLD

    def test_at_exact_sell_threshold_is_hold(self, cfg):
        # score == sell_threshold (0.45) uses strict < so it is HOLD
        r = _classify_raw(pd.Series([0.45]), pd.Series([-0.15]), cfg)
        assert r.iloc[0] == HOLD

    def test_vectorized_three_signals(self, cfg):
        r = _classify_raw(pd.Series([0.75, 0.5, 0.25]), pd.Series([0.15, 0.0, -0.15]), cfg)
        assert list(r) == [BUY, HOLD, SELL]


class TestComputeConfidence:
    def test_buy_confidence_in_range(self, cfg):
        c = _compute_confidence(pd.Series([0.8]), pd.Series([0.2]), pd.Series([BUY]), cfg)
        assert 0.0 <= c.iloc[0] <= 1.0

    def test_sell_confidence_in_range(self, cfg):
        c = _compute_confidence(pd.Series([0.2]), pd.Series([-0.2]), pd.Series([SELL]), cfg)
        assert 0.0 <= c.iloc[0] <= 1.0

    def test_hold_confidence_is_half(self, cfg):
        c = _compute_confidence(pd.Series([0.5]), pd.Series([0.0]), pd.Series([HOLD]), cfg)
        assert c.iloc[0] == pytest.approx(0.5)

    def test_stronger_buy_has_higher_confidence(self, cfg):
        c = _compute_confidence(pd.Series([0.65, 0.90]), pd.Series([0.12, 0.35]),
                                pd.Series([BUY, BUY]), cfg)
        assert c.iloc[1] > c.iloc[0]


class TestSmoothSignals:
    def _df(self, sigs):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return pd.DataFrame({
            "ticker": pd.array(["AAPL"] * len(sigs), dtype="string"),
            "date": pd.to_datetime([base + timedelta(days=i) for i in range(len(sigs))], utc=True),
            "signal": pd.array(sigs, dtype="string"),
            "confidence": [0.7] * len(sigs), "raw_score": [0.7] * len(sigs),
        })

    def test_single_buy_reverts_to_hold(self):
        r = _smooth_signals(self._df([HOLD, HOLD, BUY, HOLD, HOLD]), 2)
        assert r.iloc[2]["signal"] == HOLD

    def test_two_consecutive_buys_kept(self):
        r = _smooth_signals(self._df([HOLD, BUY, BUY, HOLD]), 2)
        assert r.iloc[2]["signal"] == BUY

    def test_consistency_one_passthrough(self):
        sigs = [BUY, SELL, HOLD, BUY]
        r = _smooth_signals(self._df(sigs), 1)
        assert list(r["signal"]) == sigs

    def test_first_rows_always_hold(self):
        r = _smooth_signals(self._df([BUY, BUY, BUY, BUY]), 3)
        assert r.iloc[0]["signal"] == HOLD
        assert r.iloc[1]["signal"] == HOLD


class TestGenerateSignals:
    def test_output_columns(self, cfg):
        df = _multi_day("AAPL", [0.7], [0.2])
        r = generate_signals(df, cfg=cfg)
        for col in ("ticker", "date", "signal", "confidence", "raw_score"):
            assert col in r.columns

    def test_buy_generated(self, cfg):
        df = _multi_day("AAPL", [0.75, 0.75], [0.2, 0.2])
        r = generate_signals(df, cfg=cfg)
        assert BUY in r["signal"].values

    def test_sell_generated(self, cfg):
        df = _multi_day("JPM", [0.25, 0.25], [-0.2, -0.2])
        r = generate_signals(df, cfg=cfg)
        assert SELL in r["signal"].values

    def test_all_neutral_is_hold(self, cfg):
        df = _multi_day("MSFT", [0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
        r = generate_signals(df, cfg=cfg)
        assert (r["signal"] == HOLD).all()

    def test_nan_momentum_no_crash(self, cfg):
        df = _single_day(["GS"], [0.7], [float("nan")])
        r = generate_signals(df, cfg=cfg)
        assert not r["signal"].isna().any()

    def test_missing_columns_raises(self, cfg):
        with pytest.raises(ValueError, match="Missing columns"):
            generate_signals(pd.DataFrame({"ticker": ["AAPL"]}), cfg=cfg)

    def test_confidence_is_float64(self, cfg):
        df = _multi_day("AAPL", [0.7], [0.2])
        r = generate_signals(df, cfg=cfg)
        assert r["confidence"].dtype == np.float64

    def test_multi_ticker(self, cfg):
        df = pd.concat([
            _multi_day("AAPL", [0.75, 0.75], [0.2, 0.2]),
            _multi_day("JPM",  [0.25, 0.25], [-0.2, -0.2]),
        ])
        r = generate_signals(df, cfg=cfg)
        assert {"AAPL", "JPM"}.issubset(set(r["ticker"].tolist()))


class TestGetLatestSignals:
    def test_one_row_per_ticker(self, cfg):
        df = _multi_day("AAPL", [0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
        r = get_latest_signals(generate_signals(df, cfg=cfg))
        assert len(r[r["ticker"] == "AAPL"]) == 1

    def test_returns_max_date(self, cfg):
        df = _multi_day("MSFT", [0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
        sigs = generate_signals(df, cfg=cfg)
        latest = get_latest_signals(sigs)
        assert latest[latest["ticker"] == "MSFT"].iloc[0]["date"] == sigs[sigs["ticker"] == "MSFT"]["date"].max()
