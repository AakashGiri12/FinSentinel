"""
test_backtest.py — Unit tests for src/backtest.py

Tests:
  - Kelly fraction calculation
  - Metrics computation with known equity curves
  - Equity curve construction
  - Trade log generation
  - Empty / edge case handling
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    compute_metrics,
    kelly_fraction,
    run_backtest,
)
from src.utils import load_config


@pytest.fixture()
def cfg():
    return load_config()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_equity(n: int = 252, drift: float = 0.0005) -> pd.DataFrame:
    """Build a synthetic equity curve with known properties."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    daily_rets = np.random.default_rng(42).normal(drift, 0.01, n)
    port = 100_000 * (1 + daily_rets).cumprod()
    bench = 100_000 * (1 + np.random.default_rng(99).normal(0.0003, 0.009, n)).cumprod()
    roll_max = np.maximum.accumulate(port)
    dd = (port - roll_max) / roll_max
    return pd.DataFrame({"date": rng, "portfolio_value": port,
                          "benchmark_value": bench, "drawdown": dd})


def _make_trade_log(n_wins: int, n_losses: int) -> pd.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_wins):
        rows.append({"ticker": "AAPL", "entry_date": base + timedelta(days=i*5),
                     "exit_date": base + timedelta(days=i*5+3),
                     "entry_price": 150.0, "exit_price": 155.0,
                     "shares": 0.1, "pnl": 5.0, "pnl_pct": 0.033,
                     "signal_confidence": 0.75, "exit_reason": "SELL signal"})
    for i in range(n_losses):
        rows.append({"ticker": "AAPL", "entry_date": base + timedelta(days=100+i*5),
                     "exit_date": base + timedelta(days=100+i*5+3),
                     "entry_price": 150.0, "exit_price": 145.0,
                     "shares": 0.1, "pnl": -5.0, "pnl_pct": -0.033,
                     "signal_confidence": 0.60, "exit_reason": "Max hold"})
    df = pd.DataFrame(rows)
    for c in ["entry_date", "exit_date"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Kelly Fraction Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKellyFraction:
    def test_output_in_valid_range(self):
        f = kelly_fraction(0.6, 0.05, 0.03, 0.5)
        assert 0.05 <= f <= 0.25

    def test_higher_win_rate_increases_fraction(self):
        f_low  = kelly_fraction(0.5, 0.05, 0.03, 0.5)
        f_high = kelly_fraction(0.8, 0.05, 0.03, 0.5)
        assert f_high >= f_low

    def test_zero_win_rate_returns_default(self):
        f = kelly_fraction(0.0, 0.05, 0.03, 0.5)
        assert f == pytest.approx(0.10)

    def test_zero_avg_win_returns_default(self):
        f = kelly_fraction(0.6, 0.0, 0.03, 0.5)
        assert f == pytest.approx(0.10)

    def test_clamps_to_maximum(self):
        f = kelly_fraction(0.99, 100.0, 0.001, 1.0)
        assert f <= 0.25

    def test_clamps_to_minimum(self):
        f = kelly_fraction(0.1, 0.001, 100.0, 0.5)
        assert f >= 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Computation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_all_keys_present(self, cfg):
        eq = _make_equity(252, 0.0005)
        tl = _make_trade_log(10, 5)
        m = compute_metrics(eq, tl, cfg)
        expected = ["total_return_pct", "annualized_return_pct", "sharpe_ratio",
                    "sortino_ratio", "max_drawdown_pct", "win_rate_pct",
                    "profit_factor", "benchmark_total_return_pct",
                    "benchmark_annualized_pct", "total_trades", "avg_hold_days", "alpha_pct"]
        for k in expected:
            assert k in m, f"Missing metric: {k}"

    def test_max_drawdown_is_negative_or_zero(self, cfg):
        eq = _make_equity(252, 0.0)
        tl = _make_trade_log(5, 5)
        m = compute_metrics(eq, tl, cfg)
        assert m["max_drawdown_pct"] <= 0.0

    def test_win_rate_correct_with_known_trades(self, cfg):
        eq = _make_equity(252, 0.0003)
        tl = _make_trade_log(6, 4)   # 6 wins / 10 total = 60%
        m = compute_metrics(eq, tl, cfg)
        assert m["win_rate_pct"] == pytest.approx(60.0, abs=0.01)

    def test_total_trades_correct(self, cfg):
        eq = _make_equity(252, 0.0003)
        tl = _make_trade_log(7, 3)
        m = compute_metrics(eq, tl, cfg)
        assert m["total_trades"] == 10

    def test_profit_factor_positive_when_winners_dominate(self, cfg):
        eq = _make_equity(252, 0.0005)
        tl = _make_trade_log(8, 2)
        m = compute_metrics(eq, tl, cfg)
        assert m["profit_factor"] > 1.0

    def test_positive_drift_yields_positive_return(self, cfg):
        eq = _make_equity(252, 0.001)   # strong positive drift
        tl = _make_trade_log(10, 2)
        m = compute_metrics(eq, tl, cfg)
        assert m["total_return_pct"] > 0

    def test_empty_equity_returns_zero_metrics(self, cfg):
        m = compute_metrics(pd.DataFrame(columns=["date","portfolio_value",
                                                   "benchmark_value","drawdown"]),
                            pd.DataFrame(), cfg)
        assert m["total_return_pct"] == 0.0
        assert m["sharpe_ratio"] == 0.0

    def test_empty_trade_log_sets_win_rate_zero(self, cfg):
        eq = _make_equity(252, 0.0003)
        m = compute_metrics(eq, pd.DataFrame(), cfg)
        assert m["win_rate_pct"] == 0.0

    def test_sharpe_ratio_positive_with_positive_drift(self, cfg):
        eq = _make_equity(252, 0.001)
        m = compute_metrics(eq, _make_trade_log(5, 2), cfg)
        assert m["sharpe_ratio"] > 0

    def test_avg_hold_days_correct(self, cfg):
        """All trades have 3-day hold → avg_hold_days ≈ 3."""
        eq = _make_equity(252, 0.0003)
        tl = _make_trade_log(5, 5)
        m = compute_metrics(eq, tl, cfg)
        assert m["avg_hold_days"] == pytest.approx(3.0, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# run_backtest Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunBacktest:
    def _make_signal_df(self) -> pd.DataFrame:
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base + timedelta(days=i) for i in range(30)]
        signals = []
        for i, d in enumerate(dates):
            sig = "BUY" if i % 7 == 0 else ("SELL" if i % 7 == 5 else "HOLD")
            signals.append({"ticker": "AAPL", "date": d, "signal": sig,
                            "confidence": 0.7, "raw_score": 0.65})
        df = pd.DataFrame(signals)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["ticker"] = df["ticker"].astype("string")
        df["signal"] = df["signal"].astype("string")
        return df

    def _make_prices(self) -> pd.DataFrame:
        """Synthetic OHLCV prices for AAPL only."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        idx = pd.date_range(base, periods=35, freq="B", tz="UTC")
        rng = np.random.default_rng(7)
        close = 150.0 * (1 + rng.normal(0.0005, 0.01, len(idx))).cumprod()
        opens = close * (1 + rng.normal(0, 0.003, len(idx)))
        df = pd.DataFrame({
            ("Open",  "AAPL"): opens,
            ("High",  "AAPL"): close * 1.01,
            ("Low",   "AAPL"): close * 0.99,
            ("Close", "AAPL"): close,
            ("Volume","AAPL"): rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
        }, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def test_returns_three_objects(self, cfg):
        sig_df  = self._make_signal_df()
        prices  = self._make_prices()
        equity, metrics, trades = run_backtest(sig_df, cfg=cfg, prices=prices)
        assert isinstance(equity, pd.DataFrame)
        assert isinstance(metrics, dict)
        assert isinstance(trades, pd.DataFrame)

    def test_metrics_keys_present(self, cfg):
        sig_df = self._make_signal_df()
        _, metrics, _ = run_backtest(sig_df, cfg=cfg, prices=self._make_prices())
        assert "sharpe_ratio" in metrics
        assert "total_return_pct" in metrics

    def test_equity_curve_starts_at_initial_capital(self, cfg):
        sig_df = self._make_signal_df()
        equity, _, _ = run_backtest(sig_df, cfg=cfg, prices=self._make_prices())
        if not equity.empty:
            assert equity["portfolio_value"].iloc[0] == pytest.approx(
                cfg.backtest.initial_capital, rel=0.05
            )

    def test_empty_signal_df_raises(self, cfg):
        with pytest.raises(ValueError):
            run_backtest(pd.DataFrame(), cfg=cfg)

    def test_missing_signal_columns_raises(self, cfg):
        bad = pd.DataFrame({"ticker": ["AAPL"], "date": [pd.Timestamp.now(tz="UTC")]})
        with pytest.raises(ValueError, match="missing columns"):
            run_backtest(bad, cfg=cfg)
