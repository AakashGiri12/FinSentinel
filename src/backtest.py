"""
backtest.py — Vectorized backtesting engine for FinSentinel.

Strategy:
  - Enter position on next day's OPEN after a BUY signal
  - Exit on SELL signal OR after max_hold_days, whichever comes first
  - Position sizing via simplified (half-)Kelly Criterion
  - Transaction cost applied on entry and exit
  - Benchmark: simple Buy-and-Hold from start to end date

Performance metrics returned:
  Total Return %, Annualized Return %, Sharpe Ratio, Sortino Ratio,
  Max Drawdown %, Win Rate %, Profit Factor, vs Benchmark comparisons

Output:
  - equity_curve: DataFrame[date, portfolio_value, benchmark_value, drawdown]
  - metrics:      dict with all performance statistics
  - trade_log:    DataFrame[ticker, entry_date, exit_date, entry_price,
                             exit_price, pnl, pnl_pct, signal_confidence]
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import AppConfig, get_logger, load_config

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Output schemas ────────────────────────────────────────────────────────────
_EQUITY_SCHEMA: dict[str, str] = {
    "date": "datetime64[ns, UTC]",
    "portfolio_value": "float64",
    "benchmark_value": "float64",
    "drawdown": "float64",
}

_TRADE_SCHEMA: dict[str, str] = {
    "ticker": "string",
    "entry_date": "datetime64[ns, UTC]",
    "exit_date": "datetime64[ns, UTC]",
    "entry_price": "float64",
    "exit_price": "float64",
    "shares": "float64",
    "pnl": "float64",
    "pnl_pct": "float64",
    "signal_confidence": "float64",
    "exit_reason": "string",
}

TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────────────────
# Price Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for a list of tickers using yfinance.

    Args:
        tickers: List of ticker symbols.
        start: Start date string (``"YYYY-MM-DD"``).
        end: End date string (``"YYYY-MM-DD"``).

    Returns:
        MultiIndex DataFrame with (OHLCV columns) × (ticker) structure,
        index = UTC-aware DatetimeIndex.

    Raises:
        ValueError: If no price data could be fetched for any ticker.
    """
    logger.info("Fetching OHLCV data for %s from %s to %s...", tickers, start, end)
    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        raise ValueError(f"yfinance download failed: {exc}") from exc

    if raw.empty:
        raise ValueError("yfinance returned empty DataFrame. Check tickers and date range.")

    # Ensure UTC timezone
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")

    logger.info("Fetched %d trading days of price data.", len(raw))
    return raw


def _extract_column(prices: pd.DataFrame, col: str, ticker: str) -> pd.Series:
    """Safely extract a single price column for one ticker from a MultiIndex DataFrame.

    Args:
        prices: MultiIndex OHLCV DataFrame from ``fetch_prices()``.
        col: Column name (``"Open"``, ``"Close"``, etc.).
        ticker: Ticker symbol.

    Returns:
        Price Series with DatetimeIndex.

    Raises:
        KeyError: If the ticker or column is not present.
    """
    if isinstance(prices.columns, pd.MultiIndex):
        return prices[(col, ticker)].dropna()
    return prices[col].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Kelly Position Sizing
# ─────────────────────────────────────────────────────────────────────────────

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_multiplier: float = 0.5,
) -> float:
    """Compute fractional Kelly position size.

    Formula: f* = (p / |loss|) - (q / |win|), scaled by kelly_multiplier.
    Half-Kelly (0.5) is used by default for risk management.

    Args:
        win_rate: Historical win probability (0–1).
        avg_win: Average winning trade return (positive float).
        avg_loss: Average losing trade return (positive float, magnitude only).
        kelly_multiplier: Scaling factor; 0.5 = half-Kelly.

    Returns:
        Position fraction of available capital to allocate, clamped to [0.05, 0.25].
    """
    if avg_win <= 0 or avg_loss <= 0 or win_rate <= 0:
        return 0.10  # default safe fraction

    loss_rate = 1.0 - win_rate
    f_full = (win_rate / avg_loss) - (loss_rate / avg_win)
    f_scaled = f_full * kelly_multiplier

    # Hard limits: never bet less than 5% or more than 25% per trade
    return float(np.clip(f_scaled, 0.05, 0.25))


# ─────────────────────────────────────────────────────────────────────────────
# Core Vectorized Backtest
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_ticker(
    ticker: str,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: AppConfig,
    position_fraction: float,
) -> tuple[list[dict], pd.Series]:
    """Run vectorized backtest for a single ticker.

    Uses pandas vectorized operations — no per-day Python loops.
    The trade logic is implemented via shift() and cumsum() operations.

    Args:
        ticker: Ticker symbol.
        signals: Signal DataFrame filtered for this ticker.
        prices: Full MultiIndex OHLCV DataFrame.
        cfg: AppConfig with backtest parameters.
        position_fraction: Kelly fraction of cash to deploy per trade.

    Returns:
        Tuple of:
          - trades: List of trade dicts for the trade log.
          - pnl_series: Daily P&L Series indexed by date.
    """
    tc = cfg.backtest.transaction_cost
    max_hold = cfg.backtest.max_hold_days

    try:
        opens = _extract_column(prices, "Open", ticker)
        _extract_column(prices, "Close", ticker)  # validate data exists
    except KeyError:
        logger.warning("No price data for %s — skipping.", ticker)
        return [], pd.Series(dtype="float64")

    # Align signals to trading days
    sig_df = signals[signals["ticker"] == ticker].copy()
    sig_df["date"] = pd.to_datetime(sig_df["date"], utc=True).dt.normalize()
    sig_df = sig_df.set_index("date").reindex(opens.index).fillna(
        {"signal": "HOLD", "confidence": 0.5, "raw_score": 0.5}
    )

    # ── Vectorized trade identification ─────────────────────────────────────
    all_dates = opens.index
    signals_series = sig_df["signal"].reindex(all_dates).fillna("HOLD")
    conf_series = sig_df["confidence"].reindex(all_dates).fillna(0.5)

    is_buy = (signals_series == "BUY").shift(1, fill_value=False).values
    is_sell = (signals_series == "SELL").shift(1, fill_value=False).values

    entry_indices = np.where(is_buy)[0]
    sell_indices = np.where(is_sell)[0]

    valid_entries = []
    valid_exits = []
    exit_reasons = []

    last_exit_idx = -1
    for entry_idx in entry_indices:
        if entry_idx <= last_exit_idx:
            continue
            
        future_sells = sell_indices[sell_indices > entry_idx]
        first_sell_idx = future_sells[0] if len(future_sells) > 0 else len(opens) - 1
        max_hold_idx = min(entry_idx + max_hold, len(opens) - 1)
        
        if first_sell_idx <= max_hold_idx:
            exit_idx = first_sell_idx
            reason = "SELL signal"
        else:
            exit_idx = max_hold_idx
            reason = f"Max hold ({max_hold}d)"
            
        valid_entries.append(entry_idx)
        valid_exits.append(exit_idx)
        exit_reasons.append(reason)
        last_exit_idx = exit_idx

    daily_pnl = pd.Series(0.0, index=all_dates)
    trades: list[dict] = []

    for i in range(len(valid_entries)):
        en_idx = valid_entries[i]
        ex_idx = valid_exits[i]
        
        en_price = float(opens.iloc[en_idx]) * (1 + tc)
        ex_price = float(opens.iloc[ex_idx]) * (1 - tc)
        
        pnl = ex_price - en_price
        pnl_pct = pnl / en_price
        
        trades.append({
            "ticker": ticker,
            "entry_date": all_dates[en_idx],
            "exit_date": all_dates[ex_idx],
            "entry_price": en_price,
            "exit_price": ex_price,
            "shares": position_fraction,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "signal_confidence": float(conf_series.iloc[en_idx - 1]),
            "exit_reason": exit_reasons[i],
        })
        daily_pnl.iloc[ex_idx] = pnl_pct * position_fraction

    return trades, daily_pnl


def _build_equity_curve(
    all_daily_pnl: dict[str, pd.Series],
    prices: pd.DataFrame,
    tickers: list[str],
    cfg: AppConfig,
) -> pd.DataFrame:
    """Build the portfolio equity curve from per-ticker daily P&L series.

    Args:
        all_daily_pnl: Dict mapping ticker → daily P&L fraction Series.
        prices: Full OHLCV price DataFrame (for benchmark).
        tickers: List of tickers.
        cfg: AppConfig.

    Returns:
        Equity curve DataFrame[date, portfolio_value, benchmark_value, drawdown].
    """
    if not all_daily_pnl:
        return pd.DataFrame(columns=list(_EQUITY_SCHEMA.keys()))

    # Combine all ticker P&L — sum across tickers (each is a fraction of capital)
    combined = pd.concat(all_daily_pnl.values(), axis=1).fillna(0.0)
    total_daily_return = combined.sum(axis=1)

    capital = cfg.backtest.initial_capital
    portfolio = (1 + total_daily_return).cumprod() * capital

    # ── Benchmark: equal-weight buy-and-hold ─────────────────────────────────
    bench_returns = []
    for ticker in tickers:
        try:
            c = _extract_column(prices, "Close", ticker)
            bench_returns.append(c.pct_change().fillna(0))
        except KeyError:
            continue

    if bench_returns:
        bench_combined = pd.concat(bench_returns, axis=1).mean(axis=1)
        benchmark = (1 + bench_combined).cumprod() * capital
        benchmark = benchmark.reindex(portfolio.index).ffill().bfill()
    else:
        benchmark = pd.Series(capital, index=portfolio.index)

    # ── Drawdown ─────────────────────────────────────────────────────────────
    rolling_max = portfolio.expanding().max()
    drawdown = (portfolio - rolling_max) / rolling_max  # negative values

    equity = pd.DataFrame({
        "date": portfolio.index,
        "portfolio_value": portfolio.values,
        "benchmark_value": benchmark.reindex(portfolio.index).values,
        "drawdown": drawdown.values,
    })

    equity["date"] = pd.to_datetime(equity["date"], utc=True)
    equity["portfolio_value"] = equity["portfolio_value"].astype("float64")
    equity["benchmark_value"] = equity["benchmark_value"].astype("float64")
    equity["drawdown"] = equity["drawdown"].astype("float64")

    return equity.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    cfg: AppConfig,
) -> dict:
    """Calculate comprehensive performance metrics from equity curve and trade log.

    Args:
        equity_curve: DataFrame from ``_build_equity_curve()``.
        trade_log: DataFrame of all executed trades.
        cfg: AppConfig.

    Returns:
        Dict with keys:
          total_return_pct, annualized_return_pct, sharpe_ratio, sortino_ratio,
          max_drawdown_pct, win_rate_pct, profit_factor,
          benchmark_total_return_pct, benchmark_annualized_pct,
          total_trades, avg_hold_days.
    """
    initial = cfg.backtest.initial_capital

    if equity_curve.empty or len(equity_curve) < 2:
        return _empty_metrics()

    final_value = float(equity_curve["portfolio_value"].iloc[-1])
    bench_final = float(equity_curve["benchmark_value"].iloc[-1])

    total_return = (final_value - initial) / initial
    bench_total_return = (bench_final - initial) / initial

    # Daily returns
    port_vals = equity_curve["portfolio_value"].values
    daily_rets = np.diff(port_vals) / port_vals[:-1]
    bench_vals = equity_curve["benchmark_value"].values
    _ = np.diff(bench_vals) / bench_vals[:-1]  # bench_rets reserved for future use

    n_days = len(daily_rets)
    years = n_days / TRADING_DAYS_PER_YEAR

    # Annualized return
    ann_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    bench_ann_return = (1 + bench_total_return) ** (1 / max(years, 0.01)) - 1

    # Sharpe (annualized, risk-free = 0 for simplicity)
    if daily_rets.std() > 0:
        sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0

    # Sortino (only downside deviation)
    downside = daily_rets[daily_rets < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(daily_rets.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sortino = 0.0

    # Max Drawdown
    max_dd = float(equity_curve["drawdown"].min())  # already negative

    # Trade statistics
    if not trade_log.empty:
        winning = trade_log[trade_log["pnl"] > 0]
        losing = trade_log[trade_log["pnl"] <= 0]
        win_rate = len(winning) / len(trade_log) if len(trade_log) > 0 else 0.0
        gross_profit = float(winning["pnl"].sum()) if not winning.empty else 0.0
        gross_loss = abs(float(losing["pnl"].sum())) if not losing.empty else 1e-9
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        # Average hold duration
        trade_log_copy = trade_log.copy()
        trade_log_copy["entry_date"] = pd.to_datetime(trade_log_copy["entry_date"], utc=True)
        trade_log_copy["exit_date"] = pd.to_datetime(trade_log_copy["exit_date"], utc=True)
        hold_days = (
            (trade_log_copy["exit_date"] - trade_log_copy["entry_date"])
            .dt.days
            .mean()
        )
    else:
        win_rate = 0.0
        profit_factor = 0.0
        hold_days = 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "benchmark_total_return_pct": round(bench_total_return * 100, 2),
        "benchmark_annualized_pct": round(bench_ann_return * 100, 2),
        "total_trades": len(trade_log) if not trade_log.empty else 0,
        "avg_hold_days": round(float(hold_days), 1) if hold_days else 0.0,
        "alpha_pct": round((total_return - bench_total_return) * 100, 2),
    }


def _empty_metrics() -> dict:
    """Return a metrics dict with all zero/null values.

    Returns:
        Metrics dict with zero values for all keys.
    """
    return {
        "total_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "benchmark_total_return_pct": 0.0,
        "benchmark_annualized_pct": 0.0,
        "total_trades": 0,
        "avg_hold_days": 0.0,
        "alpha_pct": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    signal_df: pd.DataFrame,
    cfg: Optional[AppConfig] = None,
    prices: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run the full vectorized backtest and return results.

    Args:
        signal_df: Output of ``signals.generate_signals()`` with columns
                   [ticker, date, signal, confidence, raw_score].
        cfg: AppConfig; loaded from config.yaml if None.
        prices: Pre-fetched OHLCV DataFrame (optional; fetched if None).

    Returns:
        Tuple of:
          - equity_curve: DataFrame[date, portfolio_value, benchmark_value, drawdown]
          - metrics: Dict of performance statistics
          - trade_log: DataFrame of all executed trades

    Raises:
        ValueError: If signal_df is empty or missing required columns.
    """
    cfg = cfg or load_config()

    required = {"ticker", "date", "signal", "confidence"}
    if missing := required - set(signal_df.columns):
        raise ValueError(f"signal_df missing columns: {missing}")

    if signal_df.empty:
        logger.warning("Empty signal_df — returning empty backtest results.")
        return pd.DataFrame(columns=list(_EQUITY_SCHEMA.keys())), _empty_metrics(), pd.DataFrame(columns=list(_TRADE_SCHEMA.keys()))

    tickers = list(signal_df["ticker"].unique())

    # Fetch prices if not provided
    if prices is None:
        prices = fetch_prices(tickers, cfg.date_range.start, cfg.date_range.end)

    # Estimate Kelly fraction from a warm-start heuristic
    # (will be updated iteratively in a live system)
    pos_fraction = cfg.backtest.kelly_fraction * 0.20  # 20% base × Kelly multiplier

    logger.info(
        "Running backtest: %d tickers, capital=$%.0f, position_size=%.1f%%, tc=%.2f%%",
        len(tickers),
        cfg.backtest.initial_capital,
        pos_fraction * 100,
        cfg.backtest.transaction_cost * 100,
    )

    # ── Per-ticker backtest ───────────────────────────────────────────────────
    all_trades: list[dict] = []
    all_daily_pnl: dict[str, pd.Series] = {}

    for ticker in tickers:
        ticker_signals = signal_df[signal_df["ticker"] == ticker]
        trades, daily_pnl = _run_single_ticker(
            ticker, ticker_signals, prices, cfg, pos_fraction
        )
        all_trades.extend(trades)
        if not daily_pnl.empty:
            all_daily_pnl[ticker] = daily_pnl

    # ── Aggregate results ─────────────────────────────────────────────────────
    trade_log = pd.DataFrame(all_trades) if all_trades else pd.DataFrame(columns=list(_TRADE_SCHEMA.keys()))
    for col, dtype in _TRADE_SCHEMA.items():
        if col in trade_log.columns:
            if "datetime" in dtype:
                trade_log[col] = pd.to_datetime(trade_log[col], utc=True, errors="coerce")
            elif dtype == "float64":
                trade_log[col] = trade_log[col].astype("float64")
            elif dtype == "string":
                trade_log[col] = trade_log[col].astype("string")

    equity_curve = _build_equity_curve(all_daily_pnl, prices, tickers, cfg)
    metrics = compute_metrics(equity_curve, trade_log, cfg)

    logger.info(
        "Backtest complete | Return: %.2f%% | Sharpe: %.3f | MaxDD: %.2f%% | Trades: %d",
        metrics["total_return_pct"],
        metrics["sharpe_ratio"],
        metrics["max_drawdown_pct"],
        metrics["total_trades"],
    )

    return equity_curve, metrics, trade_log
