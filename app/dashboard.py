"""
dashboard.py — FinSentinel Streamlit Dashboard (3 pages).

Pages:
  1. Live Signals    — BUY/SELL/HOLD table with confidence bars
  2. Sentiment Explorer — Daily score charts + sample headlines
  3. Backtest Results   — Equity curve, metrics table, trade log
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; color: #e6edf3; }
.block-container { padding: 2rem 2.5rem; }
.signal-buy  { background:#1a3a2a; color:#3fb950; border:1px solid #3fb950;
               border-radius:6px; padding:4px 12px; font-weight:700; }
.signal-sell { background:#3a1a1a; color:#f85149; border:1px solid #f85149;
               border-radius:6px; padding:4px 12px; font-weight:700; }
.signal-hold { background:#2a2a3a; color:#8b949e; border:1px solid #8b949e;
               border-radius:6px; padding:4px 12px; font-weight:700; }
.metric-card { background:#161b22; border:1px solid #30363d; border-radius:10px;
               padding:1.2rem; text-align:center; }
.metric-value { font-size:2rem; font-weight:700; }
.metric-label { font-size:0.8rem; color:#8b949e; text-transform:uppercase;
                letter-spacing:0.05em; margin-top:4px; }
.stTabs [data-baseweb="tab"] { font-size:0.95rem; font-weight:600; }
div[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
h1,h2,h3 { color:#e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders (use session_state to avoid re-running pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading FinBERT model...")
def _load_pipeline():
    from src.sentiment import load_finbert
    return load_finbert()

@st.cache_data(ttl=1800, show_spinner="Scraping & analysing news...")
def run_full_pipeline(tickers: tuple, start: str, end: str):
    """Run scrape → preprocess → sentiment → signals and return all DataFrames."""
    import dataclasses
    cfg = load_config()
    cfg = dataclasses.replace(
        cfg,
        tickers=tickers,
        date_range=dataclasses.replace(cfg.date_range, start=start, end=end),
    )
    from src.scraper import scrape_all
    from src.preprocessor import preprocess
    from src.sentiment import analyze
    from src.signals import generate_signals

    raw_df = scrape_all(cfg=cfg, save=False)
    if raw_df.empty:
        return None, None, None, None

    clean_df = preprocess(raw_df, cfg=cfg)
    if clean_df.empty:
        return raw_df, None, None, None

    pipe = _load_pipeline()
    article_df, daily_df = analyze(clean_df, cfg=cfg, sentiment_pipeline=pipe)
    signal_df = generate_signals(daily_df, cfg=cfg)
    return article_df, daily_df, signal_df, cfg


@st.cache_data(ttl=1800, show_spinner="Running backtest...")
def run_backtest_cached(tickers: tuple, start: str, end: str):
    from src.backtest import run_backtest, fetch_prices
    from datetime import datetime, timedelta
    _, daily_df, signal_df, cfg2 = run_full_pipeline(tickers, start, end)
    if signal_df is None:
        return None, None, None
    # yfinance end is exclusive — add 1 day so today's data is included
    end_inclusive = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    prices = fetch_prices(list(tickers), start, end_inclusive)
    equity, metrics, trades = run_backtest(signal_df, cfg=cfg2, prices=prices)
    return equity, metrics, trades


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ FinSentinel")
    st.markdown("*Financial Sentiment → Trading Signals*")
    st.divider()

    cfg_default = load_config()
    all_tickers = list(cfg_default.tickers)

    selected_tickers = st.multiselect(
        "Tickers", options=all_tickers + ["AMZN", "NVDA", "TSLA", "GOOGL"],
        default=all_tickers, key="tickers_select"
    )
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=90), key="start_date")
    with col2:
        end_date = st.date_input("To", value=date.today(), key="end_date")

    st.divider()
    run_btn = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")
    run_bt  = st.button("📊 Run Backtest", use_container_width=True)

    st.divider()
    st.markdown("<small>⚠️ Educational use only. Not financial advice.</small>", unsafe_allow_html=True)

# ── Init session state ────────────────────────────────────────────────────────
for k in ("article_df", "daily_df", "signal_df", "equity", "metrics", "trades", "pipeline_cfg"):
    if k not in st.session_state:
        st.session_state[k] = None

if run_btn and selected_tickers:
    with st.spinner("Running pipeline…"):
        art, daily, sigs, pcfg = run_full_pipeline(
            tuple(selected_tickers), str(start_date), str(end_date)
        )
    st.session_state.update(article_df=art, daily_df=daily, signal_df=sigs, pipeline_cfg=pcfg)
    st.success("Pipeline complete!")

if run_bt and selected_tickers:
    with st.spinner("Backtesting…"):
        eq, met, trd = run_backtest_cached(
            tuple(selected_tickers), str(start_date), str(end_date)
        )
    st.session_state.update(equity=eq, metrics=met, trades=trd)
    st.success("Backtest complete!")

# ─────────────────────────────────────────────────────────────────────────────
# Page tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📡 Live Signals", "🧠 Sentiment Explorer", "📈 Backtest Results"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📡 Live Trading Signals")
    st.caption("Signals generated from FinBERT sentiment analysis. Refresh every 30 min.")

    signal_df: Optional[pd.DataFrame] = st.session_state.signal_df

    if signal_df is None or signal_df.empty:
        st.info("👈 Select tickers & click **Run Pipeline** to generate signals.")

        # Demo data for layout preview
        demo = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "JPM", "GS", "MS"],
            "signal": ["BUY", "HOLD", "SELL", "BUY", "HOLD"],
            "confidence": [0.82, 0.51, 0.74, 0.68, 0.49],
            "raw_score": [0.71, 0.53, 0.31, 0.66, 0.52],
            "date": [pd.Timestamp.now(tz="UTC")] * 5,
        })
        signal_df = demo
        st.markdown("*Preview with demo data:*")

    # Latest signal per ticker
    latest = (
        signal_df.sort_values("date", ascending=False)
        .groupby("ticker", as_index=False).first()
        .sort_values("signal")
    )

    # Summary KPI row
    buy_c = (latest["signal"] == "BUY").sum()
    sell_c = (latest["signal"] == "SELL").sum()
    hold_c = (latest["signal"] == "HOLD").sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#3fb950">{buy_c}</div><div class="metric-label">BUY Signals</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f85149">{sell_c}</div><div class="metric-label">SELL Signals</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#8b949e">{hold_c}</div><div class="metric-label">HOLD Signals</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#58a6ff">{len(latest)}</div><div class="metric-label">Tickers Tracked</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Signal cards
    cols = st.columns(min(len(latest), 5))
    for i, (_, row) in enumerate(latest.iterrows()):
        sig = row["signal"]
        css = {"BUY": "signal-buy", "SELL": "signal-sell"}.get(sig, "signal-hold")
        color = {"BUY": "#3fb950", "SELL": "#f85149"}.get(sig, "#8b949e")
        conf = float(row["confidence"])
        score = float(row.get("raw_score", 0.5))
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:1rem">
              <div style="font-size:1.4rem;font-weight:700;color:#e6edf3">{row['ticker']}</div>
              <div class="metric-label" style="margin:6px 0">
                <span class="{css}">{sig}</span>
              </div>
              <div style="margin:8px 0">
                <div style="font-size:0.75rem;color:#8b949e;margin-bottom:3px">Confidence</div>
                <div style="background:#21262d;border-radius:4px;height:8px;overflow:hidden">
                  <div style="width:{conf*100:.0f}%;height:100%;background:{color};border-radius:4px"></div>
                </div>
                <div style="font-size:0.8rem;color:{color};margin-top:3px">{conf:.0%}</div>
              </div>
              <div style="font-size:0.75rem;color:#8b949e">Sentiment: <b style="color:#e6edf3">{score:.3f}</b></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Full Signal Table")

    def _color_signal(val: str) -> str:
        colors = {"BUY": "color: #3fb950; font-weight:bold",
                  "SELL": "color: #f85149; font-weight:bold",
                  "HOLD": "color: #8b949e"}
        return colors.get(val, "")

    display = latest[["ticker", "signal", "confidence", "raw_score", "date"]].copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
    display["confidence"] = display["confidence"].map("{:.1%}".format)
    display["raw_score"] = display["raw_score"].map("{:.4f}".format)

    styled = display.style.applymap(_color_signal, subset=["signal"])
    st.dataframe(styled, use_container_width=True, height=250)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SENTIMENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🧠 Sentiment Explorer")

    daily_df: Optional[pd.DataFrame] = st.session_state.daily_df
    article_df: Optional[pd.DataFrame] = st.session_state.article_df

    if daily_df is None or daily_df.empty:
        # Generate demo data
        rng = pd.date_range(start=str(start_date), end=str(end_date), freq="B", tz="UTC")
        demo_tickers = selected_tickers or ["AAPL", "MSFT"]
        rows = []
        for t in demo_tickers:
            scores = np.random.uniform(0.3, 0.75, len(rng))
            mom = np.diff(scores, prepend=scores[0])
            for d, s, m in zip(rng, scores, mom):
                rows.append({"ticker": t, "date": d, "daily_score": s,
                             "sentiment_momentum": m, "article_count": np.random.randint(3, 20)})
        daily_df = pd.DataFrame(rows)
        st.info("👈 Run the pipeline for real data. Showing demo below.")

    ticker_sel = st.selectbox("Select Ticker", options=daily_df["ticker"].unique().tolist(), key="sent_ticker")
    df_t = daily_df[daily_df["ticker"] == ticker_sel].sort_values("date")

    # Sentiment score line chart
    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=df_t["date"], y=df_t["daily_score"],
        name="Daily Score", line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"
    ))
    fig_score.add_hline(y=0.6, line_dash="dash", line_color="#3fb950",
                        annotation_text="BUY threshold", annotation_position="right")
    fig_score.add_hline(y=0.4, line_dash="dash", line_color="#f85149",
                        annotation_text="SELL threshold", annotation_position="right")
    fig_score.update_layout(
        title=f"Daily Sentiment Score — {ticker_sel}",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3", yaxis_range=[0, 1],
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        height=350, margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_score, use_container_width=True)

    col_l, col_r = st.columns(2)

    # Momentum chart
    with col_l:
        fig_mom = go.Figure()
        colors_mom = ["#3fb950" if v >= 0 else "#f85149" for v in df_t["sentiment_momentum"]]
        fig_mom.add_trace(go.Bar(
            x=df_t["date"], y=df_t["sentiment_momentum"],
            marker_color=colors_mom, name="Momentum"
        ))
        fig_mom.update_layout(
            title="Sentiment Momentum (3-day rolling)",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e6edf3", height=300,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_mom, use_container_width=True)

    # Article volume bar chart
    with col_r:
        fig_vol = px.bar(
            df_t, x="date", y="article_count",
            title="Article Volume per Day",
            color_discrete_sequence=["#8b949e"]
        )
        fig_vol.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e6edf3", height=300,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # Sample headlines
    st.markdown("---")
    st.markdown("#### 📰 Sample Headlines")
    if article_df is not None and not article_df.empty:
        art_t = article_df[article_df["ticker"] == ticker_sel].copy()
        if "pos" in art_t.columns and "neg" in art_t.columns:
            top_pos = art_t.nlargest(3, "pos")[["headline", "pos", "date"]]
            top_neg = art_t.nlargest(3, "neg")[["headline", "neg", "date"]]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🟢 Most Positive**")
                for _, r in top_pos.iterrows():
                    st.markdown(f'<div class="metric-card" style="margin-bottom:8px;text-align:left">'
                                f'<span style="color:#3fb950;font-weight:600">{r["pos"]:.2%}</span> '
                                f'— {r["headline"][:120]}…</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("**🔴 Most Negative**")
                for _, r in top_neg.iterrows():
                    st.markdown(f'<div class="metric-card" style="margin-bottom:8px;text-align:left">'
                                f'<span style="color:#f85149;font-weight:600">{r["neg"]:.2%}</span> '
                                f'— {r["headline"][:120]}…</div>', unsafe_allow_html=True)
    else:
        st.info("Run the pipeline to see headline analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📈 Backtest Results")

    equity: Optional[pd.DataFrame] = st.session_state.equity
    metrics: Optional[dict] = st.session_state.metrics
    trades: Optional[pd.DataFrame] = st.session_state.trades

    if metrics is None:
        # Demo metrics
        metrics = {
            "total_return_pct": 18.4, "annualized_return_pct": 22.1,
            "sharpe_ratio": 1.38, "sortino_ratio": 1.91,
            "max_drawdown_pct": -11.2, "win_rate_pct": 58.3,
            "profit_factor": 1.72, "benchmark_total_return_pct": 12.6,
            "benchmark_annualized_pct": 15.0, "total_trades": 47,
            "avg_hold_days": 6.2, "alpha_pct": 5.8,
        }
        rng = pd.date_range(start=str(start_date), end=str(end_date), freq="B", tz="UTC")
        port = 100000 * (1 + pd.Series(np.random.normal(0.0008, 0.012, len(rng)))).cumprod()
        bench = 100000 * (1 + pd.Series(np.random.normal(0.0005, 0.010, len(rng)))).cumprod()
        roll_max = port.expanding().max()
        equity = pd.DataFrame({"date": rng, "portfolio_value": port.values,
                               "benchmark_value": bench.values,
                               "drawdown": ((port - roll_max) / roll_max).values})
        st.info("👈 Click **Run Backtest** for real results. Showing demo data.")

    # KPI metrics row
    m = metrics
    def kpi(label, value, color="#58a6ff"):
        return (f'<div class="metric-card"><div class="metric-value" style="color:{color}">'
                f'{value}</div><div class="metric-label">{label}</div></div>')

    r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns(6)
    ret_color = "#3fb950" if m["total_return_pct"] >= 0 else "#f85149"
    alpha_color = "#3fb950" if m["alpha_pct"] >= 0 else "#f85149"
    dd_color = "#f85149"

    r1c1.markdown(kpi("Total Return", f'{m["total_return_pct"]:+.1f}%', ret_color), unsafe_allow_html=True)
    r1c2.markdown(kpi("Ann. Return", f'{m["annualized_return_pct"]:+.1f}%', ret_color), unsafe_allow_html=True)
    r1c3.markdown(kpi("Sharpe Ratio", f'{m["sharpe_ratio"]:.2f}'), unsafe_allow_html=True)
    r1c4.markdown(kpi("Sortino Ratio", f'{m["sortino_ratio"]:.2f}'), unsafe_allow_html=True)
    r1c5.markdown(kpi("Max Drawdown", f'{m["max_drawdown_pct"]:.1f}%', dd_color), unsafe_allow_html=True)
    r1c6.markdown(kpi("Alpha vs BnH", f'{m["alpha_pct"]:+.1f}%', alpha_color), unsafe_allow_html=True)

    r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns(6)
    r2c1.markdown(kpi("Win Rate", f'{m["win_rate_pct"]:.1f}%'), unsafe_allow_html=True)
    r2c2.markdown(kpi("Profit Factor", f'{m["profit_factor"]:.2f}'), unsafe_allow_html=True)
    r2c3.markdown(kpi("Total Trades", str(m["total_trades"])), unsafe_allow_html=True)
    r2c4.markdown(kpi("Avg Hold Days", f'{m["avg_hold_days"]:.1f}d'), unsafe_allow_html=True)
    r2c5.markdown(kpi("Bench Return", f'{m["benchmark_total_return_pct"]:+.1f}%', "#8b949e"), unsafe_allow_html=True)
    r2c6.markdown(kpi("Bench Ann.", f'{m["benchmark_annualized_pct"]:+.1f}%', "#8b949e"), unsafe_allow_html=True)

    st.markdown("---")

    # Equity curve
    fig_eq = go.Figure()
    if equity is not None and not equity.empty:
        fig_eq.add_trace(go.Scatter(
            x=equity["date"], y=equity["portfolio_value"],
            name="FinSentinel Strategy", line=dict(color="#58a6ff", width=2.5)
        ))
        fig_eq.add_trace(go.Scatter(
            x=equity["date"], y=equity["benchmark_value"],
            name="Buy & Hold Benchmark", line=dict(color="#8b949e", width=1.5, dash="dot")
        ))
    fig_eq.update_layout(
        title="Portfolio Equity Curve vs Buy-and-Hold Benchmark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d", tickprefix="$"),
        height=420, legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        hovermode="x unified", margin=dict(l=60, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown chart
    if equity is not None and not equity.empty:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=equity["date"], y=equity["drawdown"] * 100,
            fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
            line=dict(color="#f85149", width=1.5), name="Drawdown %"
        ))
        fig_dd.update_layout(
            title="Drawdown (%)",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d", ticksuffix="%"),
            height=220, showlegend=False, margin=dict(l=60, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # Trade log
    st.markdown("---")
    st.markdown("#### 🗂️ Trade Log")
    if trades is not None:
        if not trades.empty:
            tl = trades.copy()
            for dc in ["entry_date", "exit_date"]:
                if dc in tl.columns:
                    tl[dc] = pd.to_datetime(tl[dc]).dt.strftime("%Y-%m-%d")
            for nc in ["entry_price", "exit_price"]:
                if nc in tl.columns:
                    tl[nc] = tl[nc].map("${:.2f}".format)
            if "pnl" in tl.columns:
                tl["pnl"] = tl["pnl"].map("${:+.2f}".format)
            if "pnl_pct" in tl.columns:
                tl["pnl_pct"] = tl["pnl_pct"].map("{:+.2%}".format)
            if "signal_confidence" in tl.columns:
                tl["signal_confidence"] = tl["signal_confidence"].map("{:.1%}".format)

            def _pnl_color(val: str) -> str:
                if isinstance(val, str) and "+" in val:
                    return "color: #3fb950; font-weight:600"
                elif isinstance(val, str) and "-" in val:
                    return "color: #f85149; font-weight:600"
                return ""

            st.dataframe(
                tl.style.applymap(_pnl_color, subset=["pnl", "pnl_pct"] if "pnl" in tl.columns else []),
                use_container_width=True, height=350
            )
            csv = trades.to_csv(index=False)
            st.download_button("⬇️ Download Trade Log CSV", csv, "trade_log.csv", "text/csv")
        else:
            st.info("Backtest completed, but 0 trades were executed. (Signals likely fired too recently to enter trades).")
    else:
        st.info("Run the backtest to see the trade log.")
