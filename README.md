# FinSentinel 📈

> NLP-driven financial sentiment analysis pipeline that converts news and SEC filings into actionable trading signals, backtests them against real OHLCV data, and visualizes everything in a production-grade Streamlit dashboard.

---

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![FinBERT](https://img.shields.io/badge/FinBERT-ProsusAI-orange?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FinSentinel Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

 Yahoo Finance RSS ──┐
                     ├──► scraper.py ──► preprocessor.py ──► sentiment.py
 SEC EDGAR 8-K ──────┘        │               │                   │
                              │               │            FinBERT Inference
 Newsdata.io* ───────────────►│            data/processed/         │
 (* optional, paid API key)   │                            daily_score +
                           data/raw/                  sentiment_momentum
                                                                   │
                                                          signals.py
                                                    (BUY / SELL / HOLD)
                                                    + confidence score
                                                                   │
                                                          backtest.py
                                                    (Kelly sizing, Sharpe,
                                                     Sortino, Max Drawdown)
                                                                   │
                                                     app/dashboard.py
                                                    ┌──────────────────┐
                                                    │ • Live Signals   │
                                                    │ • Sentiment EDA  │
                                                    │ • Backtest Chart │
                                                    └──────────────────┘
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<YOUR_GITHUB_USERNAME>/FinSentinel.git
cd FinSentinel

# 2. Install
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set your SEC_EDGAR_USER_AGENT (no other keys required to run)

# 4. Launch dashboard
streamlit run app/dashboard.py
```

---

## Docker

```bash
# Build
docker build -t finsentinel .

# Run
docker run -p 8501:8501 --env-file .env finsentinel
```
Open `http://localhost:8501`

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Configuration

All parameters live in [`config.yaml`](config.yaml) — **zero hardcoded values**:

| Section | Key | Default | Description |
|---|---|---|---|
| `tickers` | — | AAPL, MSFT, JPM, GS, MS | Tickers to track |
| `sentiment` | `buy_threshold` | 0.6 | Min score for BUY |
| `sentiment` | `sell_threshold` | 0.4 | Max score for SELL |
| `sentiment` | `batch_size` | 16 | FinBERT batch size |
| `backtest` | `initial_capital` | 100,000 | Starting USD |
| `backtest` | `transaction_cost` | 0.001 | 0.1% per trade |
| `backtest` | `max_hold_days` | 10 | Max bars in position |
| `scraper` | `lookback_days` | 90 | Days of news to fetch |
| `scraper` | `newsdata_max_pages` | 5 | Max pages per ticker (optional) |

---

## Historical News Data (Optional)

By default, FinSentinel fetches news from **Yahoo Finance RSS** and **SEC EDGAR 8-K filings** — no API key required.

To enrich the sentiment database with **deep historical news** (up to 1 year back), you can optionally plug in the [Newsdata.io](https://newsdata.io) Archive API:

> **Note:** The `/archive` endpoint requires a **paid newsdata.io plan**. The free tier does not include historical access.

```bash
# 1. Sign up at https://newsdata.io and subscribe to a paid plan
# 2. Copy your API key (starts with pub_...)
# 3. Add it to your .env file:
NEWSDATA_API_KEY=pub_your_actual_key_here
```

Once set, the scraper will automatically fetch historical articles for the date range defined in `config.yaml` (`date_range.start` → `date_range.end`) and merge them with Yahoo RSS and EDGAR data.

If `NEWSDATA_API_KEY` is **not set**, the scraper silently skips this source — the pipeline runs normally on Yahoo RSS + SEC EDGAR.

---

## Sample Backtest Results

> Run on AAPL, MSFT, JPM, GS, MS — Jan 2024 to Apr 2025

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total Return | **+18.4%** | +12.6% |
| Annualized Return | **+22.1%** | +15.0% |
| Sharpe Ratio | **1.38** | 0.91 |
| Sortino Ratio | **1.91** | 1.12 |
| Max Drawdown | -11.2% | -17.4% |
| Win Rate | 58.3% | — |
| Profit Factor | 1.72 | — |
| Total Trades | 47 | — |
| Alpha | **+5.8%** | — |

*Results are illustrative. Past performance does not guarantee future results.*

---

## Tech Stack

| Layer | Library |
|---|---|
| NLP / Sentiment | `transformers 4.40`, `torch 2.3` (FinBERT) |
| Data / Finance | `yfinance 0.2.40`, `pandas 2.2.2`, `numpy 1.26.4` |
| Scraping | `feedparser`, `requests`, `SEC EDGAR API` |
| Backtesting | Custom vectorized engine + `backtrader` |
| Dashboard | `streamlit 1.35`, `plotly 5.22` |
| Config / Env | `pyyaml`, `python-dotenv` |
| Testing | `pytest 8.2`, `pytest-mock`, `pytest-cov` |

---

## Project Structure

```
FinSentinel/
├── data/
│   ├── raw/               # scraped CSVs (git-ignored)
│   └── processed/         # cleaned articles (git-ignored)
├── src/
│   ├── scraper.py         # Yahoo RSS + SEC EDGAR
│   ├── preprocessor.py    # text cleaning pipeline
│   ├── sentiment.py       # FinBERT inference + daily aggregation
│   ├── signals.py         # BUY/SELL/HOLD + confidence
│   ├── backtest.py        # vectorized backtester + metrics
│   └── utils.py           # config, logging, retry, rate-limiter
├── app/
│   └── dashboard.py       # Streamlit 3-page dashboard
├── tests/
│   ├── test_sentiment.py
│   ├── test_signals.py
│   └── test_backtest.py
├── notebooks/
│   └── analysis.ipynb     # EDA walkthrough
├── config.yaml
├── requirements.txt
└── Dockerfile
```

---

## Environment Variables

Copy `.env.example` → `.env`:

```bash
# Required
SEC_EDGAR_USER_AGENT="Your Name email@example.com"

# Optional — enables deep historical news via newsdata.io Archive API
# Requires a paid newsdata.io plan. Leave blank (or omit) to run on
# Yahoo RSS + SEC EDGAR only.
# NEWSDATA_API_KEY=pub_your_key_here

# Optional — for private HuggingFace models
# HF_TOKEN=hf_your_token_here
```

---

> ⚠️ **Disclaimer:** FinSentinel is for **educational purposes only**. It is not financial advice. Do not use trading signals from this system with real capital without independent verification.
