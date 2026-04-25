"""
scraper.py — Financial news and SEC filing scraper for FinSentinel.

Data sources (default — no API key required):
  1. Yahoo Finance RSS feed — headlines + summaries for each ticker.
  2. SEC EDGAR full-text search API — 8-K filings (earnings surprises).

Optional historical enrichment:
  3. Newsdata.io Archive API — deep historical news (up to 1 year back).
     Requires a paid newsdata.io API key set as NEWSDATA_API_KEY in .env.
     If the key is absent the scraper silently skips this source.
     See README.md » "Historical News Data" for setup instructions.

Output: pandas DataFrame[ticker, date, headline, summary, source]
        persisted as  data/raw/{ticker}_{YYYY-MM-DD}.csv
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import feedparser
import pandas as pd
import requests

from src.utils import (
    AppConfig,
    RateLimiter,
    ensure_dir,
    get_logger,
    load_config,
    resolve_data_path,
    retry,
)

logger = get_logger(__name__)

# ── DataFrame schema ─────────────────────────────────────────────────────────
_SCHEMA: dict[str, str] = {
    "ticker": "string",
    "date": "datetime64[ns, UTC]",
    "headline": "string",
    "summary": "string",
    "source": "string",
}


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance RSS
# ─────────────────────────────────────────────────────────────────────────────

def _build_yahoo_url(ticker: str, base: str) -> str:
    """Construct the Yahoo Finance RSS URL for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        base: Base RSS URL from config.

    Returns:
        Full RSS feed URL string.
    """
    return f"{base}?s={ticker}&region=US&lang=en-US"


def _parse_rss_entry(entry: feedparser.FeedParserDict, ticker: str) -> Optional[dict]:
    """Extract normalised fields from a single RSS feed entry.

    Args:
        entry: A feedparser entry object.
        ticker: Ticker symbol to tag the record.

    Returns:
        Dict with keys matching ``_SCHEMA``, or None if the entry is malformed.
    """
    try:
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published is None:
            return None

        dt = datetime(*published[:6], tzinfo=timezone.utc)
        headline = (entry.get("title") or "").strip()
        summary = (entry.get("summary") or entry.get("description") or "").strip()

        if not headline:
            return None

        return {
            "ticker": ticker,
            "date": dt,
            "headline": headline,
            "summary": summary,
            "source": "yahoo_rss",
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skipping malformed RSS entry: %s", exc)
        return None


def fetch_yahoo_news(
    ticker: str,
    cfg: AppConfig,
    limiter: RateLimiter,
) -> list[dict]:
    """Fetch news headlines for one ticker from Yahoo Finance RSS.

    Args:
        ticker: Stock ticker symbol.
        cfg: Application configuration object.
        limiter: Shared rate limiter instance.

    Returns:
        List of article dicts ready for DataFrame construction.
    """
    url = _build_yahoo_url(ticker, cfg.scraper.yahoo_rss_base)
    cutoff: datetime = datetime.now(tz=timezone.utc) - timedelta(
        days=cfg.scraper.lookback_days
    )

    @retry(
        attempts=cfg.scraper.retry_attempts,
        backoff_factor=cfg.scraper.retry_backoff_factor,
        exceptions=(Exception,),
    )
    def _fetch() -> feedparser.FeedParserDict:
        limiter.wait()
        logger.debug("Fetching Yahoo RSS: %s", url)
        feed = feedparser.parse(url)
        if feed.get("bozo") and not feed.get("entries"):
            raise ValueError(f"feedparser bozo flag set for {url}: {feed.bozo_exception}")
        return feed

    try:
        feed = _fetch()
    except Exception as exc:
        logger.error("Failed to fetch Yahoo RSS for %s: %s", ticker, exc)
        return []

    records: list[dict] = []
    for entry in feed.get("entries", []):
        record = _parse_rss_entry(entry, ticker)
        if record and record["date"] >= cutoff:
            records.append(record)

    logger.info("Yahoo RSS: %d articles fetched for %s", len(records), ticker)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR 8-K Scraper
# ─────────────────────────────────────────────────────────────────────────────

# Map tickers → CIK numbers for EDGAR (most common fintech/bank tickers)
_TICKER_TO_CIK: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "JPM":  "0000019617",
    "GS":   "0000886982",
    "MS":   "0000895421",
    "AMZN": "0001018724",
    "GOOGL":"0001652044",
    "META": "0001326801",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
}


def _edgar_user_agent() -> str:
    """Read the SEC EDGAR user-agent from environment (required by EDGAR TOS).

    Returns:
        User-agent string for EDGAR HTTP requests.
    """
    import os
    agent = os.environ.get("SEC_EDGAR_USER_AGENT", "FinSentinel Research finsentinel@example.com")
    return agent


def _fetch_edgar_filings(
    cik: str,
    ticker: str,
    cutoff: datetime,
    cfg: AppConfig,
    limiter: RateLimiter,
    session: requests.Session,
) -> list[dict]:
    """Fetch 8-K filing metadata from SEC EDGAR submissions API.

    Args:
        cik: SEC CIK number (zero-padded 10-digit string).
        ticker: Ticker symbol to tag records.
        cutoff: Earliest date to include (UTC-aware).
        cfg: Application configuration.
        limiter: Shared rate limiter.
        session: Shared requests Session.

    Returns:
        List of article-style dicts for EDGAR 8-K filings.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": _edgar_user_agent(), "Accept-Encoding": "gzip, deflate"}

    @retry(
        attempts=cfg.scraper.retry_attempts,
        backoff_factor=cfg.scraper.retry_backoff_factor,
        exceptions=(requests.RequestException, ValueError),
    )
    def _get() -> dict:
        limiter.wait()
        resp = session.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    try:
        data = _get()
    except Exception as exc:
        logger.error("EDGAR fetch failed for CIK %s (%s): %s", cik, ticker, exc)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    dates: list[str] = recent.get("filingDate", [])
    accessions: list[str] = recent.get("accessionNumber", [])
    descriptions: list[str] = recent.get("primaryDocDescription", [])

    records: list[dict] = []
    for form, date_str, accession, desc in zip(forms, dates, accessions, descriptions):
        if form not in ("8-K", "8-K/A"):
            continue
        try:
            filing_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if filing_date < cutoff:
            continue

        accession_clean = accession.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
            f"/{accession_clean}/{accession}.txt"
        )
        headline = f"{ticker} SEC 8-K Filing: {desc or 'Earnings / Material Event'}"
        summary = (
            f"SEC EDGAR 8-K filing for {ticker} (CIK: {cik}). "
            f"Filed: {date_str}. Accession: {accession}. "
            f"Document: {filing_url}"
        )
        records.append(
            {
                "ticker": ticker,
                "date": filing_date,
                "headline": headline,
                "summary": summary,
                "source": "sec_edgar_8k",
            }
        )

    logger.info("EDGAR: %d 8-K filings fetched for %s", len(records), ticker)
    return records


def fetch_edgar_news(
    ticker: str,
    cfg: AppConfig,
    limiter: RateLimiter,
    session: requests.Session,
) -> list[dict]:
    """Public interface to fetch SEC EDGAR filings for a ticker.

    Gracefully skips tickers without a known CIK mapping.

    Args:
        ticker: Stock ticker symbol.
        cfg: Application configuration.
        limiter: Shared rate limiter.
        session: Shared requests Session.

    Returns:
        List of article-style dicts, possibly empty.
    """
    cik = _TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        logger.warning("No CIK mapping for ticker %s — skipping EDGAR.", ticker)
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=cfg.scraper.lookback_days)
    return _fetch_edgar_filings(cik, ticker, cutoff, cfg, limiter, session)


# ─────────────────────────────────────────────────────────────────────────────
# Newsdata.io Historical Archive (optional — requires paid API key)
# ─────────────────────────────────────────────────────────────────────────────

# Archive endpoint — requires a paid newsdata.io plan.
# Set NEWSDATA_API_KEY in .env to enable; scraper skips this source if absent.
_NEWSDATA_BASE = "https://newsdata.io/api/1/archive"

# Map ticker → search query for newsdata.io (company name is more reliable)
_TICKER_TO_QUERY: dict[str, str] = {
    "AAPL":  "Apple stock",
    "MSFT":  "Microsoft stock",
    "JPM":   "JPMorgan stock",
    "GS":    "Goldman Sachs stock",
    "MS":    "Morgan Stanley stock",
    "AMZN":  "Amazon stock",
    "GOOGL": "Google Alphabet stock",
    "META":  "Meta stock",
    "NVDA":  "Nvidia stock",
    "TSLA":  "Tesla stock",
}


def fetch_newsdata_historical(
    ticker: str,
    start_date: str,
    end_date: str,
    cfg: AppConfig,
    limiter: RateLimiter,
    session: requests.Session,
) -> list[dict]:
    """Fetch historical news from newsdata.io Archive API for a ticker.

    This function is **optional** — it is skipped silently when
    ``NEWSDATA_API_KEY`` is not set in the environment.  The default pipeline
    runs on Yahoo Finance RSS + SEC EDGAR without any API key.

    To enable historical enrichment:
      1. Sign up at https://newsdata.io (paid plan required for /archive).
      2. Add ``NEWSDATA_API_KEY=pub_your_key`` to your ``.env`` file.
      3. Re-run the scraper; articles from ``start_date`` to ``end_date``
         will be fetched and merged into the sentiment database.

    Paginates through result pages using the ``nextPage`` cursor up to
    ``newsdata_max_pages`` pages (configurable in config.yaml).

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date string ``"YYYY-MM-DD"`` (inclusive).
        end_date: End date string ``"YYYY-MM-DD"`` (inclusive).
        cfg: AppConfig instance.
        limiter: Shared rate limiter.
        session: Shared requests Session.

    Returns:
        List of article dicts matching the standard ``_SCHEMA``.
        Returns an empty list if the API key is absent or the request fails.
    """
    import os
    api_key = os.environ.get("NEWSDATA_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "NEWSDATA_API_KEY not set — skipping historical news for %s. "
            "Using Yahoo RSS + SEC EDGAR only. See README for setup.",
            ticker,
        )
        return []

    query = _TICKER_TO_QUERY.get(ticker.upper(), f"{ticker} stock")
    records: list[dict] = []
    next_page: str | None = None
    page_num = 0
    max_pages = cfg.scraper.newsdata_max_pages

    while page_num < max_pages:
        page_num += 1
        params: dict = {
            "apikey":    api_key,
            "q":         query,
            "language":  "en",
            "from_date": start_date,
            "to_date":   end_date,
        }
        if next_page:
            params["page"] = next_page

        @retry(
            attempts=cfg.scraper.retry_attempts,
            backoff_factor=cfg.scraper.retry_backoff_factor,
            exceptions=(requests.RequestException, ValueError),
        )
        def _get() -> dict:
            limiter.wait()
            resp = session.get(_NEWSDATA_BASE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                raise ValueError(f"Newsdata API error: {data.get('results', data)}")
            return data

        try:
            data = _get()
        except Exception as exc:
            logger.error("Newsdata fetch failed for %s (page %d): %s", ticker, page_num, exc)
            break

        articles = data.get("results", []) or []
        for art in articles:
            pub_date = art.get("pubDate") or art.get("pubDateTZ")
            if not pub_date:
                continue
            try:
                # newsdata returns "YYYY-MM-DD HH:MM:SS" format
                dt = datetime.strptime(pub_date[:19], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

            headline = (art.get("title") or "").strip()
            summary  = (art.get("description") or art.get("content") or "").strip()
            source   = art.get("source_id") or "newsdata"

            if not headline:
                continue

            records.append({
                "ticker":   ticker,
                "date":     dt,
                "headline": headline,
                "summary":  summary,
                "source":   f"newsdata_{source}",
            })

        next_page = data.get("nextPage")
        logger.info(
            "Newsdata.io: page %d fetched %d articles for %s (nextPage=%s)",
            page_num, len(articles), ticker, next_page,
        )

        if not next_page:
            break  # no more pages

    logger.info(
        "Newsdata.io: %d total articles fetched for %s (%s → %s)",
        len(records), ticker, start_date, end_date,
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame Assembly + Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert a list of article dicts into a typed DataFrame.

    Args:
        records: List of dicts with keys matching ``_SCHEMA``.

    Returns:
        DataFrame with explicit dtypes and UTC-aware datetime column.
    """
    if not records:
        df = pd.DataFrame(columns=list(_SCHEMA.keys()))
    else:
        df = pd.DataFrame(records)

    # Enforce datetime type and UTC timezone
    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)

    # Enforce string dtypes
    for col in ("ticker", "headline", "summary", "source"):
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df.reset_index(drop=True)


def _save_raw(df: pd.DataFrame, ticker: str, cfg: AppConfig) -> Path:
    """Persist raw DataFrame to CSV in data/raw/{ticker}_{date}.csv.

    Args:
        df: DataFrame to save.
        ticker: Ticker symbol (used in filename).
        cfg: Application configuration (provides output_dir).

    Returns:
        Path to the saved CSV file.
    """
    out_dir = ensure_dir(resolve_data_path(cfg.scraper.output_dir))
    today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    path = out_dir / f"{ticker}_{today_str}.csv"
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved %d rows → %s", len(df), path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def scrape_ticker(
    ticker: str,
    cfg: Optional[AppConfig] = None,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
    save: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Scrape Yahoo RSS + SEC EDGAR + Newsdata.io for one ticker.

    When ``start_date``/``end_date`` are provided (or the config date_range
    spans more than ``lookback_days``), the Newsdata.io Archive API is used
    to supplement Yahoo RSS with historical articles.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        cfg: AppConfig instance; loaded from config.yaml if None.
        limiter: Shared RateLimiter; created fresh if None.
        session: Shared requests Session; created fresh if None.
        save: If True, persist raw CSV to data/raw/.
        start_date: Override start date ``"YYYY-MM-DD"`` for newsdata fetch.
        end_date: Override end date ``"YYYY-MM-DD"`` for newsdata fetch.

    Returns:
        DataFrame[ticker, date, headline, summary, source] for the given ticker.
    """
    cfg = cfg or load_config()
    limiter = limiter or RateLimiter(cfg.scraper.rate_limit_seconds)
    session = session or requests.Session()

    logger.info("── Scraping %s ──", ticker)

    yahoo_records = fetch_yahoo_news(ticker, cfg, limiter)
    edgar_records = fetch_edgar_news(ticker, cfg, limiter, session)

    # ── Newsdata.io historical enrichment ────────────────────────────────────
    _start = start_date or cfg.date_range.start
    _end   = end_date   or cfg.date_range.end
    newsdata_records = fetch_newsdata_historical(
        ticker, _start, _end, cfg, limiter, session
    )

    all_records = yahoo_records + edgar_records + newsdata_records
    df = _to_dataframe(all_records)

    if save and not df.empty:
        _save_raw(df, ticker, cfg)

    return df


def scrape_all(
    cfg: Optional[AppConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Scrape all tickers defined in config and return a concatenated DataFrame.

    Args:
        cfg: AppConfig instance; loaded from config.yaml if None.
        save: If True, save per-ticker CSVs to data/raw/.

    Returns:
        Combined DataFrame for all configured tickers, sorted by date descending.
    """
    cfg = cfg or load_config()
    limiter = RateLimiter(cfg.scraper.rate_limit_seconds)
    session = requests.Session()
    session.headers.update({"User-Agent": _edgar_user_agent()})

    frames: list[pd.DataFrame] = []
    for ticker in cfg.tickers:
        try:
            df = scrape_ticker(ticker, cfg=cfg, limiter=limiter, session=session, save=save)
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unhandled error scraping %s: %s", ticker, exc)

    if not frames:
        logger.warning("No data scraped for any ticker.")
        return _to_dataframe([])

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("date", ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        "Scraping complete. Total articles: %d across %d tickers.",
        len(combined),
        len(cfg.tickers),
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinSentinel scraper CLI")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override tickers from config (e.g. --tickers AAPL MSFT)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist raw CSV files",
    )
    args = parser.parse_args()

    _cfg = load_config()
    if args.tickers:
        import dataclasses
        _cfg = dataclasses.replace(_cfg, tickers=tuple(args.tickers))

    result_df = scrape_all(cfg=_cfg, save=not args.no_save)
    print(result_df.to_string(max_rows=20))
