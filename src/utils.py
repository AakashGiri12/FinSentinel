"""
utils.py — Shared utilities for FinSentinel.

Provides:
  - Config loading (YAML → typed dataclass)
  - Centralized logging setup
  - Path helpers
  - Retry decorator
  - Rate-limiter
  - Common type aliases
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import yaml
from dotenv import load_dotenv

# ── Load .env early so every module sees the variables ───────────────────────
load_dotenv()

# ── Type aliases ─────────────────────────────────────────────────────────────
F = TypeVar("F", bound=Callable[..., Any])
PathLike = str | Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DateRangeConfig:
    """Date range for data fetching."""
    start: str
    end: str  # "today" is resolved at runtime


@dataclass(frozen=True)
class SentimentConfig:
    """FinBERT inference and aggregation parameters."""
    model_name: str
    buy_threshold: float
    sell_threshold: float
    momentum_threshold: float
    batch_size: int
    recency_weight_24h: float
    momentum_window_days: int


@dataclass(frozen=True)
class SignalsConfig:
    """Signal generation parameters."""
    consistency_days: int
    confidence_scale: bool


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting engine parameters."""
    initial_capital: float
    transaction_cost: float
    max_hold_days: int
    kelly_fraction: float


@dataclass(frozen=True)
class ScraperConfig:
    """HTTP scraping parameters."""
    rate_limit_seconds: float
    retry_attempts: int
    retry_backoff_factor: float
    yahoo_rss_base: str
    sec_edgar_base: str
    lookback_days: int
    output_dir: str
    newsdata_max_pages: int  # max pages to fetch from newsdata.io per ticker


@dataclass(frozen=True)
class PreprocessorConfig:
    """Text cleaning parameters."""
    min_article_length: int
    irrelevant_keywords: tuple[str, ...]
    keep_press_releases: bool


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    datefmt: str
    file: Optional[str]


@dataclass(frozen=True)
class AppConfig:
    """Root configuration — mirrors config.yaml exactly."""
    tickers: tuple[str, ...]
    date_range: DateRangeConfig
    sentiment: SentimentConfig
    signals: SignalsConfig
    backtest: BacktestConfig
    scraper: ScraperConfig
    preprocessor: PreprocessorConfig
    logging: LoggingConfig


# ─────────────────────────────────────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_CACHE: Optional[AppConfig] = None
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns:
        Path object pointing to the FinSentinel project root.
    """
    return _PROJECT_ROOT


def load_config(config_path: Optional[PathLike] = None) -> AppConfig:
    """Load and parse config.yaml into a typed AppConfig dataclass.

    Results are cached — subsequent calls return the same object.

    Args:
        config_path: Path to config.yaml. Defaults to <project_root>/config.yaml.

    Returns:
        Fully populated AppConfig instance.

    Raises:
        FileNotFoundError: If config_path does not exist.
        KeyError: If a required config key is missing.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = Path(config_path) if config_path else _PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    # Resolve "today" lazily at load time
    from datetime import date
    end_date = str(date.today()) if raw["date_range"]["end"] == "today" else raw["date_range"]["end"]

    cfg = AppConfig(
        tickers=tuple(raw["tickers"]),
        date_range=DateRangeConfig(
            start=raw["date_range"]["start"],
            end=end_date,
        ),
        sentiment=SentimentConfig(
            model_name=raw["sentiment"]["model_name"],
            buy_threshold=float(raw["sentiment"]["buy_threshold"]),
            sell_threshold=float(raw["sentiment"]["sell_threshold"]),
            momentum_threshold=float(raw["sentiment"]["momentum_threshold"]),
            batch_size=int(raw["sentiment"]["batch_size"]),
            recency_weight_24h=float(raw["sentiment"]["recency_weight_24h"]),
            momentum_window_days=int(raw["sentiment"]["momentum_window_days"]),
        ),
        signals=SignalsConfig(
            consistency_days=int(raw["signals"]["consistency_days"]),
            confidence_scale=bool(raw["signals"]["confidence_scale"]),
        ),
        backtest=BacktestConfig(
            initial_capital=float(raw["backtest"]["initial_capital"]),
            transaction_cost=float(raw["backtest"]["transaction_cost"]),
            max_hold_days=int(raw["backtest"]["max_hold_days"]),
            kelly_fraction=float(raw["backtest"]["kelly_fraction"]),
        ),
        scraper=ScraperConfig(
            rate_limit_seconds=float(raw["scraper"]["rate_limit_seconds"]),
            retry_attempts=int(raw["scraper"]["retry_attempts"]),
            retry_backoff_factor=float(raw["scraper"]["retry_backoff_factor"]),
            yahoo_rss_base=raw["scraper"]["yahoo_rss_base"],
            sec_edgar_base=raw["scraper"]["sec_edgar_base"],
            lookback_days=int(raw["scraper"]["lookback_days"]),
            output_dir=raw["scraper"]["output_dir"],
            newsdata_max_pages=int(raw["scraper"].get("newsdata_max_pages", 5)),
        ),
        preprocessor=PreprocessorConfig(
            min_article_length=int(raw["preprocessor"]["min_article_length"]),
            irrelevant_keywords=tuple(raw["preprocessor"]["irrelevant_keywords"]),
            keep_press_releases=bool(raw["preprocessor"]["keep_press_releases"]),
        ),
        logging=LoggingConfig(
            level=raw["logging"]["level"],
            format=raw["logging"]["format"],
            datefmt=raw["logging"]["datefmt"],
            file=raw["logging"].get("file"),
        ),
    )

    _CONFIG_CACHE = cfg
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

_LOGGERS_INITIALIZED: set[str] = set()


def get_logger(name: str, config_path: Optional[PathLike] = None) -> logging.Logger:
    """Return a configured logger for the given module name.

    Sets up a rotating file handler (if configured) and a stream handler.
    Safe to call multiple times — initialises only once per name.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        config_path: Optional path to config.yaml; uses default if None.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    if name in _LOGGERS_INITIALIZED:
        return logging.getLogger(name)

    cfg = load_config(config_path)
    log_cfg = cfg.logging

    # Resolve log level — allow env override
    level_name: str = os.environ.get("LOG_LEVEL", log_cfg.level).upper()
    level: int = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging

    formatter = logging.Formatter(fmt=log_cfg.format, datefmt=log_cfg.datefmt)

    # ── Stream handler (always on) ────────────────────────────────────────
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ── Rotating file handler (optional) ─────────────────────────────────
    if log_cfg.file:
        log_path = _PROJECT_ROOT / log_cfg.file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS_INITIALIZED.add(name)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Retry Decorator
# ─────────────────────────────────────────────────────────────────────────────

def retry(
    attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a function on failure with exponential back-off.

    Args:
        attempts: Maximum number of tries (including the first call).
        backoff_factor: Multiplier for sleep time between retries.
            Sleep time = backoff_factor ** (attempt_number - 1) seconds.
        exceptions: Tuple of exception types that trigger a retry.

    Returns:
        Decorated function that retries on specified exceptions.

    Example::

        @retry(attempts=3, backoff_factor=2.0, exceptions=(requests.HTTPError,))
        def fetch(url: str) -> str:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            last_exc: Exception | None = None

            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    sleep_time = backoff_factor ** (attempt - 1)
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. "
                        "Retrying in %.1fs...",
                        attempt,
                        attempts,
                        func.__qualname__,
                        exc,
                        sleep_time,
                    )
                    if attempt < attempts:
                        time.sleep(sleep_time)

            logger.error(
                "All %d attempts failed for %s. Last error: %s",
                attempts,
                func.__qualname__,
                last_exc,
            )
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Simple token-bucket rate limiter for HTTP requests.

    Ensures at least ``min_interval`` seconds elapse between calls.

    Args:
        min_interval: Minimum seconds between successive calls.

    Example::

        limiter = RateLimiter(min_interval=1.0)
        for url in urls:
            limiter.wait()
            response = requests.get(url)
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Block until the minimum interval since the last call has elapsed."""
        elapsed = time.monotonic() - self._last_call
        to_wait = self._min_interval - elapsed
        if to_wait > 0:
            time.sleep(to_wait)
        self._last_call = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Path Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: PathLike) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        Resolved Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_data_path(relative: str) -> Path:
    """Resolve a data path relative to the project root.

    Args:
        relative: Relative path string, e.g. ``"data/raw"``.

    Returns:
        Absolute Path for the given relative location.
    """
    return _PROJECT_ROOT / relative


# ─────────────────────────────────────────────────────────────────────────────
# Misc Helpers
# ─────────────────────────────────────────────────────────────────────────────

def chunked(iterable: list[Any], size: int) -> list[list[Any]]:
    """Split a list into chunks of at most ``size`` elements.

    Args:
        iterable: The input list to split.
        size: Maximum number of elements per chunk.

    Returns:
        List of sub-lists, each of length <= size.

    Example::

        chunked([1, 2, 3, 4, 5], 2) → [[1, 2], [3, 4], [5]]
    """
    return [iterable[i : i + size] for i in range(0, len(iterable), size)]


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the closed interval [lo, hi].

    Args:
        value: The number to clamp.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        Clamped float.
    """
    return max(lo, min(value, hi))
