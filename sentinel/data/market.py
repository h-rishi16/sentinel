"""
Market data fetcher for Sentinel.

Fetches historical price data from Yahoo Finance via yfinance, performs basic
data quality checks, and stores results as Parquet files.

Design decisions:
    - yfinance is used because it provides free, reliable historical daily prices
      for equities and indices. It is sufficient for a research-scale platform.
    - Data is stored as Parquet (columnar, compressed) rather than CSV because
      Parquet preserves dtypes, handles timestamps correctly, and is 5-10x
      smaller on disk.
    - We fetch adjusted close prices, which account for stock splits and
      dividends. This is critical — using unadjusted prices would produce
      spurious returns on split dates.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default storage location for raw market data
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    data_dir: Optional[Path] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        Stock/ETF ticker symbols (e.g., ["AAPL", "MSFT", "SPY"]).
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    data_dir : Path, optional
        Directory to save Parquet file. Defaults to data/raw/.
    save : bool
        Whether to persist the data to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per ticker, containing
        adjusted close prices. Only trading days are included.

    Raises
    ------
    ValueError
        If no data is returned for any ticker, or if critical quality
        checks fail.
    """
    logger.info(f"Fetching prices for {tickers} from {start} to {end}")

    # Fetch from Yahoo Finance
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex DataFrame when multiple tickers are passed.
    # We extract just the "Close" prices (which are adjusted close when
    # auto_adjust=True).
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker case — raw already has OHLCV columns
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # --- Data Quality Checks ---
    quality_report = validate_prices(prices, tickers)
    if quality_report["critical_errors"]:
        raise ValueError(
            f"Critical data quality errors: {quality_report['critical_errors']}"
        )

    for warning in quality_report["warnings"]:
        logger.warning(warning)

    logger.info(
        f"Fetched {len(prices)} trading days for {len(prices.columns)} assets. "
        f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}"
    )

    # Persist to Parquet
    if save:
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        filepath = data_dir / "prices.parquet"
        prices.to_parquet(filepath, engine="pyarrow")
        logger.info(f"Saved prices to {filepath}")

    return prices


def validate_prices(
    prices: pd.DataFrame, expected_tickers: list[str]
) -> dict:
    """
    Run data quality checks on fetched price data.

    Checks:
        1. All requested tickers are present.
        2. No ticker has >5% missing values (holidays/delisted).
        3. No negative prices.
        4. No suspiciously large single-day moves (>50%).
        5. Minimum 252 trading days (1 year) for meaningful analysis.

    Returns
    -------
    dict
        {"critical_errors": [...], "warnings": [...], "stats": {...}}
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    # Check 1: All tickers present
    missing_tickers = set(expected_tickers) - set(prices.columns)
    if missing_tickers:
        errors.append(f"Missing data for tickers: {missing_tickers}")

    # Check 2: Missing values
    for col in prices.columns:
        missing_pct = prices[col].isna().mean()
        stats[f"{col}_missing_pct"] = missing_pct
        if missing_pct > 0.05:
            errors.append(
                f"{col} has {missing_pct:.1%} missing values (threshold: 5%)"
            )
        elif missing_pct > 0.0:
            warnings.append(
                f"{col} has {missing_pct:.1%} missing values — will forward-fill"
            )

    # Check 3: Negative prices
    for col in prices.columns:
        if (prices[col].dropna() < 0).any():
            errors.append(f"{col} contains negative prices")

    # Check 4: Extreme single-day moves
    if not prices.empty:
        pct_changes = prices.pct_change().dropna()
        for col in pct_changes.columns:
            max_move = pct_changes[col].abs().max()
            if max_move > 0.50:
                warnings.append(
                    f"{col} has a single-day move of {max_move:.1%} — "
                    "verify this is not a data error (stock split, etc.)"
                )

    # Check 5: Minimum observation count
    stats["trading_days"] = len(prices)
    if len(prices) < 252:
        warnings.append(
            f"Only {len(prices)} trading days. At least 252 (1 year) recommended "
            "for meaningful volatility and correlation estimation."
        )

    return {"critical_errors": errors, "warnings": warnings, "stats": stats}


def load_prices(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load previously saved price data from Parquet."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    filepath = data_dir / "prices.parquet"
    if not filepath.exists():
        raise FileNotFoundError(
            f"No saved price data found at {filepath}. Run fetch_prices() first."
        )
    return pd.read_parquet(filepath, engine="pyarrow")


def forward_fill_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing prices (e.g., holidays where one exchange is open
    but another is closed).

    Only forward-fills — does NOT backfill, because backfilling introduces
    look-ahead bias.
    """
    filled = prices.ffill()
    # Drop any leading NaN rows (before the first valid observation for all assets)
    filled = filled.dropna()
    return filled
