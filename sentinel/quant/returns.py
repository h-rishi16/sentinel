"""
Financial returns calculator for Sentinel.

This module computes simple and logarithmic returns from price data.
It is the mathematical foundation upon which all subsequent quantitative
calculations (volatility, VaR, Expected Shortfall) are built.

Mathematical Background
-----------------------
Simple Return:
    R_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1

    Properties:
    - Bounded below by -1 (you can't lose more than 100% with equities)
    - Portfolio simple return = weighted sum of asset simple returns
    - NOT additive over time: R(t1→t3) ≠ R(t1→t2) + R(t2→t3)

Log Return:
    r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Properties:
    - Additive over time: r(t1→t3) = r(t1→t2) + r(t2→t3)
    - Approximately equal to simple return for small values
    - More likely to be normally distributed (a key assumption in many models)
    - Portfolio log return ≠ weighted sum of asset log returns

    Relationship:
    r_t = ln(1 + R_t)
    R_t = exp(r_t) - 1

When to use which:
    - Use log returns for: volatility estimation, GARCH modeling,
      Monte Carlo simulation, statistical tests
    - Use simple returns for: portfolio return aggregation, reporting,
      P&L calculations
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple (arithmetic) returns from price data.

    R_t = P_t / P_{t-1} - 1

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with DatetimeIndex and one column per asset.
        Must be sorted chronologically and contain no NaN values.

    Returns
    -------
    pd.DataFrame
        Simple returns. First row is dropped (no return for first day).

    Raises
    ------
    ValueError
        If prices contain NaN, non-positive values, or are not sorted.
    """
    _validate_price_input(prices)

    returns = prices.pct_change().iloc[1:]  # Drop first NaN row

    logger.info(
        f"Computed simple returns: {len(returns)} observations, "
        f"{len(returns.columns)} assets"
    )
    return returns


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute logarithmic (continuously compounded) returns from price data.

    r_t = ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with DatetimeIndex and one column per asset.
        Must be sorted chronologically, contain no NaN values,
        and all prices must be strictly positive.

    Returns
    -------
    pd.DataFrame
        Log returns. First row is dropped.
    """
    _validate_price_input(prices)

    returns = np.log(prices / prices.shift(1)).iloc[1:]  # Drop first NaN row

    logger.info(
        f"Computed log returns: {len(returns)} observations, "
        f"{len(returns.columns)} assets"
    )
    return returns


def cumulative_returns(returns: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Compute cumulative returns from a return series.

    For simple returns:
        C_t = ∏(1 + R_i) - 1  (compounding)

    For log returns:
        C_t = exp(Σ r_i) - 1   (summing then exponentiating)

    Parameters
    ----------
    returns : pd.DataFrame
        Return series (simple or log).
    method : str
        "simple" for arithmetic returns, "log" for log returns.

    Returns
    -------
    pd.DataFrame
        Cumulative return series, starting from 0.
    """
    if method == "simple":
        # Compound: (1+R_1)(1+R_2)...(1+R_t) - 1
        cumulative = (1 + returns).cumprod() - 1
    elif method == "log":
        # Sum log returns, then exponentiate
        cumulative = np.exp(returns.cumsum()) - 1
    else:
        raise ValueError(f"Method must be 'simple' or 'log', got '{method}'")

    return cumulative


def annualization_factor(frequency: str = "daily") -> float:
    """
    Return the annualization multiplier for a given data frequency.

    For returns: multiply by factor
    For volatility: multiply by sqrt(factor)

    Parameters
    ----------
    frequency : str
        "daily" (252 trading days), "weekly" (52), "monthly" (12).

    Returns
    -------
    float
        Number of periods per year.
    """
    factors = {
        "daily": 252.0,
        "weekly": 52.0,
        "monthly": 12.0,
    }
    if frequency not in factors:
        raise ValueError(
            f"Unknown frequency '{frequency}'. Use: {list(factors.keys())}"
        )
    return factors[frequency]


def return_statistics(
    returns: pd.DataFrame,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Compute summary statistics for a return series.

    Calculates: mean, std, skewness, kurtosis, min, max, and their
    annualized equivalents where appropriate.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series (typically log returns).
    frequency : str
        Data frequency for annualization.

    Returns
    -------
    pd.DataFrame
        Statistics table with one row per asset.
    """
    factor = annualization_factor(frequency)

    stats = pd.DataFrame(
        {
            "mean_daily": returns.mean(),
            "std_daily": returns.std(),
            "mean_annualized": returns.mean() * factor,
            "std_annualized": returns.std() * np.sqrt(factor),
            "skewness": returns.skew(),
            "excess_kurtosis": returns.kurtosis(),  # pandas uses excess kurtosis
            "min": returns.min(),
            "max": returns.max(),
            "count": returns.count(),
        }
    )

    # Sharpe ratio (assumes 0% risk-free rate for now)
    # NOTE: This is a simplification. A proper Sharpe ratio should subtract
    # the risk-free rate. We will add this when we integrate FRED data for
    # Treasury yields.
    stats["sharpe_ratio"] = stats["mean_annualized"] / stats["std_annualized"]

    return stats


def _validate_price_input(prices: pd.DataFrame) -> None:
    """
    Validate that price data is suitable for return calculation.

    Raises ValueError with a specific message if any check fails.
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty")

    if prices.isna().any().any():
        na_counts = prices.isna().sum()
        bad_cols = na_counts[na_counts > 0]
        raise ValueError(
            f"Price data contains NaN values. "
            f"Missing counts: {bad_cols.to_dict()}. "
            f"Clean the data before computing returns."
        )

    if (prices <= 0).any().any():
        raise ValueError(
            "Price data contains non-positive values. "
            "All prices must be strictly positive for return calculation."
        )

    # Check chronological ordering
    if not prices.index.is_monotonic_increasing:
        raise ValueError(
            "Price data is not sorted chronologically. "
            "Sort by date before computing returns."
        )

    if len(prices) < 2:
        raise ValueError(
            "Need at least 2 price observations to compute returns."
        )
