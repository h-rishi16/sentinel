"""
Portfolio analytics for Sentinel.

This module combines individual asset data into portfolio-level risk
metrics. It is the bridge between "asset analysis" and "portfolio risk."

Mathematical Background
-----------------------
Modern Portfolio Theory (Markowitz, 1952):

    Portfolio Return:
        R_p = w^T R = Σ w_i R_i
        (weighted sum of simple asset returns)

    Portfolio Variance:
        σ²_p = w^T Σ w
        (where Σ is the covariance matrix, w is the weight vector)

    Portfolio Volatility:
        σ_p = √(w^T Σ w)

    Key insight: σ_p ≤ Σ w_i σ_i
    Portfolio volatility is ALWAYS less than or equal to the weighted
    average of individual volatilities. Equality only when all correlations
    are +1. This is diversification — the central insight of portfolio theory.

Covariance Matrix Estimation:
    Sample covariance: Σ = (1/(n-1)) X^T X  (X = centered returns)

    This is unbiased but noisy when:
    - Number of assets approaches number of observations
    - Asset correlations are unstable over time

    We implement both sample covariance and Ledoit-Wolf shrinkage.
    Ledoit-Wolf shrinks the sample covariance toward a structured target
    (scaled identity), reducing estimation error at the cost of some bias.
    For portfolios with >10 assets, Ledoit-Wolf is almost always better.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.quant.returns import simple_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PortfolioDefinition:
    """
    Defines a portfolio: which assets, how much of each.

    Parameters
    ----------
    weights : dict[str, float]
        Mapping of asset name → portfolio weight.
        Weights must sum to 1.0 (±0.01 tolerance).
        For long-only portfolios, all weights must be non-negative.
    name : str
        Human-readable portfolio name.
    """

    weights: dict[str, float]
    name: str = "Portfolio"

    def __post_init__(self) -> None:
        """Validate weights on construction."""
        if not self.weights:
            raise ValueError("Portfolio must have at least one asset")

        weight_sum = sum(self.weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(
                f"Portfolio weights must sum to 1.0, got {weight_sum:.4f}. "
                f"Weights: {self.weights}"
            )

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset names."""
        return list(self.weights.keys())

    @property
    def weight_array(self) -> np.ndarray:
        """Weights as a NumPy array (same order as self.assets)."""
        return np.array([self.weights[a] for a in self.assets])

    @property
    def n_assets(self) -> int:
        """Number of assets in the portfolio."""
        return len(self.weights)


@dataclass
class PortfolioRiskMetrics:
    """
    Comprehensive risk metrics for a portfolio.

    All volatility/return metrics are annualized unless noted otherwise.
    """

    # Portfolio identification
    name: str
    n_assets: int
    weights: dict[str, float]

    # Return metrics
    portfolio_return_annualized: float
    portfolio_volatility_annualized: float
    sharpe_ratio: float  # Assumes 0% risk-free for now

    # Diversification
    diversification_ratio: float
    weighted_avg_volatility: float  # Σ w_i σ_i (what vol would be without diversification)

    # Drawdown
    max_drawdown: float
    max_drawdown_start: pd.Timestamp | None = None
    max_drawdown_end: pd.Timestamp | None = None

    # Correlation extremes
    min_correlation: float = 0.0
    max_correlation: float = 0.0  # Excluding diagonal

    # Per-asset statistics
    asset_volatilities: dict[str, float] = field(default_factory=dict)
    asset_returns: dict[str, float] = field(default_factory=dict)

    # Data quality
    n_observations: int = 0


# ---------------------------------------------------------------------------
# Portfolio return computation
# ---------------------------------------------------------------------------

def portfolio_returns(
    asset_returns: pd.DataFrame,
    portfolio: PortfolioDefinition,
) -> pd.Series:
    """
    Compute the portfolio return time series.

    R_p,t = Σ w_i R_i,t  (weighted sum of simple returns)

    IMPORTANT: This uses SIMPLE returns for portfolio aggregation.
    Log returns are NOT additive across assets. This is a common
    mistake — using log returns here would give incorrect portfolio returns.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Simple returns for each asset (columns must include all portfolio assets).
    portfolio : PortfolioDefinition
        Portfolio weights.

    Returns
    -------
    pd.Series
        Portfolio return time series.

    Raises
    ------
    ValueError
        If required assets are missing from the returns data.
    """
    # Verify all portfolio assets are in the returns data
    missing = set(portfolio.assets) - set(asset_returns.columns)
    if missing:
        raise ValueError(
            f"Missing return data for portfolio assets: {missing}. "
            f"Available: {list(asset_returns.columns)}"
        )

    # Select only the portfolio assets in the correct order
    aligned_returns = asset_returns[portfolio.assets]

    # Weighted sum: R_p = w^T R
    port_ret = aligned_returns.values @ portfolio.weight_array
    return pd.Series(port_ret, index=asset_returns.index, name=portfolio.name)


# ---------------------------------------------------------------------------
# Covariance and correlation estimation
# ---------------------------------------------------------------------------

def covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Estimate the covariance matrix of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (log or simple).
    method : str
        "sample" — standard sample covariance (unbiased, ddof=1).
        "ledoit_wolf" — Ledoit-Wolf shrinkage estimator.
            Shrinks toward a structured target to reduce noise.
            Better for high-dimensional portfolios (>10 assets).
    annualize : bool
        If True, multiply by 252 to annualize.

    Returns
    -------
    pd.DataFrame
        Covariance matrix with asset names as index and columns.
    """
    if method == "sample":
        cov = returns.cov(ddof=1)
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(returns.values)
        cov = pd.DataFrame(
            lw.covariance_, index=returns.columns, columns=returns.columns
        )
        logger.info(f"Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sample' or 'ledoit_wolf'.")

    if annualize:
        cov = cov * 252

    return cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix of asset returns.

    ρ_ij = Cov(R_i, R_j) / (σ_i × σ_j)

    Returns
    -------
    pd.DataFrame
        Correlation matrix. Diagonal = 1.0.
    """
    return returns.corr()


# ---------------------------------------------------------------------------
# Portfolio volatility
# ---------------------------------------------------------------------------

def portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Compute portfolio volatility from weights and covariance matrix.

    σ_p = √(w^T Σ w)

    This is the core formula of Modern Portfolio Theory. It captures
    the diversification benefit: σ_p is always ≤ Σ w_i σ_i.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weight vector (length N).
    cov_matrix : np.ndarray
        N×N covariance matrix (should be annualized if you want
        annualized volatility).

    Returns
    -------
    float
        Portfolio volatility (annualized if cov_matrix is annualized).
    """
    variance = weights @ cov_matrix @ weights
    if variance < 0:
        # This can happen with poorly estimated covariance matrices
        # (not positive semi-definite). It's a data quality issue.
        raise ValueError(
            f"Negative portfolio variance ({variance:.6f}). "
            "The covariance matrix may not be positive semi-definite."
        )
    return float(np.sqrt(variance))


def diversification_ratio(
    weights: np.ndarray,
    asset_volatilities: np.ndarray,
    port_volatility: float,
) -> float:
    """
    Compute the diversification ratio.

    DR = (Σ w_i σ_i) / σ_p

    Interpretation:
    - DR = 1.0: No diversification benefit (all ρ = 1)
    - DR = 1.5: Portfolio vol is 33% lower than naive weighted average
    - DR = 2.0: Portfolio vol is 50% lower — strong diversification

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    asset_volatilities : np.ndarray
        Individual asset volatilities (same order as weights).
    port_volatility : float
        Portfolio volatility.

    Returns
    -------
    float
        Diversification ratio (≥ 1.0).
    """
    weighted_avg_vol = np.sum(weights * asset_volatilities)
    if port_volatility <= 0:
        raise ValueError("Portfolio volatility must be positive")
    return float(weighted_avg_vol / port_volatility)


# ---------------------------------------------------------------------------
# Drawdown analysis
# ---------------------------------------------------------------------------

def maximum_drawdown(
    returns: pd.Series,
) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Compute maximum drawdown from a return series.

    Maximum drawdown = largest peak-to-trough decline in cumulative returns.

    This is arguably the most important risk metric for practitioners.
    VaR tells you what could happen in a single day. Maximum drawdown
    tells you the worst cumulative loss you would have experienced.

    Parameters
    ----------
    returns : pd.Series
        Simple or log returns (simple preferred for accuracy).

    Returns
    -------
    tuple[float, Timestamp, Timestamp]
        (max_drawdown, peak_date, trough_date)
        max_drawdown is negative (e.g., -0.35 means a 35% decline).
    """
    # Compute cumulative wealth index (starting at 1.0)
    wealth = (1 + returns).cumprod()

    # Running maximum
    running_max = wealth.cummax()

    # Drawdown at each point
    drawdown = (wealth - running_max) / running_max

    # Find the worst drawdown
    max_dd = drawdown.min()

    # Find the trough date
    trough_idx = drawdown.idxmin()

    # Find the peak date (the running max just before the trough)
    peak_idx = wealth.loc[:trough_idx].idxmax()

    return float(max_dd), peak_idx, trough_idx


# ---------------------------------------------------------------------------
# Full portfolio analysis
# ---------------------------------------------------------------------------

def analyze_portfolio(
    prices: pd.DataFrame,
    portfolio: PortfolioDefinition,
    cov_method: str = "sample",
) -> PortfolioRiskMetrics:
    """
    Run complete portfolio risk analysis.

    This is the main entry point. Given prices and weights, it computes
    all portfolio-level risk metrics.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical prices with DatetimeIndex. Must include all portfolio
        assets as columns.
    portfolio : PortfolioDefinition
        Portfolio weights.
    cov_method : str
        Covariance estimation method ("sample" or "ledoit_wolf").

    Returns
    -------
    PortfolioRiskMetrics
        Complete risk metrics.
    """
    # Compute simple returns (needed for portfolio return aggregation)
    rets = simple_returns(prices[portfolio.assets])

    # Portfolio return series
    port_rets = portfolio_returns(rets, portfolio)

    # Covariance matrix (annualized)
    cov = covariance_matrix(rets, method=cov_method, annualize=True)

    # Correlation matrix
    corr = correlation_matrix(rets)

    # Extract off-diagonal correlations
    mask = np.ones(corr.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diag = corr.values[mask]

    # Portfolio volatility
    port_vol = portfolio_volatility(portfolio.weight_array, cov.values)

    # Individual asset annualized volatilities
    asset_vols = rets.std() * np.sqrt(252)
    asset_annual_rets = rets.mean() * 252

    # Diversification ratio
    div_ratio = diversification_ratio(
        portfolio.weight_array, asset_vols[portfolio.assets].values, port_vol
    )

    # Portfolio annualized return
    port_annual_ret = float(port_rets.mean() * 252)

    # Sharpe ratio (0% risk-free for now)
    sharpe = port_annual_ret / port_vol if port_vol > 0 else 0.0

    # Maximum drawdown
    max_dd, peak_date, trough_date = maximum_drawdown(port_rets)

    return PortfolioRiskMetrics(
        name=portfolio.name,
        n_assets=portfolio.n_assets,
        weights=portfolio.weights,
        portfolio_return_annualized=port_annual_ret,
        portfolio_volatility_annualized=port_vol,
        sharpe_ratio=sharpe,
        diversification_ratio=div_ratio,
        weighted_avg_volatility=float(np.sum(portfolio.weight_array * asset_vols[portfolio.assets].values)),
        max_drawdown=max_dd,
        max_drawdown_start=peak_date,
        max_drawdown_end=trough_date,
        min_correlation=float(off_diag.min()) if len(off_diag) > 0 else 0.0,
        max_correlation=float(off_diag.max()) if len(off_diag) > 0 else 0.0,
        asset_volatilities={a: float(asset_vols[a]) for a in portfolio.assets},
        asset_returns={a: float(asset_annual_rets[a]) for a in portfolio.assets},
        n_observations=len(rets),
    )
