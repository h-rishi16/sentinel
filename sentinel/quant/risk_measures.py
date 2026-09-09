"""
Risk measures for Sentinel.

Implements Value at Risk (VaR) and Expected Shortfall (ES / CVaR)
as general-purpose functions that operate on any loss distribution.

This module is deliberately separate from the simulation engine
(monte_carlo.py) because VaR/ES can be computed from:
- Monte Carlo simulated P&L (this phase)
- Historical returns directly (historical VaR)
- Parametric assumptions (parametric VaR)

Separation enables code reuse across all three approaches.

Mathematical Background
-----------------------
Value at Risk (VaR) at confidence level α:
    VaR_α = -Quantile_{1-α}(P&L)

    Example: 95% VaR with P&L distribution:
    Sort all P&L values. The 5th percentile is the VaR.
    Interpretation: "We are 95% confident that losses will not
    exceed this amount over the specified time horizon."

Expected Shortfall (ES) at confidence level α:
    ES_α = -E[P&L | P&L ≤ -VaR_α]
    = average of all P&L values in the worst (1-α) fraction

    Example: 95% ES = average of the worst 5% of outcomes.
    ES answers: "If things go bad (beyond VaR), how bad on average?"

Properties:
    - ES ≥ VaR always (by definition, the average of the tail ≥ its boundary)
    - ES is a "coherent" risk measure (subadditive, monotone, etc.)
    - VaR is NOT coherent (violates subadditivity)
    - Basel III uses ES at 97.5% instead of VaR at 99% for this reason
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskMeasures:
    """Computed risk measures at a specific confidence level.

    Parameters
    ----------
    confidence_level : float
        Confidence level (e.g., 0.95, 0.99).
    var : float
        Value at Risk (positive number representing a loss).
    es : float
        Expected Shortfall / CVaR (positive number, always ≥ VaR).
    var_dollar : float
        VaR in dollar terms (VaR × portfolio value).
    es_dollar : float
        ES in dollar terms (ES × portfolio value).
    n_tail_observations : int
        Number of observations in the tail (used to compute ES).
    """

    confidence_level: float
    var: float
    es: float
    var_dollar: float
    es_dollar: float
    n_tail_observations: int


@dataclass
class VaRReport:
    """Complete VaR/ES report across multiple confidence levels.

    Parameters
    ----------
    risk_measures : list[RiskMeasures]
        Risk measures at each confidence level.
    portfolio_value : float
        Portfolio value used for dollar conversion.
    n_simulations : int
        Number of simulations (or observations) used.
    mean_pnl : float
        Mean P&L across all scenarios.
    std_pnl : float
        Standard deviation of P&L.
    skewness : float
        Skewness of P&L distribution.
    kurtosis : float
        Excess kurtosis of P&L distribution.
    worst_loss : float
        Worst single-scenario loss (positive number).
    best_gain : float
        Best single-scenario gain.
    """

    risk_measures: list[RiskMeasures]
    portfolio_value: float
    n_simulations: int
    mean_pnl: float
    std_pnl: float
    skewness: float
    kurtosis: float
    worst_loss: float
    best_gain: float


def compute_var(
    pnl: np.ndarray,
    confidence_level: float,
) -> float:
    """
    Compute Value at Risk from a P&L distribution.

    VaR_α = -Quantile_{1-α}(P&L)

    Returns a POSITIVE number representing a loss.

    Parameters
    ----------
    pnl : np.ndarray
        Profit and loss values. Negative = loss.
    confidence_level : float
        Confidence level, e.g. 0.95 or 0.99.

    Returns
    -------
    float
        VaR (positive number = loss amount).
    """
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    # The (1-α) quantile of P&L gives us the loss threshold
    quantile = np.percentile(pnl, (1 - confidence_level) * 100)
    return float(-quantile)  # Negate to make losses positive


def compute_es(
    pnl: np.ndarray,
    confidence_level: float,
) -> float:
    """
    Compute Expected Shortfall (Conditional VaR) from a P&L distribution.

    ES_α = -E[P&L | P&L ≤ -VaR_α]
    = mean of all P&L values that are worse than (or equal to) -VaR

    Returns a POSITIVE number representing an expected loss.

    Parameters
    ----------
    pnl : np.ndarray
        Profit and loss values. Negative = loss.
    confidence_level : float
        Confidence level, e.g. 0.95 or 0.99.

    Returns
    -------
    float
        Expected Shortfall (positive number, always ≥ VaR).
    """
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    var = compute_var(pnl, confidence_level)
    # Select all P&L values in the tail (worse than -VaR)
    tail = pnl[pnl <= -var]

    if len(tail) == 0:
        # Edge case: if VaR happens to be at the exact boundary,
        # include the nearest observations
        sorted_pnl = np.sort(pnl)
        n_tail = max(1, int(len(pnl) * (1 - confidence_level)))
        tail = sorted_pnl[:n_tail]

    return float(-tail.mean())


def compute_risk_measures(
    pnl: np.ndarray,
    confidence_levels: list[float],
    portfolio_value: float = 1_000_000.0,
) -> VaRReport:
    """
    Compute a complete VaR/ES report across multiple confidence levels.

    Parameters
    ----------
    pnl : np.ndarray
        Simulated or historical P&L distribution.
    confidence_levels : list[float]
        Confidence levels to compute (e.g., [0.95, 0.99, 0.999]).
    portfolio_value : float
        Portfolio notional value for dollar conversion.

    Returns
    -------
    VaRReport
        Complete risk report.
    """
    from scipy import stats as sp_stats

    measures = []
    for cl in confidence_levels:
        var = compute_var(pnl, cl)
        es = compute_es(pnl, cl)

        # Count tail observations
        n_tail = int(np.sum(pnl <= -var))

        measures.append(
            RiskMeasures(
                confidence_level=cl,
                var=var,
                es=es,
                var_dollar=var * portfolio_value,
                es_dollar=es * portfolio_value,
                n_tail_observations=n_tail,
            )
        )

    return VaRReport(
        risk_measures=measures,
        portfolio_value=portfolio_value,
        n_simulations=len(pnl),
        mean_pnl=float(pnl.mean()),
        std_pnl=float(pnl.std()),
        skewness=float(sp_stats.skew(pnl)),
        kurtosis=float(sp_stats.kurtosis(pnl)),
        worst_loss=float(-pnl.min()),
        best_gain=float(pnl.max()),
    )
