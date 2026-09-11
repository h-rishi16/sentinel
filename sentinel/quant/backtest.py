"""
Backtesting and Validation for Sentinel.

Implements the Kupiec POF (Proportion of Failures) test to statistically
validate Value at Risk (VaR) models. Also includes a Risk Limits engine
to trigger alerts when portfolio metrics breach defined boundaries.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class KupiecTestResult:
    """Results of a Kupiec POF (Proportion of Failures) backtest."""
    confidence_level: float
    observations: int
    expected_breaches: float
    actual_breaches: int
    breach_rate: float
    lr_statistic: float
    p_value: float
    is_accepted: bool  # True if p_value >= 0.05 (model is valid)


def run_kupiec_pof_test(
    pnl: np.ndarray,
    var_predictions: np.ndarray,
    confidence_level: float,
    significance_level: float = 0.05,
) -> KupiecTestResult:
    """
    Run the Kupiec Proportion of Failures (POF) Likelihood Ratio test.

    Tests the null hypothesis that the observed breach rate equals the
    expected breach rate (1 - confidence_level).

    Parameters
    ----------
    pnl : np.ndarray
        Historical daily P&L array. Negative values = losses.
    var_predictions : np.ndarray
        Historical VaR predictions corresponding to the P&L days.
        Must be positive numbers (representing loss thresholds).
    confidence_level : float
        The VaR confidence level (e.g., 0.99 for 99% VaR).
    significance_level : float, default 0.05
        The alpha level for the chi-square test to accept/reject the model.

    Returns
    -------
    KupiecTestResult
        Complete statistical results of the backtest.
    """
    if len(pnl) != len(var_predictions):
        raise ValueError("PnL and VaR predictions must have the same length.")

    n = len(pnl)
    if n == 0:
        raise ValueError("Input arrays cannot be empty.")

    p_expected = 1.0 - confidence_level

    # A breach occurs when the actual loss is worse (more negative) than -VaR
    # (assuming VaR is expressed as a positive number)
    breaches = (pnl < -var_predictions).astype(int)
    x = int(np.sum(breaches))

    p_observed = x / n

    # Log-likelihood of the null hypothesis (true failure rate = p_expected)
    # Using np.clip to prevent log(0) domain errors
    ll_null = (n - x) * np.log(max(1 - p_expected, 1e-10)) + x * np.log(max(p_expected, 1e-10))

    # Log-likelihood of the alternative hypothesis (true failure rate = p_observed)
    if x == 0:
        ll_alt = n * np.log(1.0)
    elif x == n:
        ll_alt = n * np.log(1.0)
    else:
        ll_alt = (n - x) * np.log(1 - p_observed) + x * np.log(p_observed)

    # The Likelihood Ratio statistic (clamped to 0 to prevent -0.0 from float arithmetic)
    lr_stat = max(-2.0 * (ll_null - ll_alt), 0.0)

    # Kupiec LR stat follows a Chi-Square distribution with 1 degree of freedom
    p_value = 1.0 - stats.chi2.cdf(lr_stat, df=1)

    return KupiecTestResult(
        confidence_level=confidence_level,
        observations=n,
        expected_breaches=n * p_expected,
        actual_breaches=x,
        breach_rate=p_observed,
        lr_statistic=float(lr_stat),
        p_value=float(p_value),
        is_accepted=bool(p_value >= significance_level)
    )

@dataclass
class ESBacktestResult:
    """Results of an Expected Shortfall exceedance residual test."""
    breach_count: int
    mean_exceedance_residual: float
    t_statistic: float
    p_value: float
    is_accepted: bool


def run_es_backtest(
    pnl: np.ndarray,
    var_predictions: np.ndarray,
    es_predictions: np.ndarray,
    significance_level: float = 0.05,
) -> ESBacktestResult:
    """
    Run a McNeil and Frey style Expected Shortfall backtest.

    Isolates days where PnL breached VaR, calculates the discrepancy
    (Actual Loss - Predicted ES), and runs a one-sample t-test to check
    if the mean discrepancy is statistically different from 0.

    Parameters
    ----------
    pnl : np.ndarray
        Historical daily P&L.
    var_predictions : np.ndarray
        Historical VaR predictions.
    es_predictions : np.ndarray
        Historical ES predictions.
    significance_level : float, default 0.05
        Alpha level for the t-test.

    Returns
    -------
    ESBacktestResult
    """
    # Identify breach days (losses > VaR)
    breaches = pnl < -var_predictions

    if np.sum(breaches) < 2:
        # Not enough breaches to run a statistical t-test
        return ESBacktestResult(
            breach_count=int(np.sum(breaches)),
            mean_exceedance_residual=0.0,
            t_statistic=0.0,
            p_value=1.0,
            is_accepted=True
        )

    actual_losses = -pnl[breaches]
    predicted_es = es_predictions[breaches]

    # Residual = Actual Loss - Predicted Expected Shortfall
    # Positive residual means the model UNDER-predicted the severity of the crash.
    residuals = actual_losses - predicted_es

    mean_res = float(np.mean(residuals))

    # One-sample t-test against population mean = 0
    t_stat, p_val = stats.ttest_1samp(residuals, 0.0)

    return ESBacktestResult(
        breach_count=len(residuals),
        mean_exceedance_residual=mean_res,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        is_accepted=bool(p_val >= significance_level)
    )



# ---------------------------------------------------------------------------
# Risk Limits Engine
# ---------------------------------------------------------------------------

@dataclass
class LimitBreach:
    """Represents a violated risk limit."""
    metric_name: str
    limit_value: float
    actual_value: float
    breach_type: str  # 'Upper' or 'Lower'
    severity: str     # 'Warning' or 'Critical'


class RiskLimitsEngine:
    """
    Monitors portfolio risk metrics against institutional limits.
    """
    def __init__(self):
        self.upper_limits = {}
        self.lower_limits = {}
        self.critical_multiplier = 1.2  # Critical if 20% over limit

    def set_upper_limit(self, metric: str, value: float) -> None:
        self.upper_limits[metric] = value

    def set_lower_limit(self, metric: str, value: float) -> None:
        self.lower_limits[metric] = value

    def check_limits(self, metrics: dict[str, float]) -> list[LimitBreach]:
        """
        Evaluate current metrics against all limits.
        
        Parameters
        ----------
        metrics : dict[str, float]
            Dictionary of calculated risk metrics. E.g., {'VaR_99': 50000}
            
        Returns
        -------
        list[LimitBreach]
            List of all limit breaches found. Empty if compliant.
        """
        breaches = []

        for metric, actual in metrics.items():
            # Check upper limits (e.g., VaR shouldn't exceed X)
            if metric in self.upper_limits:
                limit = self.upper_limits[metric]
                if actual > limit:
                    severity = "Critical" if actual > (limit * self.critical_multiplier) else "Warning"
                    breaches.append(LimitBreach(metric, limit, actual, "Upper", severity))

            # Check lower limits (e.g., Diversification Ratio shouldn't fall below Y)
            if metric in self.lower_limits:
                limit = self.lower_limits[metric]
                if actual < limit:
                    # For lower limits, critical means 20% *below* the limit
                    severity = "Critical" if actual < (limit / self.critical_multiplier) else "Warning"
                    breaches.append(LimitBreach(metric, limit, actual, "Lower", severity))

        return breaches
