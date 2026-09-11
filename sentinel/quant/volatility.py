"""
Volatility estimation for Sentinel.

This module implements three approaches to volatility estimation, from
simplest to most sophisticated. All three are used in practice, and no
single method dominates in all situations.

Methods
-------
1. Rolling (Historical) Volatility
   - Simplest: standard deviation over a fixed lookback window.
   - Pro: Transparent, easy to understand and audit.
   - Con: "Ghost effect" — when an extreme return exits the window,
     volatility drops abruptly. All observations weighted equally.

2. EWMA (Exponentially Weighted Moving Average)
   - JP Morgan RiskMetrics (1996) standard with λ=0.94.
   - Pro: Recent data weighted more heavily, no ghost effect.
   - Con: λ is fixed (not estimated from data), no mean-reversion.
   - Note: EWMA is a special case of IGARCH (Integrated GARCH) where
     the unconditional variance is undefined.

3. GARCH(1,1) — Generalized Autoregressive Conditional Heteroskedasticity
   - Bollerslev (1986). The workhorse of academic and institutional
     volatility modeling.
   - Pro: Parameters estimated via MLE, captures volatility clustering,
     mean-reverts to long-run variance.
   - Con: More complex, requires sufficient data for stable estimation,
     assumes symmetric response to positive/negative shocks.
   - Extensions (future): GJR-GARCH, EGARCH (asymmetric response).

Mathematical Background
-----------------------
Volatility clustering: Large absolute returns tend to be followed by
large absolute returns, and small by small. This means volatility is
NOT constant — it varies over time (heteroskedasticity). All three
methods attempt to capture this time-varying behavior, with increasing
sophistication.

Annualization: Daily volatility is annualized by multiplying by √252
(there are ~252 trading days per year). This assumes returns are
independent and identically distributed, which is only approximately
true. GARCH explicitly models the violation of this assumption.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class GARCHResult:
    """Results from fitting a GARCH(1,1) model.

    Parameters
    ----------
    omega : float
        Constant term (ω). Must be positive.
    alpha : float
        ARCH coefficient (α). Weight on yesterday's squared return.
    beta : float
        GARCH coefficient (β). Weight on yesterday's variance.
    long_run_variance : float
        Unconditional variance = ω / (1 - α - β).
    long_run_volatility : float
        √(long_run_variance), annualized.
    conditional_volatility : pd.Series
        Time series of estimated daily conditional volatility.
    persistence : float
        α + β. Measures how slowly volatility reverts to the long-run level.
        Values close to 1.0 indicate very slow mean-reversion.
    log_likelihood : float
        Log-likelihood of the fitted model.
    aic : float
        Akaike Information Criterion. Lower is better (for model comparison).
    bic : float
        Bayesian Information Criterion. Lower is better.
    """

    omega: float
    alpha: float
    beta: float
    long_run_variance: float
    long_run_volatility: float
    conditional_volatility: pd.Series
    persistence: float
    log_likelihood: float
    aic: float
    bic: float


# ---------------------------------------------------------------------------
# Rolling (Historical) Volatility
# ---------------------------------------------------------------------------

def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 30,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling (historical) volatility.

    σ_t = std(r_{t-w+1}, ..., r_t)

    Parameters
    ----------
    returns : pd.DataFrame
        Log or simple returns with DatetimeIndex.
    window : int
        Lookback window in trading days. Common values:
        - 21 (1 month)
        - 63 (1 quarter)
        - 252 (1 year)
    annualize : bool
        If True, multiply by √252 to annualize.

    Returns
    -------
    pd.DataFrame
        Rolling volatility series. First (window-1) values are NaN.
    """
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    if window > len(returns):
        raise ValueError(
            f"Window ({window}) exceeds number of observations ({len(returns)})"
        )

    # ddof=1 for sample standard deviation (Bessel's correction)
    vol = returns.rolling(window=window).std(ddof=1)

    if annualize:
        vol = vol * np.sqrt(252)

    logger.info(
        f"Computed {window}-day rolling volatility for {len(returns.columns)} assets"
    )
    return vol


# ---------------------------------------------------------------------------
# EWMA Volatility
# ---------------------------------------------------------------------------

def ewma_volatility(
    returns: pd.DataFrame,
    lambda_: float = 0.94,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute EWMA (Exponentially Weighted Moving Average) volatility.

    σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

    This is the JP Morgan RiskMetrics (1996) approach. λ=0.94 is the
    industry standard for daily data, chosen by JP Morgan based on
    empirical analysis across asset classes.

    Why λ=0.94?
    - The effective window length of an EWMA is approximately 1/(1-λ).
    - For λ=0.94, this gives ~17 days — roughly 3-4 trading weeks.
    - JP Morgan found this balances responsiveness to new information
      against stability of the estimate.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns with DatetimeIndex.
    lambda_ : float
        Decay factor. Must be in (0, 1). Default 0.94 (RiskMetrics daily).
        Common alternatives: 0.97 (monthly), 0.99 (very slow decay).
    annualize : bool
        If True, multiply by √252 to annualize.

    Returns
    -------
    pd.DataFrame
        EWMA volatility (standard deviation) series.
    """
    if not 0 < lambda_ < 1:
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")

    result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for col in returns.columns:
        r = returns[col].values
        n = len(r)
        variance = np.zeros(n)

        # Initialize with the first squared return
        variance[0] = r[0] ** 2

        # Recursive EWMA update
        for t in range(1, n):
            variance[t] = lambda_ * variance[t - 1] + (1 - lambda_) * r[t - 1] ** 2

        result[col] = np.sqrt(variance)

    if annualize:
        result = result * np.sqrt(252)

    logger.info(
        f"Computed EWMA volatility (λ={lambda_}) for {len(returns.columns)} assets"
    )
    return result


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

def garch_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> GARCHResult:
    """
    Fit a GARCH(p,q) model and extract conditional volatility.

    σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}   (for GARCH(1,1))

    Uses the `arch` library for maximum likelihood estimation.

    Why GARCH(1,1)?
    - Bollerslev (1986) showed that GARCH(1,1) captures the key
      empirical features of financial return volatility: clustering,
      mean-reversion, and heavy tails.
    - Higher-order GARCH(p,q) models rarely improve fit significantly.
    - Hansen & Lunde (2005) compared 330 ARCH-type models on exchange
      rate data and found that none consistently outperformed GARCH(1,1).

    Parameters
    ----------
    returns : pd.Series
        Log returns for a single asset. Must have >100 observations
        for stable parameter estimation.
    p : int
        Order of the GARCH term (lagged variance). Default 1.
    q : int
        Order of the ARCH term (lagged squared returns). Default 1.
    dist : str
        Error distribution: "normal", "t" (Student-t), or "skewt".
        "normal" is simplest; "t" captures heavy tails better.

    Returns
    -------
    GARCHResult
        Structured result containing parameters, diagnostics, and
        the conditional volatility time series.

    Raises
    ------
    ValueError
        If insufficient data or convergence failure.
    """
    from arch import arch_model

    if len(returns) < 100:
        raise ValueError(
            f"GARCH requires >= 100 observations for stable estimation, "
            f"got {len(returns)}"
        )

    # The arch library expects returns in percentage form (×100)
    # for numerical stability during optimization
    returns_pct = returns * 100

    # Specify the model
    model = arch_model(
        returns_pct,
        vol="Garch",
        p=p,
        q=q,
        dist=dist,
        mean="Constant",  # Constant mean model (simplest)
    )

    # Fit via maximum likelihood estimation
    result = model.fit(disp="off")  # Suppress optimizer output

    # Extract parameters (convert back from percentage scale)
    # arch library reports ω in percentage-squared terms,
    # so divide by 10000 to get decimal-scale variance
    omega = result.params.get("omega", 0.0) / 10000
    alpha = result.params.get("alpha[1]", 0.0)
    beta = result.params.get("beta[1]", 0.0)

    persistence = alpha + beta

    # Long-run (unconditional) variance
    if persistence < 1.0:
        long_run_var = omega / (1 - persistence)
    else:
        # IGARCH case: unconditional variance is undefined
        long_run_var = float("nan")
        logger.warning(
            f"GARCH persistence α+β = {persistence:.4f} >= 1.0 (IGARCH). "
            "Long-run variance is undefined."
        )

    long_run_vol_annual = np.sqrt(long_run_var * 252) if not np.isnan(long_run_var) else float("nan")

    # Conditional volatility series (convert back from percentage)
    cond_vol = result.conditional_volatility / 100

    garch_result = GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        long_run_variance=long_run_var,
        long_run_volatility=long_run_vol_annual,
        conditional_volatility=cond_vol,
        persistence=persistence,
        log_likelihood=result.loglikelihood,
        aic=result.aic,
        bic=result.bic,
    )

    logger.info(
        f"GARCH({p},{q}) fitted: ω={omega:.6f}, α={alpha:.4f}, β={beta:.4f}, "
        f"persistence={persistence:.4f}, long-run vol={long_run_vol_annual:.4f}"
    )

    return garch_result


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_volatility_methods(
    returns: pd.Series,
    rolling_window: int = 30,
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """
    Compute and compare all three volatility methods side-by-side.

    Useful for understanding how each method responds to market events.

    Parameters
    ----------
    returns : pd.Series
        Log returns for a single asset.
    rolling_window : int
        Window for rolling volatility.
    ewma_lambda : float
        Decay factor for EWMA.

    Returns
    -------
    pd.DataFrame
        Columns: rolling, ewma, garch (all annualized).
    """
    # Convert to DataFrame for rolling/ewma functions
    returns_df = returns.to_frame()

    rolling = rolling_volatility(returns_df, window=rolling_window)[returns.name]
    ewma = ewma_volatility(returns_df, lambda_=ewma_lambda)[returns.name]

    garch_result = garch_volatility(returns)
    garch_vol = garch_result.conditional_volatility * np.sqrt(252)

    comparison = pd.DataFrame(
        {
            f"rolling_{rolling_window}d": rolling,
            "ewma": ewma,
            "garch": garch_vol,
        },
        index=returns.index,
    )

    return comparison
