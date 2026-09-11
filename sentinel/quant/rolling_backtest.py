"""
Rolling Backtest Pipeline for Sentinel.

Executes out-of-sample validation of VaR models by sliding a historical
training window over time, predicting VaR for T+1, and comparing against
actual T+1 P&L.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from sentinel.quant.backtest import (
    ESBacktestResult,
    KupiecTestResult,
    run_kupiec_pof_test,
)
from sentinel.quant.portfolio import (
    PortfolioDefinition,
    covariance_matrix,
    portfolio_volatility,
)
from sentinel.quant.returns import simple_returns

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Contains the daily rolling results and the statistical Kupiec and ES tests."""
    method_name: str
    daily_results: pd.DataFrame  # Columns: ['Date', 'Actual_PnL', 'Predicted_VaR', 'Predicted_ES', 'Is_Breach']
    kupiec_result: KupiecTestResult
    es_result: ESBacktestResult | None = None


def parametric_var(portfolio_vol: float, confidence_level: float, portfolio_value: float = 1.0) -> float:
    """
    Delta-Normal Parametric VaR.
    Assumes returns follow a Normal distribution.
    VaR = Z-score * portfolio_volatility
    """
    z_score = norm.ppf(confidence_level)
    return float(z_score * portfolio_vol * portfolio_value)


def historical_var(historical_pnls: np.ndarray, confidence_level: float) -> float:
    """
    Historical Simulation VaR.
    Sorts historical returns and picks the exact percentile.
    """
    percentile = (1.0 - confidence_level) * 100
    return float(-np.percentile(historical_pnls, percentile))


def run_rolling_backtest(
    prices: pd.DataFrame,
    portfolio: PortfolioDefinition,
    window_size: int = 252,
    confidence_level: float = 0.99,
    method: str = "parametric"
) -> BacktestResults:
    """
    Run an out-of-sample sliding window backtest.

    For each day T (starting from index `window_size`):
    1. Extract prices from [T - window_size : T - 1] (Strictly out of sample)
    2. Estimate parameters and predict VaR for day T.
    3. Record actual PnL on day T.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data.
    portfolio : PortfolioDefinition
        Portfolio weights.
    window_size : int
        Number of days to use for training (default 252 = 1 year).
    confidence_level : float
        VaR confidence level.
    method : str
        "parametric", "historical", or "monte_carlo".

    Returns
    -------
    BacktestResults
    """
    if len(prices) <= window_size:
        raise ValueError(f"Price history ({len(prices)}) must be larger than window_size ({window_size})")

    from sentinel.quant.monte_carlo import SimulationConfig, simulate_from_historical
    from sentinel.quant.risk_measures import compute_es, compute_var

    # Compute ALL daily returns upfront for the entire dataset for efficiency
    all_returns = simple_returns(prices[portfolio.assets])

    # Pre-calculate the actual portfolio daily PnL (percentage)
    actual_port_pnls = (all_returns.values @ portfolio.weight_array)
    actual_port_pnls_series = pd.Series(actual_port_pnls, index=all_returns.index)

    dates = []
    actual_pnl_list = []
    predicted_var_list = []
    predicted_es_list = []

    # Slide the window
    for i in range(window_size, len(all_returns)):
        target_date = all_returns.index[i]

        # Training window: strictly strictly BEFORE the target date
        train_returns = all_returns.iloc[i - window_size : i]

        if method == "parametric":
            cov = covariance_matrix(train_returns, method="sample", annualize=False)
            daily_vol = portfolio_volatility(portfolio.weight_array, cov.values)
            var_pred = parametric_var(daily_vol, confidence_level)
            # Parametric ES for Normal distribution: (phi(Z) / (1 - alpha)) * sigma
            z_score = norm.ppf(confidence_level)
            phi = norm.pdf(z_score)
            es_pred = (phi / (1 - confidence_level)) * daily_vol

        elif method == "historical":
            hist_port_pnls = train_returns.values @ portfolio.weight_array
            var_pred = historical_var(hist_port_pnls, confidence_level)

            # Historical ES: Average of all losses > VaR
            tail = hist_port_pnls[hist_port_pnls <= -var_pred]
            es_pred = -tail.mean() if len(tail) > 0 else var_pred

        elif method == "monte_carlo":
            config = SimulationConfig(n_simulations=10_000, time_horizon=1, random_seed=42 + i)
            sim = simulate_from_historical(train_returns, portfolio.weight_array, config)
            var_pred = compute_var(sim.portfolio_pnl, confidence_level)
            es_pred = compute_es(sim.portfolio_pnl, confidence_level)

        else:
            raise ValueError(f"Unknown method {method}")

        actual_pnl = actual_port_pnls_series.iloc[i]

        dates.append(target_date)
        actual_pnl_list.append(actual_pnl)
        predicted_var_list.append(var_pred)
        predicted_es_list.append(es_pred)

    df_results = pd.DataFrame({
        "Date": dates,
        "Actual_PnL": actual_pnl_list,
        "Predicted_VaR": predicted_var_list,
        "Predicted_ES": predicted_es_list
    })

    df_results["Is_Breach"] = df_results["Actual_PnL"] < -df_results["Predicted_VaR"]

    kupiec = run_kupiec_pof_test(
        pnl=df_results["Actual_PnL"].values,
        var_predictions=df_results["Predicted_VaR"].values,
        confidence_level=confidence_level
    )

    from sentinel.quant.backtest import run_es_backtest
    es_res = run_es_backtest(
        pnl=df_results["Actual_PnL"].values,
        var_predictions=df_results["Predicted_VaR"].values,
        es_predictions=df_results["Predicted_ES"].values,
        significance_level=0.05
    )

    return BacktestResults(
        method_name=method,
        daily_results=df_results,
        kupiec_result=kupiec,
        es_result=es_res
    )
