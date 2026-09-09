"""
Stress Testing and Scenario Analysis for Sentinel.

This module applies hypothetical and historical stress scenarios to portfolios
to measure extreme tail risk that standard VaR models might miss.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sentinel.quant.portfolio import PortfolioDefinition
from sentinel.quant.returns import simple_returns

logger = logging.getLogger(__name__)


@dataclass
class Shock:
    """
    Defines a shock to a specific asset.

    Parameters
    ----------
    asset : str
        The asset ticker (e.g., "AAPL").
    shock_value : float
        The size of the shock (e.g., -0.20 for a 20% drop).
    type : Literal["relative", "absolute"]
        "relative": A percentage change (e.g., drop 20% -> price * (1 - 0.20)).
        "absolute": An absolute change (used more for rates/yields, e.g., +2%).
    """
    asset: str
    shock_value: float
    type: Literal["relative", "absolute"] = "relative"


@dataclass
class Scenario:
    """
    A named collection of shocks representing a stress event.

    Parameters
    ----------
    name : str
        Name of the scenario (e.g., "Tech Sector Crash").
    shocks : list[Shock]
        List of shocks to apply.
    description : str, optional
        Context about the scenario.
    """
    name: str
    shocks: list[Shock]
    description: str = ""

    def get_shock_for_asset(self, asset: str) -> float:
        """Get the relative shock value for an asset, defaulting to 0.0 if not shocked."""
        for shock in self.shocks:
            if shock.asset == asset:
                if shock.type == "relative":
                    return shock.shock_value
                else:
                    raise NotImplementedError("Absolute shocks require current price context.")
        return 0.0


@dataclass
class StressTestResult:
    """Results of applying a scenario to a portfolio."""
    scenario_name: str
    portfolio_pnl_pct: float
    portfolio_pnl_dollar: float
    asset_pnls_pct: dict[str, float]


def apply_hypothetical_scenario(
    portfolio: PortfolioDefinition,
    scenario: Scenario,
    portfolio_value: float = 1_000_000.0,
) -> StressTestResult:
    """
    Apply a hypothetical scenario of instantaneous shocks to a portfolio.

    Math:
        P&L_port = Σ (w_i * Shock_i)

    Parameters
    ----------
    portfolio : PortfolioDefinition
        The portfolio to stress.
    scenario : Scenario
        The hypothetical scenario containing shocks.
    portfolio_value : float
        The total dollar value of the portfolio.

    Returns
    -------
    StressTestResult
        The P&L impact on the portfolio and individual assets.
    """
    asset_pnls = {}
    port_pnl_pct = 0.0

    for asset in portfolio.assets:
        weight = portfolio.weights[asset]
        shock_pct = scenario.get_shock_for_asset(asset)
        
        asset_pnl = shock_pct
        asset_pnls[asset] = asset_pnl
        
        port_pnl_pct += weight * asset_pnl

    return StressTestResult(
        scenario_name=scenario.name,
        portfolio_pnl_pct=port_pnl_pct,
        portfolio_pnl_dollar=port_pnl_pct * portfolio_value,
        asset_pnls_pct=asset_pnls,
    )


def historical_scenario_impact(
    prices: pd.DataFrame,
    portfolio: PortfolioDefinition,
    start_date: str,
    end_date: str,
    scenario_name: str,
    portfolio_value: float = 1_000_000.0,
) -> StressTestResult:
    """
    Replay a historical time window and calculate the peak-to-trough 
    portfolio drawdown during that specific window.

    This answers: "If this exact historical crisis happened to my current 
    portfolio, what is the maximum I would have lost during that period?"

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data containing the crisis period.
    portfolio : PortfolioDefinition
        The current portfolio weights.
    start_date : str
        Start of the crisis window (YYYY-MM-DD).
    end_date : str
        End of the crisis window (YYYY-MM-DD).
    scenario_name : str
        Name of the historical event (e.g., "COVID-19 Crash").
    portfolio_value : float
        The total dollar value of the portfolio.

    Returns
    -------
    StressTestResult
        The max drawdown impact of that historical period.
    """
    # Slice the historical period
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    crisis_prices = prices.loc[mask]

    if crisis_prices.empty:
        raise ValueError(f"No price data found between {start_date} and {end_date}")

    missing = set(portfolio.assets) - set(crisis_prices.columns)
    if missing:
        raise ValueError(f"Missing price data for assets: {missing}")

    # Calculate the normalized value of each asset (start at 1.0)
    normalized_prices = crisis_prices / crisis_prices.iloc[0]
    
    # Calculate portfolio wealth over time
    wealth = normalized_prices.values @ portfolio.weight_array
    wealth = pd.Series(wealth, index=crisis_prices.index)
    
    # Calculate max drawdown during this specific window
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    
    max_loss_pct = float(drawdown.min())
    
    # Calculate what each asset did over the full window (peak to trough)
    # Using the portfolio's max drawdown dates
    trough_idx = drawdown.idxmin()
    peak_idx = wealth.loc[:trough_idx].idxmax()
    
    asset_pnls = {}
    if len(crisis_prices) > 0:
        asset_rets = (crisis_prices.loc[trough_idx] / crisis_prices.loc[peak_idx]) - 1
        asset_pnls = {asset: float(asset_rets[asset]) for asset in portfolio.assets}

    return StressTestResult(
        scenario_name=scenario_name,
        portfolio_pnl_pct=max_loss_pct,
        portfolio_pnl_dollar=max_loss_pct * portfolio_value,
        asset_pnls_pct=asset_pnls,
    )
