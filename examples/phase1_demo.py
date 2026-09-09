"""
Phase 1 Integration Test / Demo

Runs data ingestion, portfolio analytics, volatility estimation, 
Monte Carlo VaR, and Stress Testing end-to-end on real market data.
"""

import warnings
warnings.filterwarnings('ignore')  # Suppress arch/pandas warnings for clean output

import numpy as np
import pandas as pd
from sentinel.data.market import load_prices, forward_fill_prices
from sentinel.quant.returns import log_returns
from sentinel.quant.portfolio import PortfolioDefinition, analyze_portfolio
from sentinel.quant.volatility import garch_volatility, ewma_volatility, rolling_volatility
from sentinel.quant.monte_carlo import simulate_from_historical, SimulationConfig
from sentinel.quant.risk_measures import compute_risk_measures
from sentinel.quant.stress import Scenario, Shock, apply_hypothetical_scenario

def main():
    print("="*60)
    print("SENTINEL: PHASE 1 END-TO-END INTEGRATION TEST")
    print("="*60)

    # 1. Load Data
    prices = load_prices()
    prices = forward_fill_prices(prices)
    returns = log_returns(prices)
    print(f"[DATA] Loaded {len(prices)} days of prices for {list(prices.columns)}")

    # 2. Define Portfolio
    portfolio = PortfolioDefinition(
        weights={"SPY": 0.40, "AAPL": 0.30, "MSFT": 0.30}, 
        name="Core_Tech_Blend"
    )
    port_value = 1_000_000.0
    print(f"\n[PORTFOLIO] {portfolio.name} - ${port_value:,.0f}")
    for asset, weight in portfolio.weights.items():
        print(f"  - {asset}: {weight:.0%}")

    # 3. Portfolio Analytics
    analytics = analyze_portfolio(prices, portfolio, cov_method="ledoit_wolf")
    print(f"\n[ANALYTICS] Historical Metrics")
    print(f"  - Annualized Return: {analytics.portfolio_return_annualized:.2%}")
    print(f"  - Annualized Volatility: {analytics.portfolio_volatility_annualized:.2%}")
    print(f"  - Diversification Ratio: {analytics.diversification_ratio:.3f} (Values > 1 mean diversification works)")
    print(f"  - Max Historical Drawdown: {analytics.max_drawdown:.2%}")

    # 4. Volatility Deep-Dive (AAPL)
    print(f"\n[VOLATILITY] Current Volatility Estimates for AAPL (Annualized)")
    # We look at the very last day in the dataset
    roll_vol = rolling_volatility(returns[["AAPL"]], window=30).iloc[-1, 0]
    ewma_vol = ewma_volatility(returns[["AAPL"]], lambda_=0.94).iloc[-1, 0]
    garch_res = garch_volatility(returns["AAPL"])
    garch_current = garch_res.conditional_volatility.iloc[-1] * np.sqrt(252)
    
    print(f"  - 30-Day Rolling: {roll_vol:.2%}")
    print(f"  - EWMA (λ=0.94):  {ewma_vol:.2%}")
    print(f"  - GARCH(1,1):     {garch_current:.2%} (Persistence: {garch_res.persistence:.3f})")

    # 5. Monte Carlo VaR & ES
    print(f"\n[SIMULATION] Running 100,000 Monte Carlo paths (10-day horizon)...")
    config = SimulationConfig(n_simulations=100_000, time_horizon=10, random_seed=42)
    
    # We pass the weights in the exact order of the returns columns
    weights_array = np.array([portfolio.weights[col] for col in returns.columns])
    sim = simulate_from_historical(returns, weights_array, config)
    
    risk = compute_risk_measures(sim.portfolio_pnl, [0.95, 0.99], portfolio_value=port_value)
    print(f"  - Time elapsed: {sim.elapsed_seconds:.3f} seconds")
    
    print(f"\n[RISK MEASURES] 10-Day Horizon on ${port_value:,.0f}")
    for rm in risk.risk_measures:
        print(f"  - {rm.confidence_level:.0%} VaR: ${rm.var_dollar:,.0f} ({rm.var:.2%} loss)")
        print(f"  - {rm.confidence_level:.0%} Expected Shortfall: ${rm.es_dollar:,.0f} ({rm.es:.2%} loss)")

    # 6. Stress Testing
    print(f"\n[STRESS TEST] Hypothetical 'Severe Tech Crash'")
    crash_scenario = Scenario("Severe Tech Crash", [
        Shock("AAPL", -0.30),
        Shock("MSFT", -0.25),
        Shock("SPY", -0.10)  # SPY drops less because it's diversified
    ])
    stress_res = apply_hypothetical_scenario(portfolio, crash_scenario, port_value)
    
    print(f"  - Scenario Impact: {stress_res.portfolio_pnl_pct:.2%} loss")
    print(f"  - Dollar Loss:     ${stress_res.portfolio_pnl_dollar:,.0f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
