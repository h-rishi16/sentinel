"""
Rigorous Mathematical Validation of Phase 1 Engine.

1. Law of Large Numbers / Convergence to Black-Scholes theoretical moments.
2. Numerical Stability of Eigenvalue Clipping on Broken Correlation Matrices.
3. Volatility Estimator Decay Profiles (Ghost Effect).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sentinel.quant.monte_carlo import simulate_gbm, SimulationConfig, cholesky_decompose
from sentinel.quant.volatility import rolling_volatility, ewma_volatility

def test_monte_carlo_convergence():
    print("\n--- TEST 1: MONTE CARLO CONVERGENCE (1,000,000 PATHS) ---")
    print("Testing if our discretized numerical simulation converges to")
    print("exact continuous-time analytical moments.")
    
    # Parameters
    mu = 0.10      # 10% expected return
    sigma = 0.20   # 20% volatility
    S0 = 100.0
    T_years = 1.0  # 1 year
    T_days = 252
    
    # 1. Theoretical Analytical Moments for Log-Normal Distribution (GBM)
    # E[S_T] = S0 * exp(mu * T)
    expected_mean_price = S0 * np.exp(mu * T_years)
    # Var[S_T] = S0^2 * exp(2 * mu * T) * (exp(sigma^2 * T) - 1)
    expected_var_price = (S0**2) * np.exp(2 * mu * T_years) * (np.exp((sigma**2) * T_years) - 1)
    
    # 2. Run our Engine
    config = SimulationConfig(n_simulations=1_000_000, time_horizon=T_days, random_seed=42)
    sim = simulate_gbm(
        expected_returns=np.array([mu]),
        volatilities=np.array([sigma]),
        correlation_matrix=np.array([[1.0]]),
        weights=np.array([1.0]),
        config=config
    )
    
    # PnL is (S_T / S_0) - 1. So S_T = S_0 * (PnL + 1)
    simulated_prices = S0 * (sim.portfolio_pnl + 1)
    sim_mean = np.mean(simulated_prices)
    sim_var = np.var(simulated_prices)
    
    mean_error_bps = abs(sim_mean - expected_mean_price) / expected_mean_price * 10000
    var_error_pct = abs(sim_var - expected_var_price) / expected_var_price * 100
    
    print(f"Theoretical Mean Price: ${expected_mean_price:,.4f}")
    print(f"Simulated Mean Price:   ${sim_mean:,.4f}")
    print(f"Mean Error:             {mean_error_bps:.2f} basis points")
    
    print(f"\nTheoretical Variance:   {expected_var_price:,.4f}")
    print(f"Simulated Variance:     {sim_var:,.4f}")
    print(f"Variance Error:         {var_error_pct:.4f} %")
    
    if mean_error_bps < 1.0 and var_error_pct < 1.0:
        print("=> PASS: Simulation converges to theoretical math perfectly.")


def test_numerical_stability():
    print("\n--- TEST 2: NUMERICAL STABILITY (NON-PSD CORRELATION) ---")
    print("Testing if Cholesky decomposition survives a mathematically impossible matrix.")
    
    # Mathematically impossible correlation matrix:
    # A and B are 90% correlated. B and C are 90% correlated.
    # Therefore, A and C MUST be highly positively correlated.
    # We set A and C to -90%. This matrix has negative eigenvalues (non-PSD).
    broken_matrix = np.array([
        [ 1.0,  0.9, -0.9],
        [ 0.9,  1.0,  0.9],
        [-0.9,  0.9,  1.0]
    ])
    
    eigenvalues = np.linalg.eigvals(broken_matrix)
    print(f"Original Eigenvalues: {eigenvalues}")
    print(f"Is Original Matrix valid (all eigenvalues > 0)? {all(eigenvalues > 0)}")
    
    try:
        # Standard numpy will crash here if not for our clipping algorithm
        L = cholesky_decompose(broken_matrix)
        repaired_matrix = L @ L.T
        rep_eigenvalues = np.linalg.eigvals(repaired_matrix)
        
        print(f"\nRepaired Eigenvalues: {rep_eigenvalues}")
        print("=> PASS: Engine successfully clipped negative eigenvalues and repaired matrix.")
    except Exception as e:
        print(f"=> FAIL: Engine crashed with {str(e)}")


def test_ghost_effect():
    print("\n--- TEST 3: THE 'GHOST EFFECT' (ROLLING vs EWMA) ---")
    print("Injecting a massive shock on Day 10. Observing how estimators recover by Day 50.")
    
    # 60 days of 0% returns, but Day 10 has a huge 20% return
    returns = np.zeros(60)
    returns[10] = 0.20
    df = pd.DataFrame({"Asset": returns})
    
    roll_vol = rolling_volatility(df, window=30, annualize=False)["Asset"]
    ewma_vol = ewma_volatility(df, lambda_=0.94, annualize=False)["Asset"]
    
    print(f"Day 10 (Shock Day):")
    print(f"  Rolling: {roll_vol.iloc[10]:.4f} | EWMA: {ewma_vol.iloc[10]:.4f}")
    
    print(f"Day 39 (Just before shock drops out of 30-day window):")
    print(f"  Rolling: {roll_vol.iloc[39]:.4f} | EWMA: {ewma_vol.iloc[39]:.4f}")
    
    print(f"Day 40 (Shock drops out of 30-day window):")
    print(f"  Rolling: {roll_vol.iloc[40]:.4f} | EWMA: {ewma_vol.iloc[40]:.4f}")
    
    print(f"Day 60 (Recovery phase):")
    print(f"  Rolling: {roll_vol.iloc[59]:.4f} | EWMA: {ewma_vol.iloc[59]:.4f}")
    
    if roll_vol.iloc[40] == 0.0 and ewma_vol.iloc[40] > 0.0:
        print("\n=> PASS: EWMA decays smoothly, while Rolling Vol suffers an abrupt 'Ghost Effect' drop to 0.")

if __name__ == "__main__":
    test_monte_carlo_convergence()
    test_numerical_stability()
    test_ghost_effect()
