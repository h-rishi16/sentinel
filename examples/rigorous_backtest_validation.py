"""
Rigorous Mathematical Validation of Phase 2 (Backtesting Engine).

1. Statistical Power of Expected Shortfall Test (Fat-Tail Detection)
2. Statistical Power of Kupiec POF Test (Type II Error Detection)
3. Risk Limits Engine Boundary Conditions
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats

from sentinel.quant.backtest import RiskLimitsEngine, run_es_backtest, run_kupiec_pof_test


def test_es_fat_tail_detection():
    print("\n--- TEST 1: ES FAT-TAIL DETECTION ---")
    print("Market has extreme 'Fat Tails' (Student-T df=3).")
    print("Model assumes perfectly Normal market.")

    np.random.seed(42)
    # 1. Simulate Reality: Fat-tailed market (10,000 days, scaled to 1% daily vol)
    true_returns = stats.t.rvs(df=3, size=10_000) * 0.01

    # 2. Model's Delusion: It thinks volatility is 1.73% (the standard dev of the T-dist)
    # but assumes the shape is perfectly Normal.
    model_vol = np.std(true_returns)

    # 95% VaR and ES for a Normal Distribution
    normal_var_pred = stats.norm.ppf(0.95) * model_vol
    phi = stats.norm.pdf(stats.norm.ppf(0.95))
    normal_es_pred = (phi / 0.05) * model_vol

    # Create constant arrays for the predictions
    var_preds = np.full(10_000, normal_var_pred)
    es_preds = np.full(10_000, normal_es_pred)

    # 3. Run the Backtests
    # The normal VaR prediction might actually pass Kupiec because the threshold
    # itself might catch roughly 5% of days. Let's see.
    kupiec = run_kupiec_pof_test(true_returns, var_preds, confidence_level=0.95)

    # BUT, the days that DO breach will be much worse than the Normal ES predicts.
    es_test = run_es_backtest(true_returns, var_preds, es_preds, significance_level=0.05)

    print(f"Kupiec VaR Test: p-value = {kupiec.p_value:.4f} -> {'ACCEPTED' if kupiec.is_accepted else 'REJECTED'}")
    print("McNeil-Frey ES Test:")
    print(f"  Breach count: {es_test.breach_count}")
    print(f"  Mean Exceedance Residual (Error): {es_test.mean_exceedance_residual:.4%}")
    print(f"  p-value: {es_test.p_value:.6f}")

    if not es_test.is_accepted and kupiec.is_accepted:
        print("=> PASS: The ES test successfully caught the fat-tail flaw that VaR missed!")
    elif not es_test.is_accepted:
        print("=> PASS: The ES test successfully rejected the flawed Normal assumption.")
    else:
        print("=> FAIL: The ES test failed to detect the fat tails.")


def test_kupiec_sensitivity():
    print("\n--- TEST 2: KUPIEC STATISTICAL SENSITIVITY ---")
    print("Testing a 99% VaR model that secretly has a 2% failure rate.")

    # 5,000 days (~20 years of data)
    n = 5_000
    np.random.seed(99)

    # True failures should be 1% (50 breaches).
    # We will simulate a model that is slightly too aggressive and breaches 2% of the time (100 breaches).
    # We create artificial PnL and VaR to explicitly force exactly 100 breaches.
    pnl = np.zeros(n)
    var = np.full(n, 10.0)

    # Force 100 breaches
    breach_indices = np.random.choice(n, size=100, replace=False)
    pnl[breach_indices] = -20.0

    # Run test at 99% confidence (expecting 50 breaches)
    kupiec = run_kupiec_pof_test(pnl, var, confidence_level=0.99)

    print("Confidence Level: 99%")
    print(f"Expected Breaches: {kupiec.expected_breaches:.1f}")
    print(f"Actual Breaches:   {kupiec.actual_breaches}")
    print(f"LR Statistic:      {kupiec.lr_statistic:.2f}")
    print(f"p-value:           {kupiec.p_value:.8f}")

    if not kupiec.is_accepted:
        print("=> PASS: Kupiec POF successfully rejected the structurally biased model.")
    else:
        print("=> FAIL: Kupiec POF accepted a biased model.")


def test_risk_limits_boundaries():
    print("\n--- TEST 3: RISK LIMITS BOUNDARY CONDITIONS ---")
    print("Testing floating point edge cases on hard limits.")

    engine = RiskLimitsEngine()
    engine.set_upper_limit("VaR", 1_000_000.0)

    # Test 1: Exactly on the limit
    b1 = engine.check_limits({"VaR": 1_000_000.0})
    print(f"Exact limit (1.0M): Breaches = {len(b1)}")

    # Test 2: 1 cent over the limit
    b2 = engine.check_limits({"VaR": 1_000_000.01})
    if len(b2) > 0:
        print(f"1 cent over limit:  Breached. Severity = {b2[0].severity}")

    # Test 3: 20% over limit (Critical threshold)
    b3 = engine.check_limits({"VaR": 1_200_000.01})
    if len(b3) > 0:
        print(f"20% over limit:     Breached. Severity = {b3[0].severity}")

    if len(b1) == 0 and len(b2) == 1 and len(b3) == 1 and b3[0].severity == "Critical":
        print("=> PASS: Boundary conditions evaluated precisely.")

if __name__ == "__main__":
    test_es_fat_tail_detection()
    test_kupiec_sensitivity()
    test_risk_limits_boundaries()
