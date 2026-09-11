"""
Backtest demonstration on real historical data.
"""

import warnings

warnings.filterwarnings('ignore')

from sentinel.data.market import forward_fill_prices, load_prices
from sentinel.quant.portfolio import PortfolioDefinition
from sentinel.quant.rolling_backtest import run_rolling_backtest


def main():
    print("="*60)
    print("SENTINEL: REAL DATA VaR BACKTEST")
    print("="*60)

    # 1. Load Data
    prices = forward_fill_prices(load_prices())
    portfolio = PortfolioDefinition(weights={"SPY": 0.40, "AAPL": 0.30, "MSFT": 0.30})

    # 2. Run Historical Backtest (1-year sliding window)
    print("Running Historical Simulation VaR Backtest...")
    res_hist = run_rolling_backtest(
        prices, portfolio, window_size=252, confidence_level=0.95, method="historical"
    )

    # 3. Run Parametric Backtest (1-year sliding window)
    print("Running Parametric (Delta-Normal) VaR Backtest...")
    res_para = run_rolling_backtest(
        prices, portfolio, window_size=252, confidence_level=0.95, method="parametric"
    )

    # 4. Run Monte Carlo Backtest (1-year sliding window)
    print("Running Monte Carlo VaR Backtest (10k paths per day, takes ~5 seconds)...")
    res_mc = run_rolling_backtest(
        prices, portfolio, window_size=252, confidence_level=0.95, method="monte_carlo"
    )

    # 5. Results
    print("\n" + "="*60)
    print("RESULTS (95% Confidence, 252-Day Window)")
    print(f"Total out-of-sample days tested: {res_hist.kupiec_result.observations}")
    print(f"Expected breaches: {res_hist.kupiec_result.expected_breaches:.1f}")

    for res in [res_hist, res_para, res_mc]:
        print(f"\n[{res.method_name.upper()} VaR & ES]")

        # Kupiec POF VaR Test
        k = res.kupiec_result
        var_pass = "✅ PASS" if k.is_accepted else "❌ FAIL"
        print(f"  VaR Breaches:   {k.actual_breaches} (p-val: {k.p_value:.4f}) -> {var_pass}")

        # McNeil-Frey ES Test
        e = res.es_result
        if e.breach_count > 1:
            es_pass = "✅ PASS" if e.is_accepted else "❌ FAIL"
            print(f"  ES Mean Resid:  {e.mean_exceedance_residual:.4%} (p-val: {e.p_value:.4f}) -> {es_pass}")
        else:
            print("  ES Mean Resid:  N/A (Not enough breaches to test ES)")
    print("="*60)

if __name__ == "__main__":
    main()
