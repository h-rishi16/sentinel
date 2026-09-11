"""
Tests for sentinel.quant.rolling_backtest
"""
import numpy as np
import pandas as pd

from sentinel.quant.portfolio import PortfolioDefinition
from sentinel.quant.rolling_backtest import historical_var, parametric_var, run_rolling_backtest


def test_parametric_var():
    # 99% Z-score is approx 2.326
    vol = 0.01  # 1% daily vol
    conf = 0.99
    var = parametric_var(vol, conf)
    np.testing.assert_almost_equal(var, 2.32634 * 0.01, decimal=4)

def test_historical_var():
    # 100 days of PnL: from -0.05 to +0.05
    # The 5th percentile (95% VaR) should be roughly the 5th worst day = -0.045
    pnl = np.linspace(-0.05, 0.05, 100)
    var = historical_var(pnl, 0.95)

    # Value should be positive (representing the loss)
    assert var > 0
    np.testing.assert_almost_equal(var, 0.045, decimal=2)

def test_rolling_backtest_alignment():
    """
    Ensure the rolling window doesn't leak future data.
    """
    dates = pd.date_range("2020-01-01", periods=10, freq="B")

    # Asset A goes up slightly every day, but crashes massively on Day 6
    # Day 0 to 4: Training window
    # Day 5: Prediction day (should use days 0-4 to predict day 5)
    prices = pd.DataFrame({
        "A": [100, 101, 102, 103, 104, 50, 50, 50, 50, 50]
    }, index=dates)

    port = PortfolioDefinition(weights={"A": 1.0})

    # Use a tiny window to test the exact crash day
    res = run_rolling_backtest(prices, port, window_size=4, confidence_level=0.95, method="historical")

    df = res.daily_results

    # We expect len(df) = Total prices (10) - window (4) - return drop (1) = 5 days predicted
    assert len(df) == 5

    # The crash happens on index 5 (which is the 2nd row in the results dataframe, since results start at index 4)
    # The actual return from 104 to 50 is roughly -52%
    crash_day_result = df.iloc[0]

    assert crash_day_result["Actual_PnL"] < -0.50

    # The Predicted VaR for the crash day should be very small (based on days 0-4 which were peaceful)
    assert crash_day_result["Predicted_VaR"] < 0.05

    # Thus, it must be flagged as a breach
    assert crash_day_result["Is_Breach"] == True
