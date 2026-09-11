"""
Tests for sentinel.quant.stress

Verifies hypothetical shock application and historical window extraction.
"""

import numpy as np
import pandas as pd
import pytest

from sentinel.quant.portfolio import PortfolioDefinition
from sentinel.quant.stress import (
    Scenario,
    Shock,
    apply_hypothetical_scenario,
    historical_scenario_impact,
)


@pytest.fixture
def sample_portfolio():
    return PortfolioDefinition(weights={"AAPL": 0.6, "MSFT": 0.4})


@pytest.fixture
def sample_prices():
    """10 days of prices. Days 4-7 represent a 50% crash."""
    dates = pd.date_range("2020-03-01", periods=10, freq="D")

    # AAPL goes 100 -> 50 -> 100
    aapl = [100, 100, 100, 100, 75, 50, 75, 100, 100, 100]

    # MSFT goes 100 -> 80 -> 100
    msft = [100, 100, 100, 100, 90, 80, 90, 100, 100, 100]

    return pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=dates)


class TestHypotheticalScenarios:

    def test_single_asset_shock(self, sample_portfolio):
        # Only shock AAPL by -20%
        scen = Scenario("Apple shock", [Shock("AAPL", -0.20)])

        result = apply_hypothetical_scenario(sample_portfolio, scen, 100_000)

        # Port drop = 0.6 * -0.20 = -0.12 (-12%)
        np.testing.assert_almost_equal(result.portfolio_pnl_pct, -0.12)
        np.testing.assert_almost_equal(result.portfolio_pnl_dollar, -12_000)

        assert result.asset_pnls_pct["AAPL"] == -0.20
        assert result.asset_pnls_pct["MSFT"] == 0.0

    def test_multi_asset_shock(self, sample_portfolio):
        scen = Scenario("Tech Crash", [
            Shock("AAPL", -0.30),
            Shock("MSFT", -0.10)
        ])

        result = apply_hypothetical_scenario(sample_portfolio, scen)

        # 0.6 * -0.3 + 0.4 * -0.1 = -0.18 - 0.04 = -0.22
        np.testing.assert_almost_equal(result.portfolio_pnl_pct, -0.22)


class TestHistoricalScenarios:

    def test_historical_drawdown_capture(self, sample_portfolio, sample_prices):
        # The crash happens exactly between 2020-03-04 and 2020-03-06
        # AAPL drops 50% (100 -> 50). MSFT drops 20% (100 -> 80).
        # Portfolio drop = 0.6 * (-0.5) + 0.4 * (-0.2) = -0.3 + -0.08 = -0.38 (-38%)

        result = historical_scenario_impact(
            prices=sample_prices,
            portfolio=sample_portfolio,
            start_date="2020-03-04",
            end_date="2020-03-06",
            scenario_name="COVID Drop",
            portfolio_value=1_000_000
        )

        np.testing.assert_almost_equal(result.portfolio_pnl_pct, -0.38)
        np.testing.assert_almost_equal(result.portfolio_pnl_dollar, -380_000)

    def test_missing_data_raises(self, sample_portfolio, sample_prices):
        with pytest.raises(ValueError, match="No price data found"):
            historical_scenario_impact(
                prices=sample_prices,
                portfolio=sample_portfolio,
                start_date="1999-01-01",
                end_date="1999-12-31",
                scenario_name="Old Crash"
            )
