"""
Tests for sentinel.quant.returns

These tests verify both software correctness (does the code run?) and
quantitative correctness (does the math produce the right numbers?).

We use hand-calculated expected values for small datasets where the
correct answer can be verified by hand. This is critical for quantitative
software — you cannot rely solely on "it runs without errors."
"""

import numpy as np
import pandas as pd
import pytest

from sentinel.quant.returns import (
    annualization_factor,
    cumulative_returns,
    log_returns,
    return_statistics,
    simple_returns,
)


# ---------------------------------------------------------------------------
# Fixtures: Small, hand-verifiable datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_prices() -> pd.DataFrame:
    """
    Three-day price series for two assets.

    Asset A: 100 → 110 → 99
        Simple returns: +10%, -10%
        Log returns: ln(1.1) ≈ 0.09531, ln(0.9) ≈ -0.10536

    Asset B: 50 → 55 → 60.5
        Simple returns: +10%, +10%
        Log returns: ln(1.1) ≈ 0.09531, ln(1.1) ≈ 0.09531
    """
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    return pd.DataFrame(
        {"A": [100.0, 110.0, 99.0], "B": [50.0, 55.0, 60.5]},
        index=dates,
    )


@pytest.fixture
def longer_prices() -> pd.DataFrame:
    """252 trading days of synthetic but realistic price data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-02", periods=252, freq="B")
    # Simulate GBM-like paths (drift + noise)
    log_returns_a = np.random.normal(0.0004, 0.02, 252)
    log_returns_b = np.random.normal(0.0002, 0.015, 252)
    prices_a = 100.0 * np.exp(np.cumsum(log_returns_a))
    prices_b = 200.0 * np.exp(np.cumsum(log_returns_b))
    return pd.DataFrame({"A": prices_a, "B": prices_b}, index=dates)


# ---------------------------------------------------------------------------
# Simple Returns
# ---------------------------------------------------------------------------

class TestSimpleReturns:
    """Tests for simple (arithmetic) returns."""

    def test_hand_calculated_values(self, simple_prices: pd.DataFrame) -> None:
        """Verify returns match hand-calculated values."""
        result = simple_returns(simple_prices)

        # Asset A: 100 → 110 is +10%, 110 → 99 is -10%
        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result["A"].iloc[0], 0.10, decimal=10)
        np.testing.assert_almost_equal(result["A"].iloc[1], -0.10, decimal=10)

        # Asset B: 50 → 55 is +10%, 55 → 60.5 is +10%
        np.testing.assert_almost_equal(result["B"].iloc[0], 0.10, decimal=10)
        np.testing.assert_almost_equal(result["B"].iloc[1], 0.10, decimal=10)

    def test_first_row_dropped(self, simple_prices: pd.DataFrame) -> None:
        """First row has no return — it should be dropped, not NaN."""
        result = simple_returns(simple_prices)
        assert not result.isna().any().any()
        assert len(result) == len(simple_prices) - 1

    def test_non_symmetry_of_simple_returns(self) -> None:
        """
        Demonstrate that simple returns are NOT symmetric.
        A +50% return followed by -50% does NOT return to the starting value.
        This is WHY log returns exist.
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"X": [100.0, 150.0, 75.0]}, index=dates)
        result = simple_returns(prices)

        # +50% then -50% = you end up at 75, not 100
        assert result["X"].iloc[0] == pytest.approx(0.50)
        assert result["X"].iloc[1] == pytest.approx(-0.50)
        # Sum of simple returns = 0, but actual cumulative return = -25%
        assert result["X"].sum() == pytest.approx(0.0)
        # This demonstrates the asymmetry problem with simple returns

    def test_rejects_nan_prices(self) -> None:
        """Prices with NaN should raise ValueError, not silently produce garbage."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"A": [100.0, np.nan, 110.0]}, index=dates)
        with pytest.raises(ValueError, match="NaN"):
            simple_returns(prices)

    def test_rejects_negative_prices(self) -> None:
        """Negative stock prices are impossible — catch data errors."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"A": [100.0, -10.0, 110.0]}, index=dates)
        with pytest.raises(ValueError, match="non-positive"):
            simple_returns(prices)

    def test_rejects_empty_dataframe(self) -> None:
        """Empty DataFrame should fail explicitly."""
        with pytest.raises(ValueError, match="empty"):
            simple_returns(pd.DataFrame())

    def test_rejects_single_observation(self) -> None:
        """Need at least 2 prices to compute a return."""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        prices = pd.DataFrame({"A": [100.0]}, index=dates)
        with pytest.raises(ValueError, match="at least 2"):
            simple_returns(prices)


# ---------------------------------------------------------------------------
# Log Returns
# ---------------------------------------------------------------------------

class TestLogReturns:
    """Tests for logarithmic (continuously compounded) returns."""

    def test_hand_calculated_values(self, simple_prices: pd.DataFrame) -> None:
        """Verify log returns match hand-calculated values."""
        result = log_returns(simple_prices)

        # Asset A: ln(110/100) = ln(1.1) ≈ 0.09531
        expected_a_0 = np.log(110.0 / 100.0)
        np.testing.assert_almost_equal(result["A"].iloc[0], expected_a_0, decimal=10)

        # Asset A: ln(99/110) = ln(0.9) ≈ -0.10536
        expected_a_1 = np.log(99.0 / 110.0)
        np.testing.assert_almost_equal(result["A"].iloc[1], expected_a_1, decimal=10)

    def test_additivity_over_time(self, simple_prices: pd.DataFrame) -> None:
        """
        KEY PROPERTY: Log returns are additive over time.
        Sum of daily log returns = log return over the full period.
        This is the primary reason quants use log returns.
        """
        result = log_returns(simple_prices)

        for col in result.columns:
            # Sum of daily log returns
            sum_log_returns = result[col].sum()

            # Direct log return over the full period
            direct_log_return = np.log(
                simple_prices[col].iloc[-1] / simple_prices[col].iloc[0]
            )

            np.testing.assert_almost_equal(
                sum_log_returns, direct_log_return, decimal=10
            )

    def test_relationship_to_simple_returns(
        self, simple_prices: pd.DataFrame
    ) -> None:
        """
        Verify the mathematical relationship: r = ln(1 + R)
        where r = log return and R = simple return.
        """
        simple = simple_returns(simple_prices)
        log = log_returns(simple_prices)

        # r_t = ln(1 + R_t) for all t and all assets
        expected_log = np.log(1 + simple)
        pd.testing.assert_frame_equal(log, expected_log, atol=1e-12)

    def test_approximate_equality_for_small_returns(self) -> None:
        """
        For small returns (<2%), simple and log returns are approximately equal.
        This is important — it means for typical daily stock returns, the
        distinction barely matters numerically.
        """
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        # Small price change: 0.5%
        prices = pd.DataFrame({"X": [100.0, 100.5]}, index=dates)

        simple = simple_returns(prices)["X"].iloc[0]
        log = log_returns(prices)["X"].iloc[0]

        # For a 0.5% return, the difference is negligible
        assert abs(simple - log) < 0.0001


# ---------------------------------------------------------------------------
# Cumulative Returns
# ---------------------------------------------------------------------------

class TestCumulativeReturns:
    """Tests for cumulative return calculation."""

    def test_simple_cumulative_returns(self, simple_prices: pd.DataFrame) -> None:
        """
        Cumulative simple returns should recover the actual price change.
        If price goes 100 → 110 → 99, cumulative return = -1%.
        """
        rets = simple_returns(simple_prices)
        cumul = cumulative_returns(rets, method="simple")

        # Final cumulative return for A: (99 - 100) / 100 = -1%
        expected_a = (99.0 / 100.0) - 1.0
        np.testing.assert_almost_equal(cumul["A"].iloc[-1], expected_a, decimal=10)

        # Final cumulative return for B: (60.5 - 50) / 50 = 21%
        expected_b = (60.5 / 50.0) - 1.0
        np.testing.assert_almost_equal(cumul["B"].iloc[-1], expected_b, decimal=10)

    def test_log_cumulative_returns(self, simple_prices: pd.DataFrame) -> None:
        """Log cumulative returns should also recover the actual price change."""
        rets = log_returns(simple_prices)
        cumul = cumulative_returns(rets, method="log")

        expected_a = (99.0 / 100.0) - 1.0
        np.testing.assert_almost_equal(cumul["A"].iloc[-1], expected_a, decimal=10)

    def test_invalid_method_raises(self, simple_prices: pd.DataFrame) -> None:
        """Invalid method should raise, not silently produce garbage."""
        rets = simple_returns(simple_prices)
        with pytest.raises(ValueError, match="Method must be"):
            cumulative_returns(rets, method="invalid")


# ---------------------------------------------------------------------------
# Annualization
# ---------------------------------------------------------------------------

class TestAnnualization:
    """Tests for annualization factors."""

    def test_daily_factor(self) -> None:
        assert annualization_factor("daily") == 252.0

    def test_weekly_factor(self) -> None:
        assert annualization_factor("weekly") == 52.0

    def test_monthly_factor(self) -> None:
        assert annualization_factor("monthly") == 12.0

    def test_invalid_frequency(self) -> None:
        with pytest.raises(ValueError, match="Unknown frequency"):
            annualization_factor("hourly")


# ---------------------------------------------------------------------------
# Return Statistics
# ---------------------------------------------------------------------------

class TestReturnStatistics:
    """Tests for summary statistics computation."""

    def test_basic_statistics(self, longer_prices: pd.DataFrame) -> None:
        """Verify statistics are computed and have sensible values."""
        rets = log_returns(longer_prices)
        stats = return_statistics(rets)

        # Should have one row per asset
        assert len(stats) == 2
        assert "A" in stats.index
        assert "B" in stats.index

        # Standard deviation should be positive
        assert (stats["std_daily"] > 0).all()

        # Annualized std should be daily std * sqrt(252)
        for asset in stats.index:
            expected = stats.loc[asset, "std_daily"] * np.sqrt(252)
            np.testing.assert_almost_equal(
                stats.loc[asset, "std_annualized"], expected, decimal=10
            )

    def test_count_matches_data(self, longer_prices: pd.DataFrame) -> None:
        """Observation count should match the return series length."""
        rets = log_returns(longer_prices)
        stats = return_statistics(rets)
        assert stats["count"].iloc[0] == len(rets)
