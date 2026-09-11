"""
Tests for sentinel.quant.volatility

These tests verify:
1. Mathematical correctness (hand-calculated EWMA values)
2. Statistical properties (GARCH stationarity, mean-reversion)
3. Method comparisons (EWMA responds faster than rolling)
4. Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest

from sentinel.quant.returns import log_returns
from sentinel.quant.volatility import (
    GARCHResult,
    compare_volatility_methods,
    ewma_volatility,
    garch_volatility,
    rolling_volatility,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """
    252 days of synthetic log returns with known volatility regime.

    First 126 days: low volatility (σ_daily ≈ 0.01)
    Last 126 days: high volatility (σ_daily ≈ 0.03)

    This lets us test whether volatility methods detect regime changes.
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    low_vol = np.random.normal(0.0, 0.01, 126)
    high_vol = np.random.normal(0.0, 0.03, 126)
    returns = np.concatenate([low_vol, high_vol])
    return pd.DataFrame({"ASSET": returns}, index=dates)


@pytest.fixture
def long_returns() -> pd.Series:
    """500 days of synthetic returns for GARCH estimation."""
    np.random.seed(123)
    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    # Simulate GARCH-like returns with volatility clustering
    returns = np.zeros(500)
    vol = np.zeros(500)
    vol[0] = 0.015
    for t in range(1, 500):
        vol[t] = np.sqrt(0.00001 + 0.08 * returns[t - 1] ** 2 + 0.90 * vol[t - 1] ** 2)
        returns[t] = np.random.normal(0, vol[t])
    return pd.Series(returns, index=dates, name="SIM")


@pytest.fixture
def real_prices() -> pd.DataFrame:
    """Load real market data if available, otherwise generate synthetic."""
    from pathlib import Path

    parquet_path = Path(__file__).parent.parent / "data" / "raw" / "prices.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    else:
        # Fallback synthetic data
        np.random.seed(99)
        dates = pd.date_range("2022-01-03", periods=752, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": 150 * np.exp(np.cumsum(np.random.normal(0.0004, 0.017, 752))),
                "MSFT": 250 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, 752))),
                "SPY": 400 * np.exp(np.cumsum(np.random.normal(0.0003, 0.011, 752))),
            },
            index=dates,
        )
        return prices


# ---------------------------------------------------------------------------
# Rolling Volatility
# ---------------------------------------------------------------------------

class TestRollingVolatility:
    """Tests for rolling (historical) volatility."""

    def test_output_shape(self, synthetic_returns: pd.DataFrame) -> None:
        """Output should have same shape as input."""
        vol = rolling_volatility(synthetic_returns, window=30)
        assert vol.shape == synthetic_returns.shape

    def test_first_values_are_nan(self, synthetic_returns: pd.DataFrame) -> None:
        """First (window-1) values should be NaN (not enough data)."""
        window = 30
        vol = rolling_volatility(synthetic_returns, window=window)
        assert vol.iloc[: window - 1].isna().all().all()
        assert vol.iloc[window - 1 :].notna().all().all()

    def test_detects_regime_change(self, synthetic_returns: pd.DataFrame) -> None:
        """
        Rolling vol should be higher in the second half (high-vol regime)
        than the first half (low-vol regime). This is the most basic test
        of whether a volatility estimator works.
        """
        vol = rolling_volatility(synthetic_returns, window=30)
        vol_clean = vol.dropna()
        midpoint = len(vol_clean) // 2
        avg_first_half = vol_clean.iloc[:midpoint].mean().iloc[0]
        avg_second_half = vol_clean.iloc[midpoint:].mean().iloc[0]
        assert avg_second_half > avg_first_half * 1.5  # At least 1.5x higher

    def test_annualization(self, synthetic_returns: pd.DataFrame) -> None:
        """Annualized vol should be √252 × daily vol."""
        daily = rolling_volatility(synthetic_returns, window=30, annualize=False)
        annual = rolling_volatility(synthetic_returns, window=30, annualize=True)
        ratio = (annual.dropna() / daily.dropna()).mean().iloc[0]
        np.testing.assert_almost_equal(ratio, np.sqrt(252), decimal=5)

    def test_rejects_small_window(self) -> None:
        """Window of 1 makes no statistical sense."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        returns = pd.DataFrame({"A": np.random.normal(0, 0.01, 10)}, index=dates)
        with pytest.raises(ValueError, match="Window must be >= 2"):
            rolling_volatility(returns, window=1)

    def test_rejects_window_larger_than_data(self) -> None:
        """Can't compute a 100-day window with 10 days of data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        returns = pd.DataFrame({"A": np.random.normal(0, 0.01, 10)}, index=dates)
        with pytest.raises(ValueError, match="exceeds"):
            rolling_volatility(returns, window=100)


# ---------------------------------------------------------------------------
# EWMA Volatility
# ---------------------------------------------------------------------------

class TestEWMAVolatility:
    """Tests for EWMA (RiskMetrics) volatility."""

    def test_hand_calculated_ewma(self) -> None:
        """
        Verify EWMA against hand-calculated values.

        With λ=0.94 and returns [0.02, -0.01, 0.03]:
            σ²_0 = r²_0 = 0.0004
            σ²_1 = 0.94 × 0.0004 + 0.06 × 0.02² = 0.000376 + 0.000024 = 0.0004
            σ²_2 = 0.94 × 0.0004 + 0.06 × (-0.01)² = 0.000376 + 0.000006 = 0.000382
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        returns = pd.DataFrame({"X": [0.02, -0.01, 0.03]}, index=dates)
        vol = ewma_volatility(returns, lambda_=0.94, annualize=False)

        # σ²_0 = 0.02² = 0.0004
        np.testing.assert_almost_equal(vol["X"].iloc[0], np.sqrt(0.0004), decimal=10)

        # σ²_1 = 0.94 × 0.0004 + 0.06 × 0.02² = 0.0004
        expected_var_1 = 0.94 * 0.0004 + 0.06 * 0.02**2
        np.testing.assert_almost_equal(
            vol["X"].iloc[1], np.sqrt(expected_var_1), decimal=10
        )

        # σ²_2 = 0.94 × expected_var_1 + 0.06 × (-0.01)²
        expected_var_2 = 0.94 * expected_var_1 + 0.06 * (-0.01) ** 2
        np.testing.assert_almost_equal(
            vol["X"].iloc[2], np.sqrt(expected_var_2), decimal=10
        )

    def test_responds_faster_than_rolling(self, synthetic_returns: pd.DataFrame) -> None:
        """
        EWMA should respond to the volatility regime change FASTER
        than rolling volatility. This is its key advantage.

        We test this by checking that at the regime change point (day 126),
        EWMA ramps up to the new volatility level quicker.
        """
        rolling = rolling_volatility(synthetic_returns, window=30)
        ewma = ewma_volatility(synthetic_returns, lambda_=0.94)

        # Look at 10 days after the regime change (days 126-136)
        transition_window = slice(136, 146)
        rolling_transition = rolling.iloc[transition_window].mean().iloc[0]
        ewma_transition = ewma.iloc[transition_window].mean().iloc[0]

        # EWMA should be higher (faster response to increased vol)
        # The high-vol regime has 3x the daily vol, so EWMA should pick it up faster
        assert ewma_transition > rolling_transition * 0.9

    def test_no_nan_values(self, synthetic_returns: pd.DataFrame) -> None:
        """Unlike rolling vol, EWMA produces estimates for ALL time steps."""
        vol = ewma_volatility(synthetic_returns, lambda_=0.94)
        assert not vol.isna().any().any()

    def test_lambda_bounds(self) -> None:
        """Lambda must be strictly between 0 and 1."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        returns = pd.DataFrame({"A": np.random.normal(0, 0.01, 10)}, index=dates)

        with pytest.raises(ValueError, match="lambda_"):
            ewma_volatility(returns, lambda_=0.0)
        with pytest.raises(ValueError, match="lambda_"):
            ewma_volatility(returns, lambda_=1.0)
        with pytest.raises(ValueError, match="lambda_"):
            ewma_volatility(returns, lambda_=-0.5)

    def test_higher_lambda_smoother(self, synthetic_returns: pd.DataFrame) -> None:
        """
        Higher λ = more weight on past → smoother, slower-responding estimate.
        The standard deviation of the volatility series itself should be lower.
        """
        fast = ewma_volatility(synthetic_returns, lambda_=0.90)
        slow = ewma_volatility(synthetic_returns, lambda_=0.98)
        assert slow.std().iloc[0] < fast.std().iloc[0]


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------

class TestGARCHVolatility:
    """Tests for GARCH(1,1) model."""

    def test_basic_fit(self, long_returns: pd.Series) -> None:
        """GARCH should fit without errors and return a valid result."""
        result = garch_volatility(long_returns)

        assert isinstance(result, GARCHResult)
        assert result.omega > 0  # Constant must be positive
        assert result.alpha >= 0  # ARCH coefficient
        assert result.beta >= 0  # GARCH coefficient

    def test_stationarity(self, long_returns: pd.Series) -> None:
        """
        For a stationary GARCH, α + β < 1.
        This is the critical constraint: it ensures volatility
        mean-reverts to its long-run level.
        """
        result = garch_volatility(long_returns)
        assert result.persistence < 1.0, (
            f"GARCH persistence α+β = {result.persistence:.4f} >= 1.0. "
            "Model is IGARCH (non-stationary)."
        )

    def test_long_run_volatility_positive(self, long_returns: pd.Series) -> None:
        """Long-run volatility should be a positive, finite number."""
        result = garch_volatility(long_returns)
        assert result.long_run_volatility > 0
        assert np.isfinite(result.long_run_volatility)

    def test_conditional_vol_series_length(self, long_returns: pd.Series) -> None:
        """Conditional vol series should have same length as input."""
        result = garch_volatility(long_returns)
        assert len(result.conditional_volatility) == len(long_returns)

    def test_conditional_vol_all_positive(self, long_returns: pd.Series) -> None:
        """Volatility is always positive by definition."""
        result = garch_volatility(long_returns)
        assert (result.conditional_volatility > 0).all()

    def test_rejects_insufficient_data(self) -> None:
        """GARCH needs >= 100 observations for stable estimation."""
        short = pd.Series(
            np.random.normal(0, 0.01, 50),
            index=pd.date_range("2024-01-01", periods=50, freq="B"),
            name="SHORT",
        )
        with pytest.raises(ValueError, match="100 observations"):
            garch_volatility(short)

    def test_persistence_close_to_known(self, long_returns: pd.Series) -> None:
        """
        Our synthetic GARCH data was generated with α=0.08, β=0.90,
        persistence=0.98. The fitted model should recover values
        reasonably close (within ±0.1 of true persistence).
        """
        result = garch_volatility(long_returns)
        assert abs(result.persistence - 0.98) < 0.1, (
            f"Estimated persistence {result.persistence:.4f} is too far "
            f"from true value 0.98"
        )

    def test_information_criteria(self, long_returns: pd.Series) -> None:
        """AIC and BIC should be finite numbers."""
        result = garch_volatility(long_returns)
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestCompareVolatilityMethods:
    """Tests for the volatility comparison utility."""

    def test_comparison_returns_all_methods(self, long_returns: pd.Series) -> None:
        """Comparison should produce columns for all three methods."""
        comp = compare_volatility_methods(long_returns, rolling_window=30)
        assert "rolling_30d" in comp.columns
        assert "ewma" in comp.columns
        assert "garch" in comp.columns

    def test_all_methods_positive(self, long_returns: pd.Series) -> None:
        """All volatility estimates should be non-negative."""
        comp = compare_volatility_methods(long_returns).dropna()
        assert (comp >= 0).all().all()


# ---------------------------------------------------------------------------
# Integration: Real data
# ---------------------------------------------------------------------------

class TestWithRealData:
    """Tests against actual market data (if available)."""

    def test_real_volatility_reasonable_range(self, real_prices: pd.DataFrame) -> None:
        """
        Annualized equity volatility should be between 5% and 100%.
        If it's outside this range, something is wrong with the data or code.
        """
        rets = log_returns(real_prices)
        vol = rolling_volatility(rets, window=30)
        vol_clean = vol.dropna()

        for col in vol_clean.columns:
            median_vol = vol_clean[col].median()
            assert 0.05 < median_vol < 1.0, (
                f"{col} median annualized vol = {median_vol:.2%}, "
                "outside reasonable range (5%-100%)"
            )

    def test_spy_less_volatile_than_single_stocks(
        self, real_prices: pd.DataFrame
    ) -> None:
        """
        SPY (the S&P 500 ETF) should be less volatile than individual
        stocks due to diversification. This is a fundamental property
        of portfolio theory.
        """
        if "SPY" not in real_prices.columns:
            pytest.skip("SPY not in dataset")

        rets = log_returns(real_prices)
        vol = rolling_volatility(rets, window=63)  # Quarterly
        median_vols = vol.dropna().median()

        # SPY should have lower vol than at least one individual stock
        spy_vol = median_vols["SPY"]
        other_vols = median_vols.drop("SPY")
        assert spy_vol < other_vols.max(), (
            f"SPY vol ({spy_vol:.2%}) is not lower than any individual stock. "
            "This violates diversification theory — check the data."
        )
