"""
Tests for sentinel.quant.portfolio

These tests verify Modern Portfolio Theory calculations against
hand-computed values and known mathematical properties.
"""

import numpy as np
import pandas as pd
import pytest

from sentinel.quant.portfolio import (
    PortfolioDefinition,
    PortfolioRiskMetrics,
    analyze_portfolio,
    correlation_matrix,
    covariance_matrix,
    diversification_ratio,
    maximum_drawdown,
    portfolio_returns,
    portfolio_volatility,
)
from sentinel.quant.returns import log_returns, simple_returns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_asset_prices() -> pd.DataFrame:
    """
    Two assets with known, simple price paths.
    A: 100 → 110 → 105 → 115 → 120
    B: 200 → 190 → 195 → 185 → 200
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "A": [100.0, 110.0, 105.0, 115.0, 120.0],
            "B": [200.0, 190.0, 195.0, 185.0, 200.0],
        },
        index=dates,
    )


@pytest.fixture
def equal_weight_portfolio() -> PortfolioDefinition:
    """50/50 portfolio of two assets."""
    return PortfolioDefinition(weights={"A": 0.50, "B": 0.50}, name="EqualWeight")


@pytest.fixture
def real_prices() -> pd.DataFrame:
    """Load real market data if available."""
    from pathlib import Path

    parquet_path = Path(__file__).parent.parent / "data" / "raw" / "prices.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    else:
        np.random.seed(99)
        dates = pd.date_range("2022-01-03", periods=752, freq="B")
        return pd.DataFrame(
            {
                "AAPL": 150 * np.exp(np.cumsum(np.random.normal(0.0004, 0.017, 752))),
                "MSFT": 250 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, 752))),
                "SPY": 400 * np.exp(np.cumsum(np.random.normal(0.0003, 0.011, 752))),
            },
            index=dates,
        )


# ---------------------------------------------------------------------------
# PortfolioDefinition
# ---------------------------------------------------------------------------

class TestPortfolioDefinition:
    """Tests for portfolio weight validation."""

    def test_valid_portfolio(self) -> None:
        """Standard portfolio should construct without errors."""
        p = PortfolioDefinition(weights={"A": 0.6, "B": 0.4})
        assert p.n_assets == 2
        assert p.assets == ["A", "B"]
        np.testing.assert_array_equal(p.weight_array, [0.6, 0.4])

    def test_weights_must_sum_to_one(self) -> None:
        """Reject portfolios where weights don't sum to 1."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            PortfolioDefinition(weights={"A": 0.5, "B": 0.3})

    def test_empty_portfolio_rejected(self) -> None:
        """Empty portfolio makes no sense."""
        with pytest.raises(ValueError, match="at least one"):
            PortfolioDefinition(weights={})

    def test_single_asset_portfolio(self) -> None:
        """100% in one asset is valid."""
        p = PortfolioDefinition(weights={"SPY": 1.0})
        assert p.n_assets == 1

    def test_tolerates_small_rounding(self) -> None:
        """Allow small rounding errors (±0.01)."""
        p = PortfolioDefinition(weights={"A": 0.333, "B": 0.333, "C": 0.334})
        assert p.n_assets == 3


# ---------------------------------------------------------------------------
# Portfolio Returns
# ---------------------------------------------------------------------------

class TestPortfolioReturns:
    """Tests for portfolio return computation."""

    def test_hand_calculated_portfolio_return(
        self, two_asset_prices: pd.DataFrame, equal_weight_portfolio: PortfolioDefinition
    ) -> None:
        """
        Verify portfolio return against hand calculation.

        Day 1:
            A: (110-100)/100 = +10%
            B: (190-200)/200 = -5%
            Portfolio: 0.5×10% + 0.5×(-5%) = +2.5%
        """
        rets = simple_returns(two_asset_prices)
        port_rets = portfolio_returns(rets, equal_weight_portfolio)

        expected_day1 = 0.5 * 0.10 + 0.5 * (-0.05)
        np.testing.assert_almost_equal(port_rets.iloc[0], expected_day1, decimal=10)

    def test_single_asset_portfolio_equals_asset(
        self, two_asset_prices: pd.DataFrame
    ) -> None:
        """100% in one asset → portfolio return = asset return."""
        p = PortfolioDefinition(weights={"A": 1.0}, name="AllA")
        rets = simple_returns(two_asset_prices)
        port_rets = portfolio_returns(rets, p)
        pd.testing.assert_series_equal(
            port_rets, rets["A"].rename("AllA"), check_names=True
        )

    def test_missing_asset_raises(self, two_asset_prices: pd.DataFrame) -> None:
        """Portfolio referencing non-existent asset should fail."""
        p = PortfolioDefinition(weights={"A": 0.5, "MISSING": 0.5})
        rets = simple_returns(two_asset_prices)
        with pytest.raises(ValueError, match="Missing return data"):
            portfolio_returns(rets, p)


# ---------------------------------------------------------------------------
# Covariance and Correlation
# ---------------------------------------------------------------------------

class TestCovarianceMatrix:
    """Tests for covariance estimation."""

    def test_covariance_is_symmetric(self, two_asset_prices: pd.DataFrame) -> None:
        """Covariance matrices must be symmetric."""
        rets = simple_returns(two_asset_prices)
        cov = covariance_matrix(rets, annualize=False)
        pd.testing.assert_frame_equal(cov, cov.T)

    def test_diagonal_is_variance(self, two_asset_prices: pd.DataFrame) -> None:
        """Diagonal of covariance matrix = individual asset variance."""
        rets = simple_returns(two_asset_prices)
        cov = covariance_matrix(rets, annualize=False)

        for col in rets.columns:
            expected_var = rets[col].var(ddof=1)
            np.testing.assert_almost_equal(cov.loc[col, col], expected_var, decimal=10)

    def test_annualization(self, two_asset_prices: pd.DataFrame) -> None:
        """Annualized covariance = daily covariance × 252."""
        rets = simple_returns(two_asset_prices)
        daily = covariance_matrix(rets, annualize=False)
        annual = covariance_matrix(rets, annualize=True)
        pd.testing.assert_frame_equal(annual, daily * 252)

    def test_invalid_method(self, two_asset_prices: pd.DataFrame) -> None:
        """Invalid method should raise."""
        rets = simple_returns(two_asset_prices)
        with pytest.raises(ValueError, match="Unknown method"):
            covariance_matrix(rets, method="invalid")


class TestCorrelationMatrix:
    """Tests for correlation estimation."""

    def test_diagonal_is_one(self, two_asset_prices: pd.DataFrame) -> None:
        """Diagonal of correlation matrix = 1.0 (asset correlated with itself)."""
        rets = simple_returns(two_asset_prices)
        corr = correlation_matrix(rets)
        for col in rets.columns:
            np.testing.assert_almost_equal(corr.loc[col, col], 1.0, decimal=10)

    def test_correlation_bounded(self, two_asset_prices: pd.DataFrame) -> None:
        """Correlations must be in [-1, 1]."""
        rets = simple_returns(two_asset_prices)
        corr = correlation_matrix(rets)
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()

    def test_symmetric(self, two_asset_prices: pd.DataFrame) -> None:
        """Correlation matrix must be symmetric."""
        rets = simple_returns(two_asset_prices)
        corr = correlation_matrix(rets)
        pd.testing.assert_frame_equal(corr, corr.T)


# ---------------------------------------------------------------------------
# Portfolio Volatility
# ---------------------------------------------------------------------------

class TestPortfolioVolatility:
    """Tests for w^T Σ w computation."""

    def test_single_asset_equals_asset_vol(self) -> None:
        """With one asset, portfolio vol = asset vol."""
        cov = np.array([[0.04]])  # σ² = 0.04, σ = 0.20
        w = np.array([1.0])
        vol = portfolio_volatility(w, cov)
        np.testing.assert_almost_equal(vol, 0.20, decimal=10)

    def test_perfectly_correlated_no_diversification(self) -> None:
        """
        Two assets with ρ=1.0: portfolio vol = weighted average vol.
        No diversification benefit.
        """
        # σ_A = 0.20, σ_B = 0.30, ρ = 1.0
        cov = np.array([
            [0.04, 0.06],   # 0.20 × 0.30 × 1.0 = 0.06
            [0.06, 0.09],
        ])
        w = np.array([0.5, 0.5])
        vol = portfolio_volatility(w, cov)
        expected = 0.5 * 0.20 + 0.5 * 0.30  # = 0.25
        np.testing.assert_almost_equal(vol, expected, decimal=10)

    def test_uncorrelated_diversification_benefit(self) -> None:
        """
        Two assets with ρ=0: portfolio vol < weighted average.
        This IS diversification.
        """
        # σ_A = σ_B = 0.20, ρ = 0.0
        cov = np.array([
            [0.04, 0.00],
            [0.00, 0.04],
        ])
        w = np.array([0.5, 0.5])
        vol = portfolio_volatility(w, cov)

        weighted_avg = 0.5 * 0.20 + 0.5 * 0.20  # = 0.20
        assert vol < weighted_avg  # Diversification benefit!

        # For ρ=0, equal weights: σ_p = σ / √2 ≈ 0.1414
        expected = 0.20 / np.sqrt(2)
        np.testing.assert_almost_equal(vol, expected, decimal=10)

    def test_negative_variance_raises(self) -> None:
        """Non-PSD matrix can produce negative variance — should raise."""
        # Deliberately broken covariance matrix
        cov = np.array([
            [0.04, -0.10],
            [-0.10, 0.04],
        ])
        w = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="Negative portfolio variance"):
            portfolio_volatility(w, cov)


# ---------------------------------------------------------------------------
# Diversification Ratio
# ---------------------------------------------------------------------------

class TestDiversificationRatio:
    """Tests for the diversification ratio."""

    def test_perfectly_correlated_dr_equals_one(self) -> None:
        """With ρ=1, DR should be 1.0 (no benefit)."""
        w = np.array([0.5, 0.5])
        asset_vols = np.array([0.20, 0.30])
        # Port vol for ρ=1 = weighted average = 0.25
        port_vol = 0.5 * 0.20 + 0.5 * 0.30
        dr = diversification_ratio(w, asset_vols, port_vol)
        np.testing.assert_almost_equal(dr, 1.0, decimal=10)

    def test_uncorrelated_dr_greater_than_one(self) -> None:
        """With ρ=0, DR > 1 (diversification helps)."""
        w = np.array([0.5, 0.5])
        asset_vols = np.array([0.20, 0.20])
        port_vol = 0.20 / np.sqrt(2)  # ρ=0 case
        dr = diversification_ratio(w, asset_vols, port_vol)
        assert dr > 1.0
        np.testing.assert_almost_equal(dr, np.sqrt(2), decimal=10)


# ---------------------------------------------------------------------------
# Maximum Drawdown
# ---------------------------------------------------------------------------

class TestMaximumDrawdown:
    """Tests for maximum drawdown computation."""

    def test_known_drawdown(self) -> None:
        """
        Price path: 100 → 120 → 90 → 110
        Peak at 120, trough at 90.
        Max drawdown = (90-120)/120 = -25%
        """
        simple_rets = pd.Series(
            [0.20, -0.25, 0.2222],
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        max_dd, peak, trough = maximum_drawdown(simple_rets)
        np.testing.assert_almost_equal(max_dd, -0.25, decimal=4)

    def test_no_drawdown_for_monotonic_increase(self) -> None:
        """If prices only go up, drawdown = 0."""
        rets = pd.Series(
            [0.01, 0.02, 0.01, 0.03],
            index=pd.date_range("2024-01-01", periods=4, freq="B"),
        )
        max_dd, _, _ = maximum_drawdown(rets)
        np.testing.assert_almost_equal(max_dd, 0.0, decimal=10)


# ---------------------------------------------------------------------------
# Full Portfolio Analysis (Integration)
# ---------------------------------------------------------------------------

class TestAnalyzePortfolio:
    """Integration tests for the full portfolio analysis pipeline."""

    def test_basic_analysis(
        self, two_asset_prices: pd.DataFrame, equal_weight_portfolio: PortfolioDefinition
    ) -> None:
        """Full analysis should return all expected fields."""
        result = analyze_portfolio(two_asset_prices, equal_weight_portfolio)
        assert isinstance(result, PortfolioRiskMetrics)
        assert result.n_assets == 2
        assert result.n_observations == 4  # 5 prices → 4 returns
        assert result.portfolio_volatility_annualized > 0
        assert result.diversification_ratio >= 1.0

    def test_real_data_analysis(self, real_prices: pd.DataFrame) -> None:
        """Run full analysis on real market data."""
        tickers = list(real_prices.columns)
        n = len(tickers)
        weights = {t: 1.0 / n for t in tickers}
        portfolio = PortfolioDefinition(weights=weights, name="EqualWeight")

        result = analyze_portfolio(real_prices, portfolio)

        # Sanity checks on real data
        assert 0.0 < result.portfolio_volatility_annualized < 1.0  # 0-100%
        assert result.max_drawdown < 0  # Must have some drawdown
        assert result.diversification_ratio >= 1.0  # Must show diversification
        assert result.n_observations > 500  # Should have >2 years of data

    def test_diversification_with_real_data(self, real_prices: pd.DataFrame) -> None:
        """
        Portfolio vol should be LESS than weighted average of individual vols.
        This proves diversification is working in the code.
        """
        tickers = list(real_prices.columns)
        n = len(tickers)
        weights = {t: 1.0 / n for t in tickers}
        portfolio = PortfolioDefinition(weights=weights, name="Test")

        result = analyze_portfolio(real_prices, portfolio)

        assert result.portfolio_volatility_annualized < result.weighted_avg_volatility, (
            f"Portfolio vol ({result.portfolio_volatility_annualized:.4f}) is not less "
            f"than weighted avg vol ({result.weighted_avg_volatility:.4f}). "
            "Diversification is broken."
        )
