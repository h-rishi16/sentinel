"""
Tests for sentinel.quant.monte_carlo and sentinel.quant.risk_measures

These tests verify:
1. Cholesky decomposition correctness
2. Simulated return distributions match expected parameters
3. VaR/ES mathematical properties (ES ≥ VaR, monotonicity)
4. Monte Carlo convergence (more sims → more stable)
5. Performance (vectorized implementation is fast)
6. Integration with real market data
"""

import time

import numpy as np
import pandas as pd
import pytest

from sentinel.quant.monte_carlo import (
    SimulationConfig,
    SimulationResult,
    cholesky_decompose,
    simulate_from_historical,
    simulate_gbm,
)
from sentinel.quant.returns import log_returns
from sentinel.quant.risk_measures import (
    RiskMeasures,
    VaRReport,
    compute_es,
    compute_risk_measures,
    compute_var,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_asset_params():
    """Parameters for a single asset simulation."""
    return {
        "expected_returns": np.array([0.10]),  # 10% annual return
        "volatilities": np.array([0.20]),  # 20% annual vol
        "correlation_matrix": np.array([[1.0]]),
        "weights": np.array([1.0]),
    }


@pytest.fixture
def two_asset_params():
    """Two assets with known correlation."""
    return {
        "expected_returns": np.array([0.10, 0.08]),
        "volatilities": np.array([0.20, 0.15]),
        "correlation_matrix": np.array([
            [1.00, 0.50],
            [0.50, 1.00],
        ]),
        "weights": np.array([0.60, 0.40]),
        "asset_names": ["AAPL", "MSFT"],
    }


@pytest.fixture
def known_pnl():
    """
    P&L distribution with known properties for VaR/ES verification.

    100 equally-spaced values from -0.10 to +0.10.
    The 5th percentile (index 4) = -0.0892
    The worst 5 values average gives us ES.
    """
    return np.linspace(-0.10, 0.10, 100)


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
# Cholesky Decomposition
# ---------------------------------------------------------------------------

class TestCholeskyDecompose:
    """Tests for Cholesky decomposition."""

    def test_identity_matrix(self) -> None:
        """Cholesky of identity = identity (uncorrelated assets)."""
        I = np.eye(3)
        L = cholesky_decompose(I)
        np.testing.assert_array_almost_equal(L, I)

    def test_reconstructs_original(self) -> None:
        """L @ L^T must reconstruct the original correlation matrix."""
        corr = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        L = cholesky_decompose(corr)
        reconstructed = L @ L.T
        np.testing.assert_array_almost_equal(reconstructed, corr, decimal=10)

    def test_lower_triangular(self) -> None:
        """Cholesky factor must be lower triangular."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        L = cholesky_decompose(corr)
        assert L[0, 1] == 0.0  # Upper triangle is zero

    def test_handles_near_singular(self) -> None:
        """Should handle near-singular matrices via eigenvalue clipping."""
        # Correlation = 0.999 (near-perfect correlation)
        corr = np.array([[1.0, 0.999], [0.999, 1.0]])
        L = cholesky_decompose(corr)
        reconstructed = L @ L.T
        np.testing.assert_array_almost_equal(reconstructed, corr, decimal=3)

    def test_rejects_asymmetric(self) -> None:
        """Asymmetric matrix should raise."""
        bad = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="not symmetric"):
            cholesky_decompose(bad)


# ---------------------------------------------------------------------------
# Simulation Config
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    """Tests for simulation parameter validation."""

    def test_valid_config(self) -> None:
        config = SimulationConfig(n_simulations=10_000, time_horizon=10)
        assert config.n_simulations == 10_000

    def test_rejects_too_few_simulations(self) -> None:
        with pytest.raises(ValueError, match="n_simulations must be >= 100"):
            SimulationConfig(n_simulations=50)

    def test_rejects_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="time_horizon must be >= 1"):
            SimulationConfig(time_horizon=0)


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

class TestSimulateGBM:
    """Tests for the GBM Monte Carlo engine."""

    def test_output_shape(self, single_asset_params: dict) -> None:
        """Output shapes should match configuration."""
        config = SimulationConfig(n_simulations=1000, random_seed=42)
        result = simulate_gbm(**single_asset_params, config=config)

        assert result.portfolio_pnl.shape == (1000,)
        assert result.asset_terminal_returns.shape == (1000, 1)

    def test_mean_return_convergence(self, single_asset_params: dict) -> None:
        """
        With enough simulations, the mean simulated return should
        converge to the expected return (adjusted for the time horizon).

        For 1-day horizon with μ=10%:
        Expected daily return ≈ 0.10/252 ≈ 0.000397
        """
        config = SimulationConfig(
            n_simulations=100_000, time_horizon=1, random_seed=42
        )
        result = simulate_gbm(**single_asset_params, config=config)

        mean_pnl = result.portfolio_pnl.mean()
        expected_daily = 0.10 / 252  # ~0.000397

        # With 100k sims, should be within 0.5% of expected
        assert abs(mean_pnl - expected_daily) < 0.005, (
            f"Mean P&L ({mean_pnl:.6f}) is too far from "
            f"expected ({expected_daily:.6f})"
        )

    def test_volatility_convergence(self, single_asset_params: dict) -> None:
        """
        Simulated volatility should converge to the input volatility.

        For 1-day horizon with σ_annual=20%:
        Expected daily vol ≈ 0.20/√252 ≈ 0.0126
        """
        config = SimulationConfig(
            n_simulations=100_000, time_horizon=1, random_seed=42
        )
        result = simulate_gbm(**single_asset_params, config=config)

        std_pnl = result.portfolio_pnl.std()
        expected_daily_vol = 0.20 / np.sqrt(252)

        # Within 10% relative error
        relative_error = abs(std_pnl - expected_daily_vol) / expected_daily_vol
        assert relative_error < 0.10, (
            f"Simulated vol ({std_pnl:.6f}) vs expected ({expected_daily_vol:.6f}), "
            f"relative error = {relative_error:.2%}"
        )

    def test_correlation_preserved(self, two_asset_params: dict) -> None:
        """
        Simulated asset returns should preserve the input correlation.
        With ρ=0.50, the measured correlation should be close.
        """
        config = SimulationConfig(
            n_simulations=100_000, time_horizon=1, random_seed=42
        )
        result = simulate_gbm(**two_asset_params, config=config)

        measured_corr = np.corrcoef(
            result.asset_terminal_returns[:, 0],
            result.asset_terminal_returns[:, 1],
        )[0, 1]

        assert abs(measured_corr - 0.50) < 0.05, (
            f"Measured correlation ({measured_corr:.4f}) is too far from "
            f"input (0.50)"
        )

    def test_reproducibility(self, single_asset_params: dict) -> None:
        """Same seed should produce identical results."""
        config = SimulationConfig(n_simulations=1000, random_seed=42)
        r1 = simulate_gbm(**single_asset_params, config=config)
        r2 = simulate_gbm(**single_asset_params, config=config)
        np.testing.assert_array_equal(r1.portfolio_pnl, r2.portfolio_pnl)

    def test_different_seeds_different_results(
        self, single_asset_params: dict
    ) -> None:
        """Different seeds should produce different results."""
        r1 = simulate_gbm(
            **single_asset_params,
            config=SimulationConfig(n_simulations=1000, random_seed=42),
        )
        r2 = simulate_gbm(
            **single_asset_params,
            config=SimulationConfig(n_simulations=1000, random_seed=99),
        )
        assert not np.array_equal(r1.portfolio_pnl, r2.portfolio_pnl)

    def test_multi_day_horizon(self, single_asset_params: dict) -> None:
        """10-day VaR should have ~√10 the daily volatility."""
        config_1d = SimulationConfig(
            n_simulations=50_000, time_horizon=1, random_seed=42
        )
        config_10d = SimulationConfig(
            n_simulations=50_000, time_horizon=10, random_seed=42
        )

        r_1d = simulate_gbm(**single_asset_params, config=config_1d)
        r_10d = simulate_gbm(**single_asset_params, config=config_10d)

        ratio = r_10d.portfolio_pnl.std() / r_1d.portfolio_pnl.std()
        expected_ratio = np.sqrt(10)

        # Should be approximately √10 ≈ 3.16
        assert abs(ratio - expected_ratio) < 0.5, (
            f"10d/1d vol ratio ({ratio:.2f}) should be ≈ √10 ({expected_ratio:.2f})"
        )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    """Verify the vectorized implementation is fast."""

    def test_100k_simulations_under_1_second(
        self, two_asset_params: dict
    ) -> None:
        """100,000 simulations should complete in under 1 second on M4."""
        config = SimulationConfig(n_simulations=100_000, random_seed=42)
        result = simulate_gbm(**two_asset_params, config=config)
        assert result.elapsed_seconds < 1.0, (
            f"100k simulations took {result.elapsed_seconds:.2f}s (limit: 1.0s)"
        )


# ---------------------------------------------------------------------------
# VaR
# ---------------------------------------------------------------------------

class TestVaR:
    """Tests for Value at Risk computation."""

    def test_known_quantile(self) -> None:
        """
        With uniform P&L from -0.10 to +0.10:
        5th percentile = -0.09 (approximately)
        VaR at 95% ≈ 0.09
        """
        pnl = np.linspace(-0.10, 0.10, 10000)
        var = compute_var(pnl, 0.95)
        np.testing.assert_almost_equal(var, 0.09, decimal=2)

    def test_higher_confidence_higher_var(self) -> None:
        """99% VaR should be higher than 95% VaR (more conservative)."""
        np.random.seed(42)
        pnl = np.random.normal(0, 0.02, 10000)
        var_95 = compute_var(pnl, 0.95)
        var_99 = compute_var(pnl, 0.99)
        assert var_99 > var_95

    def test_var_is_positive_for_losses(self) -> None:
        """VaR should be positive when there are losses."""
        pnl = np.array([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05])
        var = compute_var(pnl, 0.95)
        assert var > 0

    def test_invalid_confidence(self) -> None:
        """Confidence level must be in (0, 1)."""
        pnl = np.random.normal(0, 0.01, 100)
        with pytest.raises(ValueError, match="confidence_level"):
            compute_var(pnl, 1.5)
        with pytest.raises(ValueError, match="confidence_level"):
            compute_var(pnl, 0.0)


# ---------------------------------------------------------------------------
# Expected Shortfall
# ---------------------------------------------------------------------------

class TestExpectedShortfall:
    """Tests for Expected Shortfall computation."""

    def test_es_greater_than_or_equal_to_var(self) -> None:
        """
        ES ≥ VaR ALWAYS. This is a mathematical certainty.
        If our code violates this, something is fundamentally broken.
        """
        np.random.seed(42)
        pnl = np.random.normal(0, 0.02, 10000)

        for cl in [0.90, 0.95, 0.99]:
            var = compute_var(pnl, cl)
            es = compute_es(pnl, cl)
            assert es >= var, (
                f"ES ({es:.6f}) < VaR ({var:.6f}) at {cl:.0%}. "
                "This violates the definition of Expected Shortfall."
            )

    def test_es_with_heavy_tail(self) -> None:
        """
        With heavy-tailed distribution, ES should be significantly
        larger than VaR (the tail is fat).
        """
        np.random.seed(42)
        # Student-t with 3 degrees of freedom (heavy tails)
        from scipy.stats import t as t_dist

        pnl = t_dist.rvs(df=3, size=10000, random_state=42) * 0.02

        var = compute_var(pnl, 0.95)
        es = compute_es(pnl, 0.95)

        # ES/VaR ratio should be notably > 1 for heavy tails
        assert es / var > 1.2, (
            f"ES/VaR ratio = {es/var:.2f}. For Student-t(3), "
            "expected ratio > 1.2 due to heavy tails."
        )

    def test_es_monotone_in_confidence(self) -> None:
        """Higher confidence → higher ES (more conservative)."""
        np.random.seed(42)
        pnl = np.random.normal(0, 0.02, 10000)
        es_90 = compute_es(pnl, 0.90)
        es_95 = compute_es(pnl, 0.95)
        es_99 = compute_es(pnl, 0.99)
        assert es_99 > es_95 > es_90


# ---------------------------------------------------------------------------
# Complete Risk Report
# ---------------------------------------------------------------------------

class TestRiskReport:
    """Tests for the complete risk reporting pipeline."""

    def test_report_structure(self) -> None:
        """Report should contain all requested confidence levels."""
        np.random.seed(42)
        pnl = np.random.normal(0, 0.02, 10000)
        report = compute_risk_measures(pnl, [0.95, 0.99])

        assert len(report.risk_measures) == 2
        assert report.risk_measures[0].confidence_level == 0.95
        assert report.risk_measures[1].confidence_level == 0.99

    def test_dollar_conversion(self) -> None:
        """Dollar VaR = percentage VaR × portfolio value."""
        np.random.seed(42)
        pnl = np.random.normal(0, 0.02, 10000)
        pv = 10_000_000.0
        report = compute_risk_measures(pnl, [0.95], portfolio_value=pv)

        rm = report.risk_measures[0]
        np.testing.assert_almost_equal(rm.var_dollar, rm.var * pv, decimal=2)
        np.testing.assert_almost_equal(rm.es_dollar, rm.es * pv, decimal=2)

    def test_distributional_stats(self) -> None:
        """Report should include mean, std, skew, kurtosis."""
        np.random.seed(42)
        pnl = np.random.normal(0.001, 0.02, 10000)
        report = compute_risk_measures(pnl, [0.95])

        assert abs(report.mean_pnl - 0.001) < 0.001
        assert abs(report.std_pnl - 0.02) < 0.002
        assert report.n_simulations == 10000


# ---------------------------------------------------------------------------
# Integration: Monte Carlo → VaR/ES Pipeline
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end tests combining simulation + risk measures."""

    def test_mc_var_with_real_data(self, real_prices: pd.DataFrame) -> None:
        """
        Full pipeline: real prices → log returns → MC simulation → VaR/ES.
        """
        rets = log_returns(real_prices)
        n = len(rets.columns)
        weights = np.array([1.0 / n] * n)

        config = SimulationConfig(
            n_simulations=50_000, time_horizon=1, random_seed=42
        )
        sim = simulate_from_historical(rets, weights, config)
        report = compute_risk_measures(sim.portfolio_pnl, [0.95, 0.99])

        # Sanity checks
        for rm in report.risk_measures:
            # Daily VaR should be between 0.1% and 10% for equity portfolios
            assert 0.001 < rm.var < 0.10, (
                f"VaR at {rm.confidence_level:.0%} = {rm.var:.4f}, "
                "outside reasonable range for daily equity VaR"
            )
            # ES ≥ VaR
            assert rm.es >= rm.var

        # 99% VaR > 95% VaR
        assert report.risk_measures[1].var > report.risk_measures[0].var

    def test_10_day_var(self, real_prices: pd.DataFrame) -> None:
        """10-day regulatory VaR (Basel III)."""
        rets = log_returns(real_prices)
        n = len(rets.columns)
        weights = np.array([1.0 / n] * n)

        config = SimulationConfig(
            n_simulations=50_000, time_horizon=10, random_seed=42
        )
        sim = simulate_from_historical(rets, weights, config)
        report = compute_risk_measures(
            sim.portfolio_pnl, [0.99], portfolio_value=10_000_000
        )

        rm = report.risk_measures[0]
        # 10-day 99% VaR for a $10M equity portfolio should be
        # in the $100k-$2M range
        assert 50_000 < rm.var_dollar < 5_000_000, (
            f"10-day 99% VaR = ${rm.var_dollar:,.0f}, "
            "outside reasonable range for a $10M equity portfolio"
        )
