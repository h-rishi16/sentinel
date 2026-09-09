"""
Tests for sentinel.quant.backtest

Verifies Kupiec POF binomial math and Risk Limits engine logic.
"""

import numpy as np
import pytest

from sentinel.quant.backtest import (
    RiskLimitsEngine,
    run_kupiec_pof_test,
)


class TestKupiecPOF:
    def test_perfect_model_accepted(self):
        """A model with exactly the expected number of breaches should be strongly accepted."""
        n = 1000
        conf = 0.99
        expected_breaches = int(n * (1 - conf))  # 10 breaches expected
        
        # Create fake PnL and VaR
        # 10 days of breaches (PnL = -20, VaR = 10)
        pnl_breach = np.full(expected_breaches, -20.0)
        var_breach = np.full(expected_breaches, 10.0)
        
        # 990 days of normal (PnL = 0, VaR = 10)
        pnl_normal = np.zeros(n - expected_breaches)
        var_normal = np.full(n - expected_breaches, 10.0)
        
        pnl = np.concatenate([pnl_breach, pnl_normal])
        var = np.concatenate([var_breach, var_normal])
        
        res = run_kupiec_pof_test(pnl, var, conf)
        
        assert res.actual_breaches == 10
        assert round(res.expected_breaches) == 10
        # LR stat should be exactly 0 if actual == expected
        np.testing.assert_almost_equal(res.lr_statistic, 0.0)
        assert res.p_value == 1.0
        assert res.is_accepted is True

    def test_terrible_model_rejected(self):
        """A model with 50 breaches when 10 were expected should be rejected."""
        n = 1000
        conf = 0.99
        
        pnl_breach = np.full(50, -20.0)
        var_breach = np.full(50, 10.0)
        
        pnl_normal = np.zeros(950)
        var_normal = np.full(950, 10.0)
        
        pnl = np.concatenate([pnl_breach, pnl_normal])
        var = np.concatenate([var_breach, var_normal])
        
        res = run_kupiec_pof_test(pnl, var, conf)
        
        assert res.actual_breaches == 50
        assert res.is_accepted is False
        assert res.p_value < 0.0001  # Extremely unlikely by chance

    def test_overly_conservative_model_rejected(self):
        """A model with 0 breaches when 50 were expected should also be rejected."""
        n = 1000
        conf = 0.95  # Expect 50 breaches
        
        # 0 breaches
        pnl = np.zeros(1000)
        var = np.full(1000, 10.0)
        
        res = run_kupiec_pof_test(pnl, var, conf)
        
        assert res.actual_breaches == 0
        assert round(res.expected_breaches) == 50
        assert res.is_accepted is False
        assert res.p_value < 0.0001


class TestRiskLimitsEngine:
    def test_upper_limit_warning_and_critical(self):
        engine = RiskLimitsEngine()
        engine.set_upper_limit("VaR_99", 100_000)
        
        # Safe
        breaches = engine.check_limits({"VaR_99": 90_000})
        assert len(breaches) == 0
        
        # Warning (10% over)
        breaches = engine.check_limits({"VaR_99": 110_000})
        assert len(breaches) == 1
        assert breaches[0].severity == "Warning"
        
        # Critical (30% over)
        breaches = engine.check_limits({"VaR_99": 130_000})
        assert len(breaches) == 1
        assert breaches[0].severity == "Critical"

    def test_lower_limit(self):
        engine = RiskLimitsEngine()
        engine.set_lower_limit("Diversification_Ratio", 1.2)
        
        # Safe
        breaches = engine.check_limits({"Diversification_Ratio": 1.5})
        assert len(breaches) == 0
        
        # Breach
        breaches = engine.check_limits({"Diversification_Ratio": 1.1})
        assert len(breaches) == 1
        assert breaches[0].breach_type == "Lower"
        assert breaches[0].metric_name == "Diversification_Ratio"
