"""
Tests for sentinel.credit.models

Verifies that the Machine Learning pipelines for Probability of Default (PD)
are correctly instantiating, fitting, predicting, and returning bounded probabilities.
"""

import numpy as np
import pandas as pd
import pytest

from sentinel.credit.models import (
    LogisticPDModel, 
    XGBoostPDModel, 
    LGDModel, 
    EADModel, 
    ExpectedLossEngine
)

# dummy_credit_data fixture is provided by conftest.py


class TestLogisticPDModel:
    
    def test_fit_and_predict(self, dummy_credit_data):
        X, y = dummy_credit_data
        model = LogisticPDModel()
        
        # Fit should work
        model.fit(X, y)
        
        # Predict proba should return 1D array of floats [0, 1]
        probs = model.predict_proba(X)
        assert probs.shape == (len(X),)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        
    def test_feature_importance(self, dummy_credit_data):
        X, y = dummy_credit_data
        model = LogisticPDModel()
        model.fit(X, y)
        
        coefs = model.get_feature_importance()
        assert "fico" in coefs
        assert "dti" in coefs
        
        # Since default is driven by HIGH dti, DTI coefficient should be positive
        assert coefs["dti"] > 0
        

class TestXGBoostPDModel:
    
    def test_fit_and_predict(self, dummy_credit_data):
        X, y = dummy_credit_data
        model = XGBoostPDModel(n_estimators=10, max_depth=2)
        
        # Fit should work
        model.fit(X, y)
        
        # Predict proba should return 1D array of floats [0, 1]
        probs = model.predict_proba(X)
        assert probs.shape == (len(X),)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        
    def test_feature_importance(self, dummy_credit_data):
        X, y = dummy_credit_data
        model = XGBoostPDModel(n_estimators=10)
        model.fit(X, y)
        
        importances = model.get_feature_importance()
        assert "fico" in importances
        assert "dti" in importances
        
        # Importances must be positive and sum to ~1.0
        assert importances["dti"] >= 0
        assert importances["fico"] >= 0
        assert np.isclose(sum(importances.values()), 1.0, atol=1e-5)


class TestLGDEADModels:
    
    def test_lgd_model_bounds(self, dummy_credit_data):
        X, _ = dummy_credit_data
        # Mock some LGD targets between 0 and 1
        np.random.seed(42)
        y_lgd = pd.Series(np.random.uniform(0.1, 0.9, len(X)))
        
        model = LGDModel(n_estimators=10, max_depth=2)
        model.fit(X, y_lgd)
        
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        # Because we used reg:logistic, it MUST be bounded between 0 and 1
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_ead_model_bounds(self, dummy_credit_data):
        X, _ = dummy_credit_data
        # Mock some EAD Factor targets between 0 and 1
        np.random.seed(42)
        y_ead = pd.Series(np.random.uniform(0.5, 1.0, len(X)))
        
        model = EADModel(n_estimators=10, max_depth=2)
        model.fit(X, y_ead)
        
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)


class TestExpectedLossEngine:
    
    def test_expected_loss_math(self, dummy_credit_data):
        X, y = dummy_credit_data
        
        pd_model = XGBoostPDModel(n_estimators=5, max_depth=2)
        pd_model.fit(X, y)
        
        lgd_model = LGDModel(n_estimators=5, max_depth=2)
        lgd_model.fit(X, pd.Series(np.random.uniform(0.2, 0.8, len(X))))
        
        ead_model = EADModel(n_estimators=5, max_depth=2)
        ead_model.fit(X, pd.Series(np.random.uniform(0.6, 1.0, len(X))))
        
        engine = ExpectedLossEngine(pd_model, lgd_model, ead_model)
        
        loan_amounts = pd.Series(np.full(len(X), 10_000.0), index=X.index)
        
        results = engine.predict_expected_loss(X, loan_amounts)
        
        # Verify columns exist
        expected_cols = ["PD", "LGD", "EAD_Factor", "EAD_Dollars", "Expected_Loss"]
        for col in expected_cols:
            assert col in results.columns
            
        # Verify math: EL = PD * LGD * (EAD_Factor * Original_Amount)
        # Using numpy isclose to handle tiny float variations
        calculated_ead_dollars = results["EAD_Factor"] * 10_000.0
        assert np.allclose(results["EAD_Dollars"], calculated_ead_dollars)
        
        calculated_el = results["PD"] * results["LGD"] * results["EAD_Dollars"]
        assert np.allclose(results["Expected_Loss"], calculated_el)
        
        # EL must be <= original loan amount
        assert np.all(results["Expected_Loss"] <= 10_000.0)
        assert np.all(results["Expected_Loss"] >= 0.0)
