"""
Tests for sentinel.fraud (features + detector).
"""

import numpy as np

from sentinel.data.generator import GeneratorConfig, generate_borrowers, generate_transactions
from sentinel.fraud.detector import FraudDetector
from sentinel.fraud.features import engineer_fraud_features, get_fraud_feature_columns


class TestTransactionGenerator:

    def test_generates_correct_size(self):
        cfg = GeneratorConfig(n_transactions=1000, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        assert len(txns) == 1000

    def test_fraud_rate_is_realistic(self):
        cfg = GeneratorConfig(n_transactions=10_000, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        fraud_rate = txns["is_fraud"].mean()
        # Should be approximately 0.2%
        assert 0.001 <= fraud_rate <= 0.005

    def test_fraud_amounts_are_elevated(self):
        cfg = GeneratorConfig(n_transactions=10_000, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        fraud_mean = txns[txns["is_fraud"] == 1]["amount"].mean()
        legit_mean = txns[txns["is_fraud"] == 0]["amount"].mean()
        # Fraud transactions should have significantly higher amounts
        assert fraud_mean > legit_mean * 2


class TestFeatureEngineering:

    def test_all_features_present(self):
        cfg = GeneratorConfig(n_transactions=500, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        featured = engineer_fraud_features(txns)

        for col in get_fraud_feature_columns():
            assert col in featured.columns, f"Missing feature: {col}"

    def test_no_nan_in_features(self):
        cfg = GeneratorConfig(n_transactions=500, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        featured = engineer_fraud_features(txns)

        feature_cols = get_fraud_feature_columns()
        for col in feature_cols:
            assert not featured[col].isna().any(), f"NaN found in feature: {col}"

    def test_zscore_detects_fraud_amounts(self):
        cfg = GeneratorConfig(n_transactions=5000, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        featured = engineer_fraud_features(txns)

        fraud_zscore = featured[featured["is_fraud"] == 1]["amount_zscore"].mean()
        legit_zscore = featured[featured["is_fraud"] == 0]["amount_zscore"].mean()
        # Fraud should have significantly higher Z-scores
        assert fraud_zscore > legit_zscore + 1.0

    def test_night_flag_captures_fraud(self):
        cfg = GeneratorConfig(n_transactions=5000, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        featured = engineer_fraud_features(txns)

        fraud_night_rate = featured[featured["is_fraud"] == 1]["is_night"].mean()
        legit_night_rate = featured[featured["is_fraud"] == 0]["is_night"].mean()
        # Fraud should have a much higher night rate
        assert fraud_night_rate > legit_night_rate * 2


class TestFraudDetector:

    def test_model_trains_and_predicts(self):
        cfg = GeneratorConfig(n_transactions=2000, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        featured = engineer_fraud_features(txns)

        feature_cols = get_fraud_feature_columns()
        X = featured[feature_cols]
        y = featured["is_fraud"]

        model = FraudDetector(max_depth=3, n_estimators=20)
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X),)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_model_beats_random(self):
        cfg = GeneratorConfig(n_transactions=5000, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        featured = engineer_fraud_features(txns)

        feature_cols = get_fraud_feature_columns()
        X = featured[feature_cols]
        y = featured["is_fraud"]

        model = FraudDetector(max_depth=3, n_estimators=50)
        model.fit(X, y)

        metrics = model.evaluate(X, y)
        # ROC-AUC must be significantly better than random (0.5)
        assert metrics.roc_auc > 0.80
        # PR-AUC should be meaningfully above the baseline (fraud_rate)
        assert metrics.pr_auc > y.mean() * 5

    def test_feature_importance_returns_dict(self):
        cfg = GeneratorConfig(n_transactions=1000, seed=42)
        borrowers = generate_borrowers(cfg)
        txns = generate_transactions(borrowers, cfg)
        featured = engineer_fraud_features(txns)

        feature_cols = get_fraud_feature_columns()
        X = featured[feature_cols]
        y = featured["is_fraud"]

        model = FraudDetector(max_depth=3, n_estimators=10)
        model.fit(X, y)

        imp = model.get_feature_importance()
        assert isinstance(imp, dict)
        assert len(imp) == len(feature_cols)
