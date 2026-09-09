"""
Fraud Detection Model for Sentinel.

Uses XGBoost with class imbalance handling (scale_pos_weight)
to detect fraudulent transactions.

Key design decisions:
1. scale_pos_weight compensates for the extreme class imbalance (~0.2% fraud).
   Without it, the model would learn to always predict "not fraud" and achieve
   99.8% accuracy while catching zero actual fraud.
2. We optimize for Precision-Recall AUC (PR-AUC), not ROC-AUC, because with
   extreme class imbalance, ROC-AUC can be misleadingly high even for poor models.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class FraudModelMetrics:
    """Evaluation metrics for a fraud detection model."""
    roc_auc: float
    pr_auc: float  # Precision-Recall AUC (more meaningful for imbalanced data)
    precision_at_threshold: float
    recall_at_threshold: float
    threshold: float
    n_fraud: int
    n_legitimate: int


class FraudDetector:
    """
    XGBoost-based fraud detection model with class imbalance handling.
    """
    def __init__(self, max_depth: int = 4, n_estimators: int = 100, learning_rate: float = 0.1):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None
        self.feature_names_ = []
        self.scale_pos_weight_ = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the fraud detection model.
        
        Automatically computes scale_pos_weight from the class distribution.
        """
        self.feature_names_ = list(X.columns)
        
        # Compute class imbalance ratio
        n_legit = int((y == 0).sum())
        n_fraud = int((y == 1).sum())
        self.scale_pos_weight_ = n_legit / max(n_fraud, 1)
        
        logger.info(f"Training fraud detector: {n_legit} legit, {n_fraud} fraud, "
                     f"scale_pos_weight={self.scale_pos_weight_:.1f}")
        
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight_,
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42
        )
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns fraud probability for each transaction."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> FraudModelMetrics:
        """
        Evaluate the model using both ROC-AUC and Precision-Recall AUC.
        
        Parameters
        ----------
        threshold : float
            Decision threshold for computing precision/recall.
        """
        probs = self.predict_proba(X)
        
        roc = roc_auc_score(y, probs)
        pr = average_precision_score(y, probs)
        
        # Precision and recall at the given threshold
        predictions = (probs >= threshold).astype(int)
        tp = int(((predictions == 1) & (y == 1)).sum())
        fp = int(((predictions == 1) & (y == 0)).sum())
        fn = int(((predictions == 0) & (y == 1)).sum())
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        
        return FraudModelMetrics(
            roc_auc=float(roc),
            pr_auc=float(pr),
            precision_at_threshold=precision,
            recall_at_threshold=recall,
            threshold=threshold,
            n_fraud=int(y.sum()),
            n_legitimate=int((y == 0).sum())
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Returns feature importances."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        if importances is None:
            return {name: 0.0 for name in self.feature_names_}
        return {name: float(imp) for name, imp in zip(self.feature_names_, importances)}
