"""
Probability of Default (PD) Models for Sentinel.

Implements the statistical and machine learning models used to predict
the likelihood that a borrower will default on a loan.

Models:
1. Logistic Regression: Highly interpretable, assumes linear relationships
   in the log-odds space. Required by many regulators for fair lending.
2. XGBoost: Gradient boosted trees. Highly accurate, captures non-linear
   relationships (e.g., FICO scores having diminishing returns), but harder
   to explain ("black box").
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

logger = logging.getLogger(__name__)


class PDModel(ABC):
    """Abstract base class for Probability of Default models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default.
        Returns a 1D array of probabilities (0.0 to 1.0).
        """
        pass
        
    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance or coefficients."""
        pass


class LogisticPDModel(PDModel):
    """
    Logistic Regression PD Model.
    Includes a StandardScaler because Logistic Regression is sensitive to
    feature scaling (e.g., Income in $100k vs FICO in 700s).
    """
    def __init__(self, C: float = 1.0):
        # We use a pipeline to ensure data is always scaled before hitting the model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=C, solver='liblinear', random_state=42))
        ])
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names_ = list(X.columns)
        self.pipeline.fit(X, y)
        logger.info("Logistic Regression PD Model trained successfully.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # sklearn predict_proba returns [P(class=0), P(class=1)]
        # We only want P(class=1) which is Default.
        return self.pipeline.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        """Returns the model coefficients as feature importance."""
        model = self.pipeline.named_steps['model']
        coefs = model.coef_[0]
        return {name: float(coef) for name, coef in zip(self.feature_names_, coefs)}


class XGBoostPDModel(PDModel):
    """
    XGBoost PD Model.
    Uses gradient boosted decision trees. Excellent for tabular financial data.
    """
    def __init__(self, max_depth: int = 4, learning_rate: float = 0.1, n_estimators: int = 100):
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        logger.info("XGBoost PD Model trained successfully.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        """Returns the XGBoost feature importances (gain)."""
        importances = self.model.feature_importances_
        if importances is None:
            return {name: 0.0 for name in self.feature_names_}
        return {name: float(imp) for name, imp in zip(self.feature_names_, importances)}

# ---------------------------------------------------------
# LGD and EAD Regression Models
# ---------------------------------------------------------

class LGDModel:
    """
    Loss Given Default (LGD) Predictor.
    Uses XGBoost Regressor bounded between 0 and 1 via logistic objective.
    """
    def __init__(self, max_depth: int = 3, n_estimators: int = 50):
        # We use reg:logistic because LGD is mathematically bounded [0, 1]
        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            objective='reg:logistic',
            random_state=42
        )
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """y must be the actual observed LGD [0, 1] for defaulted loans."""
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        logger.info("XGBoost LGD Model trained successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class EADModel:
    """
    Exposure at Default (EAD) Factor Predictor.
    Predicts the % of original loan amount outstanding at default.
    """
    def __init__(self, max_depth: int = 3, n_estimators: int = 50):
        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            objective='reg:logistic',
            random_state=42
        )
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """y must be the actual observed EAD Factor [0, 1] for defaulted loans."""
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        logger.info("XGBoost EAD Model trained successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


# ---------------------------------------------------------
# Expected Loss Integration
# ---------------------------------------------------------

class ExpectedLossEngine:
    """
    Combines PD, LGD, and EAD models to compute Expected Loss (EL) 
    in dollars for a portfolio of loans.
    
    Supports different feature sets per model via optional feature name lists.
    If not specified, all models receive the same input features.
    """
    def __init__(
        self, 
        pd_model: PDModel, 
        lgd_model: LGDModel, 
        ead_model: EADModel,
        pd_features: list[str] | None = None,
        lgd_features: list[str] | None = None,
        ead_features: list[str] | None = None,
    ):
        self.pd_model = pd_model
        self.lgd_model = lgd_model
        self.ead_model = ead_model
        self.pd_features = pd_features
        self.lgd_features = lgd_features
        self.ead_features = ead_features
        
    def predict_expected_loss(self, X: pd.DataFrame, original_loan_amounts: pd.Series) -> pd.DataFrame:
        """
        Calculates EL = PD * LGD * (EAD_Factor * Original_Amount)
        """
        X_pd = X[self.pd_features] if self.pd_features else X
        X_lgd = X[self.lgd_features] if self.lgd_features else X
        X_ead = X[self.ead_features] if self.ead_features else X
        
        # Predict the 3 components
        pd_preds = self.pd_model.predict_proba(X_pd)
        lgd_preds = self.lgd_model.predict(X_lgd)
        ead_factors = self.ead_model.predict(X_ead)
        
        # Calculate EL
        predicted_ead_dollars = ead_factors * original_loan_amounts
        expected_loss_dollars = pd_preds * lgd_preds * predicted_ead_dollars
        
        return pd.DataFrame({
            "PD": pd_preds,
            "LGD": lgd_preds,
            "EAD_Factor": ead_factors,
            "EAD_Dollars": predicted_ead_dollars,
            "Expected_Loss": expected_loss_dollars
        }, index=X.index)
