"""
Evaluates PD models on synthetic loan data.
"""

from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from sentinel.credit.models import LogisticPDModel, XGBoostPDModel
from sentinel.data.generator import GeneratorConfig, generate_borrowers, generate_loans


def main():
    print("="*60)
    print("SENTINEL: CREDIT RISK (PROBABILITY OF DEFAULT)")
    print("="*60)

    # 1. Generate Data
    print("Generating synthetic loan book (50,000 loans)...")
    config = GeneratorConfig(n_borrowers=20_000, n_loans=50_000, seed=101)
    borrowers = generate_borrowers(config)
    loans = generate_loans(borrowers, config)

    # Features for the ML model
    features = ["fico_score", "annual_income", "emp_length_years", "loan_amount", "interest_rate", "dti"]
    X = loans[features]
    y = loans["default"]

    print(f"Data generated. Default rate: {y.mean():.2%}")

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Models
    print("\nTraining Logistic Regression...")
    log_model = LogisticPDModel(C=0.1)
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict_proba(X_test)

    print("Training XGBoost...")
    xgb_model = XGBoostPDModel(max_depth=3, n_estimators=50)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_test)

    # 4. Evaluation
    print("\n--- MODEL PERFORMANCE ---")

    # ROC AUC (Ranking Power)
    log_auc = roc_auc_score(y_test, log_preds)
    xgb_auc = roc_auc_score(y_test, xgb_preds)

    # Brier Score (Calibration)
    # Lower is better (0.0 is perfect)
    log_brier = brier_score_loss(y_test, log_preds)
    xgb_brier = brier_score_loss(y_test, xgb_preds)

    print("1. Logistic Regression:")
    print(f"   ROC AUC:     {log_auc:.4f}")
    print(f"   Brier Score: {log_brier:.4f}")
    print("   Top Features:")
    for f, imp in sorted(log_model.get_feature_importance().items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
        print(f"     - {f}: {imp:+.4f}")

    print("\n2. XGBoost:")
    print(f"   ROC AUC:     {xgb_auc:.4f}")
    print(f"   Brier Score: {xgb_brier:.4f}")
    print("   Top Features:")
    for f, imp in sorted(xgb_model.get_feature_importance().items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"     - {f}: {imp:.4f}")

    print("="*60)

if __name__ == "__main__":
    main()
