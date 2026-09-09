"""
Fraud Detection end-to-end demo.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from sentinel.data.generator import generate_transactions, GeneratorConfig
from sentinel.fraud.features import engineer_fraud_features, get_fraud_feature_columns
from sentinel.fraud.detector import FraudDetector

def main():
    print("="*60)
    print("SENTINEL: FRAUD DETECTION ENGINE")
    print("="*60)
    
    # 1. Generate synthetic transactions
    print("Generating 100,000 synthetic transactions tied to 50,000 borrowers...")
    cfg = GeneratorConfig(n_borrowers=50_000, n_transactions=100_000, seed=42)
    from sentinel.data.generator import generate_borrowers
    borrowers = generate_borrowers(cfg)
    txns = generate_transactions(borrowers, cfg)
    
    fraud_count = txns["is_fraud"].sum()
    print(f"Total Transactions: {len(txns):,}")
    print(f"Fraudulent: {fraud_count} ({fraud_count/len(txns):.2%})")
    
    # 2. Engineer features
    print("\nEngineering features (velocity, behavioral deviation, temporal)...")
    featured = engineer_fraud_features(txns)
    
    feature_cols = get_fraud_feature_columns()
    X = featured[feature_cols]
    y = featured["is_fraud"]
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train
    print("Training XGBoost Fraud Detector...")
    model = FraudDetector(max_depth=4, n_estimators=100)
    model.fit(X_train, y_train)
    
    # 5. Evaluate on test set
    metrics = model.evaluate(X_test, y_test, threshold=0.5)
    
    print("\n--- TEST SET PERFORMANCE ---")
    print(f"ROC-AUC:   {metrics.roc_auc:.4f}")
    print(f"PR-AUC:    {metrics.pr_auc:.4f}")
    print(f"Precision: {metrics.precision_at_threshold:.2%} (at threshold {metrics.threshold})")
    print(f"Recall:    {metrics.recall_at_threshold:.2%} (at threshold {metrics.threshold})")
    print(f"Test Fraud: {metrics.n_fraud} | Test Legit: {metrics.n_legitimate:,}")
    
    # 6. Feature Importance
    print("\n--- TOP FEATURES ---")
    imp = model.get_feature_importance()
    for name, score in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name:25s} {score:.4f}")
    
    # 7. Show some example predictions
    print("\n--- SAMPLE PREDICTIONS (5 Fraud, 5 Legit) ---")
    fraud_test = X_test[y_test == 1].head(5)
    legit_test = X_test[y_test == 0].head(5)
    
    print("FRAUD transactions:")
    for idx in fraud_test.index:
        prob = model.predict_proba(X_test.loc[[idx]])[0]
        amt = X_test.loc[idx, "amount"]
        zscore = X_test.loc[idx, "amount_zscore"]
        night = "Night" if X_test.loc[idx, "is_night"] else "Day"
        print(f"  ${amt:8.2f} | Z-score: {zscore:5.1f} | {night} | Fraud Prob: {prob:.2%}")
    
    print("LEGIT transactions:")
    for idx in legit_test.index:
        prob = model.predict_proba(X_test.loc[[idx]])[0]
        amt = X_test.loc[idx, "amount"]
        zscore = X_test.loc[idx, "amount_zscore"]
        night = "Night" if X_test.loc[idx, "is_night"] else "Day"
        print(f"  ${amt:8.2f} | Z-score: {zscore:5.1f} | {night} | Fraud Prob: {prob:.2%}")
    
    print("="*60)

if __name__ == "__main__":
    main()
