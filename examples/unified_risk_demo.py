"""
Unified Risk Demo: The Customer 360 View.

This script demonstrates Sentinel's core value proposition:
Unifying Credit Risk (Phase 3) and Fraud Risk (Phase 4) into a 
single, coherent risk profile for retail banking customers.

Instead of siloed departments, we calculate:
Total Expected Loss = Expected Credit Loss + Expected Fraud Loss
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentinel.data.generator import generate_borrowers, generate_loans, generate_transactions, GeneratorConfig
from sentinel.credit.models import LogisticPDModel, XGBoostPDModel, LGDModel, EADModel, ExpectedLossEngine
from sentinel.fraud.features import engineer_fraud_features, get_fraud_feature_columns
from sentinel.fraud.detector import FraudDetector


def main():
    print("="*70)
    print("SENTINEL: UNIFIED RETAIL RISK (CUSTOMER 360)")
    print("="*70)
    
    # 1. Generate Universe
    print("\n1. Generating unified customer universe...")
    cfg = GeneratorConfig(n_borrowers=10_000, n_transactions=50_000, seed=42)
    borrowers = generate_borrowers(cfg)
    loans = generate_loans(borrowers, cfg)
    txns = generate_transactions(borrowers, cfg)
    
    print(f"   Created {len(borrowers):,} customers with {len(loans):,} loans and {len(txns):,} card transactions.")
    
    # 2. Train Credit Engine
    print("\n2. Training Credit Risk Engine...")
    X_credit = loans[["dti", "fico_score", "emp_length_years", "loan_amount", "interest_rate", "is_secured"]]
    
    pd_model = XGBoostPDModel(max_depth=3)
    pd_model.fit(X_credit, loans["default"])
    
    lgd_model = LGDModel(max_depth=3)
    lgd_model.fit(X_credit[loans["default"] == 1], loans[loans["default"] == 1]["lgd"])
    
    ead_model = EADModel(max_depth=3)
    ead_model.fit(X_credit[loans["default"] == 1], loans[loans["default"] == 1]["ead_factor"])
    
    credit_engine = ExpectedLossEngine(pd_model, lgd_model, ead_model)
    
    # 3. Train Fraud Engine
    print("3. Training Fraud Risk Engine...")
    featured_txns = engineer_fraud_features(txns)
    X_fraud = featured_txns[get_fraud_feature_columns()]
    y_fraud = featured_txns["is_fraud"]
    
    fraud_detector = FraudDetector(max_depth=3, n_estimators=50)
    fraud_detector.fit(X_fraud, y_fraud)
    
    # 4. Compute Expected Loss per Customer
    print("\n4. Computing Unified Expected Loss...")
    
    # A) Credit EL
    el_df = credit_engine.predict_expected_loss(X_credit, loans["loan_amount"])
    loans["credit_expected_loss"] = el_df["Expected_Loss"]
    
    # B) Fraud EL (Fraud Prob * Transaction Amount)
    featured_txns["fraud_prob"] = fraud_detector.predict_proba(X_fraud)
    featured_txns["expected_fraud_loss"] = featured_txns["fraud_prob"] * featured_txns["amount"]
    
    # Aggregate fraud risk per customer
    fraud_risk_by_customer = featured_txns.groupby("card_id")["expected_fraud_loss"].sum().reset_index()
    fraud_risk_by_customer.rename(columns={"card_id": "borrower_id"}, inplace=True)
    
    # C) Combine!
    unified_view = pd.merge(borrowers, loans[["borrower_id", "loan_amount", "credit_expected_loss"]], on="borrower_id", how="left")
    unified_view = pd.merge(unified_view, fraud_risk_by_customer, on="borrower_id", how="left")
    
    # Fill NAs for customers with no transactions
    unified_view["expected_fraud_loss"] = unified_view["expected_fraud_loss"].fillna(0.0)
    unified_view["total_expected_loss"] = unified_view["credit_expected_loss"] + unified_view["expected_fraud_loss"]
    
    print("\n" + "="*70)
    print("TOP 5 HIGHEST RISK CUSTOMERS (TOTAL EXPECTED LOSS)")
    print("="*70)
    
    top_risk = unified_view.sort_values("total_expected_loss", ascending=False).head(5)
    
    for idx, row in top_risk.iterrows():
        print(f"Customer {int(row['borrower_id']):05d} | FICO: {row['fico_score']:.0f} | Income: ${row['annual_income']:,.0f}")
        print(f"  Loan Amount:          ${row['loan_amount']:9,.2f}")
        print(f"  Credit Expected Loss: ${row['credit_expected_loss']:9,.2f}  <- (Driven by Credit Defaults)")
        print(f"  Fraud Expected Loss:  ${row['expected_fraud_loss']:9,.2f}  <- (Driven by Card Activity)")
        print(f"  TOTAL RISK EXPOSURE:  ${row['total_expected_loss']:9,.2f}")
        print("-" * 50)
        
    print(f"\nPortfolio Total Credit EL: ${unified_view['credit_expected_loss'].sum():,.2f}")
    print(f"Portfolio Total Fraud EL:  ${unified_view['expected_fraud_loss'].sum():,.2f}")
    print(f"PORTFOLIO TOTAL RISK:      ${unified_view['total_expected_loss'].sum():,.2f}")
    print("="*70)


if __name__ == "__main__":
    main()
