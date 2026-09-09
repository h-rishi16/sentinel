"""
Evaluates the full Expected Loss Pipeline (PD, LGD, EAD) on synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentinel.data.generator import generate_borrowers, generate_loans, GeneratorConfig
from sentinel.credit.models import XGBoostPDModel, LGDModel, EADModel, ExpectedLossEngine

def main():
    print("="*60)
    print("SENTINEL: FULL EXPECTED LOSS (EL) PIPELINE")
    print("="*60)
    
    # 1. Generate Data
    print("Generating synthetic loan book (50,000 loans)...")
    config = GeneratorConfig(n_borrowers=20_000, n_loans=50_000, seed=102)
    borrowers = generate_borrowers(config)
    loans = generate_loans(borrowers, config)
    
    # Features for the ML model
    features = ["fico_score", "annual_income", "emp_length_years", "loan_amount", "term_months", "interest_rate", "dti", "is_secured"]
    
    # 2. Train-Test Split for PD
    X = loans[features]
    y_pd = loans["default"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_pd, test_size=0.2, random_state=42)
    
    # Train PD Model
    print("Training PD Model (XGBoost)...")
    pd_model = XGBoostPDModel(max_depth=3, n_estimators=50)
    pd_model.fit(X_train, y_train)
    
    # 3. Train LGD & EAD (Crucially, we only train these on loans that ACTUALLY defaulted)
    print("Training LGD and EAD Models (on defaulted loans only)...")
    defaulted_loans = loans[loans["default"] == 1]
    X_def = defaulted_loans[features]
    y_lgd = defaulted_loans["lgd"]
    y_ead = defaulted_loans["ead_factor"]
    
    lgd_model = LGDModel(max_depth=3, n_estimators=50)
    lgd_model.fit(X_def, y_lgd)
    
    ead_model = EADModel(max_depth=3, n_estimators=50)
    ead_model.fit(X_def, y_ead)
    
    # 4. Integrate into Expected Loss Engine
    engine = ExpectedLossEngine(pd_model, lgd_model, ead_model)
    
    # Predict on a portfolio of 5 random active loans from the test set
    test_portfolio = X_test.sample(5, random_state=99)
    results = engine.predict_expected_loss(test_portfolio, test_portfolio["loan_amount"])
    
    print("\n--- EXPECTED LOSS PREDICTIONS (5 Sample Loans) ---")
    
    for idx, row in test_portfolio.iterrows():
        print(f"\nLoan #{idx}: {row['loan_amount']:,.0f} | FICO: {row['fico_score']:.0f} | Secured: {'Yes' if row['is_secured'] else 'No'}")
        print(f"  -> Probability of Default (PD): {results.loc[idx, 'PD']:.2%}")
        print(f"  -> Loss Given Default (LGD):    {results.loc[idx, 'LGD']:.2%}")
        print(f"  -> Exposure at Default (EAD):   ${results.loc[idx, 'EAD_Dollars']:,.0f} ({results.loc[idx, 'EAD_Factor']:.1%} of orig)")
        print(f"  == EXPECTED LOSS (EL):          ${results.loc[idx, 'Expected_Loss']:,.0f}")

    print("\n" + "="*60)
    
    # Portfolio level aggregation
    full_portfolio_results = engine.predict_expected_loss(X_test, X_test["loan_amount"])
    total_exposure = X_test["loan_amount"].sum()
    total_expected_loss = full_portfolio_results["Expected_Loss"].sum()
    
    print(f"TEST PORTFOLIO TOTAL EXPOSURE:      ${total_exposure:,.0f}")
    print(f"TEST PORTFOLIO TOTAL EXPECTED LOSS: ${total_expected_loss:,.0f}")
    print(f"PORTFOLIO RISK DENSITY:             {total_expected_loss/total_exposure:.2%} (Loss / Exposure)")
    print("="*60)


if __name__ == "__main__":
    main()
