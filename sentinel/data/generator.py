"""
Data Generator for Sentinel.

Generates synthetic, interpretable, tabular data for Credit Risk and Fraud Detection.
This replaces the anonymized Kaggle PCA data and LendingClub data used in the old repos.

By generating our own data, we can:
1. Plant specific features that our models must learn to find (e.g., velocity for fraud).
2. Create realistic class imbalances.
3. Add macroeconomic factors that affect both credit default and market risk.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    n_borrowers: int = 10_000
    n_loans: int = 12_000
    n_transactions: int = 100_000
    seed: int = 42


def generate_borrowers(config: GeneratorConfig) -> pd.DataFrame:
    """Generate realistic borrower profiles."""
    rng = np.random.default_rng(config.seed)
    
    # FICO scores: normally distributed around 700, capped between 300-850
    fico = rng.normal(700, 50, config.n_borrowers)
    fico = np.clip(fico, 300, 850).astype(int)
    
    # Income: log-normally distributed, median around 65k
    income = rng.lognormal(mean=11.0, sigma=0.5, size=config.n_borrowers)
    income = np.clip(income, 15_000, 500_000).astype(int)
    
    # Employment length (years): exponential decay
    emp_length = rng.exponential(scale=5, size=config.n_borrowers)
    emp_length = np.clip(emp_length, 0, 40).astype(int)
    
    return pd.DataFrame({
        "borrower_id": np.arange(config.n_borrowers),
        "fico_score": fico,
        "annual_income": income,
        "emp_length_years": emp_length
    })


def generate_loans(borrowers: pd.DataFrame, config: GeneratorConfig) -> pd.DataFrame:
    """Generate loans linked to borrowers with realistic default logic."""
    rng = np.random.default_rng(config.seed + 1)
    
    # Assign borrowers to loans
    borrower_ids = rng.choice(borrowers["borrower_id"], size=config.n_loans, replace=True)
    
    # Merge borrower info to compute loan logic
    df = pd.DataFrame({"loan_id": np.arange(config.n_loans), "borrower_id": borrower_ids})
    df = df.merge(borrowers, on="borrower_id")
    
    # Loan amount: related to income (DTI constraints)
    base_loan = rng.lognormal(mean=9.5, sigma=0.6, size=config.n_loans)
    df["loan_amount"] = np.clip(base_loan, 1_000, 40_000).astype(int)
    
    df["term_months"] = rng.choice([36, 60], size=config.n_loans, p=[0.7, 0.3])
    
    # Interest rate depends on FICO
    base_rate = 0.15 - (df["fico_score"] - 600) * 0.0003
    noise = rng.normal(0, 0.02, config.n_loans)
    df["interest_rate"] = np.clip(base_rate + noise, 0.05, 0.36)
    
    # Debt-to-Income (simulated monthly payment vs monthly income)
    monthly_income = df["annual_income"] / 12
    monthly_payment = (df["loan_amount"] * (df["interest_rate"] / 12)) / (1 - (1 + df["interest_rate"] / 12) ** -df["term_months"])
    df["dti"] = monthly_payment / monthly_income
    
    # -- Ground Truth Default Logic --
    # Probability of default increases with low FICO, high DTI, low employment
    pd_logit = (
        -4.0 
        + 3.0 * df["dti"] 
        - 0.01 * (df["fico_score"] - 650) 
        - 0.05 * df["emp_length_years"]
        + rng.normal(0, 1.0, config.n_loans) # Unobservable noise
    )
    
    # Sigmoid
    pd_actual = 1 / (1 + np.exp(-pd_logit))
    
    # Added for LGD/EAD Modeling:
    # 1. Secured status (e.g., auto loan vs personal loan)
    df["is_secured"] = rng.choice([0, 1], size=config.n_loans, p=[0.7, 0.3])
    
    # 2. Ground-truth Recovery Rate and LGD (Loss Given Default)
    # Secured loans recover ~70% on average, unsecured ~20%. FICO adds a slight boost.
    base_recovery = 0.20 + (0.50 * df["is_secured"]) + (0.05 * (df["fico_score"] - 650) / 100)
    noise_rec = rng.normal(0, 0.10, config.n_loans)
    df["recovery_rate"] = np.clip(base_recovery + noise_rec, 0.0, 1.0)
    df["lgd"] = 1.0 - df["recovery_rate"]
    
    # 3. Ground-truth EAD Factor (Exposure at Default as a % of original loan)
    # Shorter term loans amortize faster, meaning lower EAD factor at default
    base_ead = 0.85 - (0.15 * (df["term_months"] == 36).astype(int))
    noise_ead = rng.normal(0, 0.05, config.n_loans)
    df["ead_factor"] = np.clip(base_ead + noise_ead, 0.0, 1.0)
    df["ead"] = df["loan_amount"] * df["ead_factor"]
    
    # Actual default flag
    df["default"] = (rng.uniform(0, 1, config.n_loans) < pd_actual).astype(int)
    
    # Clean up and order columns
    cols = [
        "loan_id", "borrower_id", "fico_score", "annual_income", "emp_length_years", 
        "loan_amount", "term_months", "interest_rate", "dti", "is_secured", 
        "recovery_rate", "lgd", "ead_factor", "ead", "default"
    ]
    return df[cols]


def generate_transactions(borrowers: pd.DataFrame, config: GeneratorConfig) -> pd.DataFrame:
    """
    Generate synthetic credit card transactions linked to actual borrowers,
    with planted fraud patterns.
    """
    rng = np.random.default_rng(config.seed + 10)
    n = config.n_transactions
    
    # Each borrower gets exactly one credit card (card_id == borrower_id)
    n_cards = len(borrowers)
    valid_card_ids = borrowers["borrower_id"].values
    
    # Assign each card a "personality" based on their income
    # Wealthier borrowers have higher typical transaction amounts
    incomes = borrowers["annual_income"].values
    base_spend = (incomes / 12) * 0.01  # Roughly 1% of monthly income per transaction
    
    card_mean_amount = rng.lognormal(mean=np.log(base_spend), sigma=0.5)
    card_mean_amount = np.clip(card_mean_amount, 5.0, 1000.0)
    
    # Most people shop 9AM-8PM
    card_typical_hour = rng.integers(9, 20, size=n_cards)                  
    
    # --- Generate legitimate transactions ---
    # Randomly pick which card is making the transaction
    card_ids = rng.choice(valid_card_ids, size=n)
    
    # Transaction amounts: drawn from each cardholder's personal distribution
    amounts = np.array([
        rng.lognormal(mean=np.log(card_mean_amount[cid]), sigma=0.4) 
        for cid in card_ids
    ])
    amounts = np.clip(amounts, 0.50, 10_000.0)
    
    # Transaction hour: normally distributed around the cardholder's typical hour
    hours = np.array([
        int(rng.normal(card_typical_hour[cid], 2)) % 24
        for cid in card_ids
    ])
    
    # Day index (0-365): transactions spread over a year
    days = rng.integers(0, 365, size=n)
    
    # Merchant category codes (simplified)
    categories = rng.choice(
        ["grocery", "restaurant", "gas", "online", "travel", "entertainment", "other"],
        size=n,
        p=[0.25, 0.15, 0.10, 0.20, 0.05, 0.10, 0.15]
    )
    
    # --- Plant fraud ---
    # Fraud rate: ~0.2% (realistic for card fraud)
    n_fraud = max(int(n * 0.002), 10)
    fraud_indices = rng.choice(n, size=n_fraud, replace=False)
    
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[fraud_indices] = 1
    
    # Fraud Pattern 1: Amount spike (3-10x the cardholder's normal spend)
    for idx in fraud_indices:
        cid = card_ids[idx]
        amounts[idx] = card_mean_amount[cid] * rng.uniform(3.0, 10.0)
    
    # Fraud Pattern 2: Unusual hours (shift fraud transactions to midnight-5AM)
    hours[fraud_indices] = rng.integers(0, 5, size=n_fraud)
    
    # Fraud Pattern 3: Burst velocity (cluster fraud on the same day for the same card)
    # Pick a subset of fraud cards and force their fraud transactions onto the same day per card
    burst_fraud_indices = fraud_indices[:n_fraud // 2]
    # Get the unique cards involved in burst fraud
    burst_cards = np.unique(card_ids[burst_fraud_indices])
    
    # For each card, pick one random day and move all its burst fraud transactions to that day
    for cid in burst_cards:
        mask = (card_ids == cid) & np.isin(np.arange(n), burst_fraud_indices)
        if mask.sum() > 0:
            target_day = rng.integers(0, 365)
            days[mask] = target_day
    
    df = pd.DataFrame({
        "txn_id": np.arange(n),
        "card_id": card_ids,
        "amount": np.round(amounts, 2),
        "hour": hours,
        "day": days,
        "category": categories,
        "is_fraud": is_fraud
    })
    
    # Sort by card_id and day for realistic time-series ordering
    df = df.sort_values(["card_id", "day", "hour"]).reset_index(drop=True)
    
    return df

