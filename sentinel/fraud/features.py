"""
Fraud Feature Engineering for Sentinel.

Transforms raw transaction data into ML-ready features by computing:
1. Velocity features: Transaction counts and spending over rolling windows.
2. Behavioral deviation: Z-score of amount vs. cardholder's historical mean/std.
3. Temporal features: Hour-of-day encoding, is_night flag.

These are the features that real fraud detection systems use.
The Kaggle PCA features (V1-V28) hide this complexity behind anonymization.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_fraud_features(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer fraud detection features from raw transaction data.
    
    Parameters
    ----------
    txns : pd.DataFrame
        Raw transactions with columns: card_id, amount, hour, day, category.
        Must be sorted by card_id and day.
    
    Returns
    -------
    pd.DataFrame
        Original data with engineered features appended.
    """
    df = txns.copy()
    
    # -------------------------------------------------------
    # 1. Velocity Features (per-card rolling window counts)
    # -------------------------------------------------------
    # Group by card_id and compute strictly chronological rolling statistics
    # "How many transactions has this card done so far today?"
    # Using cumcount/cumsum prevents target leakage (lookahead bias)
    
    # Transaction count so far today
    df["txn_count_1d"] = df.groupby(["card_id", "day"]).cumcount() + 1
    
    # Total spending so far today
    df["txn_sum_1d"] = df.groupby(["card_id", "day"])["amount"].cumsum()
    
    # For multi-day velocity, we use an expanding window approach
    # First, compute per-card cumulative transaction count and amount
    df["card_cumcount"] = df.groupby("card_id").cumcount() + 1
    df["card_cumsum"] = df.groupby("card_id")["amount"].cumsum()
    
    # Average transaction amount for this card up to this point
    # We use a shift to strictly prevent the current transaction from leaking into the baseline
    shifted_cumsum = df.groupby("card_id")["amount"].transform(lambda x: x.shift(1).fillna(0))
    shifted_cumcount = df.groupby("card_id").cumcount()
    df["card_avg_amount"] = np.where(shifted_cumcount == 0, df["amount"], shifted_cumsum / np.maximum(shifted_cumcount, 1))
    
    # Standard deviation of amount per card (expanding window, shifted to prevent leakage)
    df["card_std_amount"] = df.groupby("card_id")["amount"].transform(
        lambda x: x.shift(1).expanding().std().fillna(1.0)
    )
    
    # -------------------------------------------------------
    # 2. Behavioral Deviation (Z-score)
    # -------------------------------------------------------
    # How far is this transaction from the cardholder's typical behavior?
    # Z = (amount - mean) / std
    # A Z-score of 5 means this transaction is 5 standard deviations 
    # above the cardholder's average — highly suspicious.
    df["amount_zscore"] = (df["amount"] - df["card_avg_amount"]) / df["card_std_amount"].clip(lower=1.0)
    
    # -------------------------------------------------------
    # 3. Temporal Features
    # -------------------------------------------------------
    # Is this transaction happening at night (midnight to 5 AM)?
    df["is_night"] = (df["hour"] < 5).astype(int)
    
    # Hour encoded as cyclical features (so 23:00 and 01:00 are close)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # -------------------------------------------------------
    # 4. Amount Features
    # -------------------------------------------------------
    # Log-transform of amount (reduces skewness for the model)
    df["log_amount"] = np.log1p(df["amount"])
    
    # Amount relative to card's historical average
    df["amount_to_avg_ratio"] = df["amount"] / df["card_avg_amount"].clip(lower=1.0)
    
    # Clean up intermediate columns
    df = df.drop(columns=["card_cumcount", "card_cumsum"])
    
    return df


def get_fraud_feature_columns() -> list[str]:
    """Returns the list of engineered feature column names for the fraud model."""
    return [
        "amount",
        "log_amount",
        "hour",
        "hour_sin",
        "hour_cos",
        "is_night",
        "txn_count_1d",
        "txn_sum_1d",
        "card_avg_amount",
        "card_std_amount",
        "amount_zscore",
        "amount_to_avg_ratio",
    ]
