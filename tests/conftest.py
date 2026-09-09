"""
Shared test fixtures for Sentinel.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_credit_data():
    """Generates a tiny dataset where default is perfectly predicted by DTI."""
    np.random.seed(42)
    n = 100

    # Features
    fico = np.random.normal(700, 50, n)
    dti = np.random.uniform(0.1, 0.6, n)

    # Target: High DTI defaults
    default = (dti > 0.4).astype(int)

    # Introduce a tiny bit of noise so Logistic Regression doesn't throw perfect separation warnings
    noise_idx = np.random.choice(n, 5, replace=False)
    default[noise_idx] = 1 - default[noise_idx]

    X = pd.DataFrame({"fico": fico, "dti": dti})
    y = pd.Series(default, name="default")

    return X, y
