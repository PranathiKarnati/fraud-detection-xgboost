"""Shared feature engineering used by both training and inference."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

PCA_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA_COLS + ["Amount_scaled", "Hour"]


def prepare(df: pd.DataFrame, scaler: RobustScaler | None = None):
    """Create model features.

    - Hour: hour of day derived from `Time` (seconds since first transaction)
    - Amount_scaled: log1p(Amount) scaled with RobustScaler (robust to outliers)
    The scaler is fit ONLY on training data (pass scaler=None) and reused after.
    """
    out = pd.DataFrame(index=df.index)
    for c in PCA_COLS:
        out[c] = df[c]
    out["Hour"] = (df["Time"] // 3600) % 24
    amt = np.log1p(df[["Amount"]])
    if scaler is None:
        scaler = RobustScaler().fit(amt)
    out["Amount_scaled"] = scaler.transform(amt).ravel()
    return out[FEATURES], scaler
