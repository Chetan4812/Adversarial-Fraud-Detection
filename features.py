"""
features.py
-----------
Feature engineering for fraud detection.
Merges transaction + identity tables and creates interpretable fraud signals.
"""

import numpy as np
import pandas as pd


def safe_string(series: pd.Series) -> pd.Series:
    """Convert a series to string, filling NaN/NA with 'Unknown'."""
    return series.astype("string").fillna("Unknown").replace({"<NA>": "Unknown"})


def first_token(series: pd.Series, sep_pattern: str = r"[ /\\;:_\-\(\)]") -> pd.Series:
    """Extract the first token from a string column (e.g. OS from 'Windows 10')."""
    s = safe_string(series)
    return s.str.split(sep_pattern, n=1, regex=True).str[0].fillna("Unknown")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a merged transaction/identity DataFrame.

    Features created:
      - Time-based: hour-of-day, day index
      - Amount-based: log-transform, cents, round-amount flag
      - Missingness counts: overall, V-columns, D-columns
      - Categorical interactions: email match, card combinations
      - Device/browser text: OS, browser brand, screen ratio
    """
    df = df.copy()

    # --- Time features ---
    df["TransactionDT_days"] = df["TransactionDT"] / 86_400.0
    df["Transaction_hour"] = ((df["TransactionDT"] // 3600) % 24).astype("float32")
    df["Transaction_day"] = (df["TransactionDT"] // 86_400).astype("float32")

    # --- Amount features ---
    df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce")
    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_mod_1"] = np.mod(df["TransactionAmt"], 1.0)
    df["TransactionAmt_mod_10"] = np.mod(df["TransactionAmt"], 10.0)
    df["is_round_amount"] = (df["TransactionAmt_mod_1"].fillna(0) == 0).astype("int8")

    # --- Missingness features ---
    df["row_missing_count"] = df.isna().sum(axis=1).astype("float32")
    v_cols = [c for c in df.columns if c.startswith("V")]
    d_cols = [c for c in df.columns if c.startswith("D")]
    df["v_missing_count"] = df[v_cols].isna().sum(axis=1).astype("float32") if v_cols else 0
    df["d_missing_count"] = df[d_cols].isna().sum(axis=1).astype("float32") if d_cols else 0

    # --- Categorical interaction features ---
    df["email_match"] = (
        safe_string(df["P_emaildomain"]) == safe_string(df["R_emaildomain"])
    ).astype("int8")
    df["card4_card6"] = safe_string(df["card4"]) + "_" + safe_string(df["card6"])
    df["card1_card2"] = safe_string(df["card1"]) + "_" + safe_string(df["card2"])
    df["card1_card3"] = safe_string(df["card1"]) + "_" + safe_string(df["card3"])

    # --- Device / browser text cleanup ---
    if "id_30" in df.columns:
        df["id_30_os"] = first_token(df["id_30"])
    if "id_31" in df.columns:
        df["id_31_browser"] = first_token(df["id_31"])
    if "DeviceInfo" in df.columns:
        df["DeviceInfo_brand"] = first_token(df["DeviceInfo"])

    if "id_33" in df.columns:
        wh = safe_string(df["id_33"]).str.lower().str.split("x", n=1, expand=True)
        df["id_33_w"] = pd.to_numeric(wh[0], errors="coerce")
        df["id_33_h"] = pd.to_numeric(wh[1], errors="coerce")
        df["id_33_ratio"] = df["id_33_w"] / (df["id_33_h"] + 1e-3)

    if {"card1", "card2"}.issubset(df.columns):
        df["card1_card2_ratio"] = pd.to_numeric(df["card1"], errors="coerce") / (
            pd.to_numeric(df["card2"], errors="coerce") + 1e-3
        )

    return df


def load_and_merge(
    transaction_path: str,
    identity_path: str,
) -> pd.DataFrame:
    """
    Load transaction and identity CSVs, merge on TransactionID, and add features.

    Returns the merged + feature-engineered DataFrame.
    """
    transaction = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    print("Transaction shape:", transaction.shape)
    print("Identity shape:   ", identity.shape)
    print("Target balance:\n", transaction["isFraud"].value_counts(normalize=True).rename("share"))

    merged = transaction.merge(identity, on="TransactionID", how="left")
    merged = add_features(merged)

    print("Merged shape:", merged.shape)
    print("Fraud rate:  ", merged["isFraud"].mean())
    return merged
