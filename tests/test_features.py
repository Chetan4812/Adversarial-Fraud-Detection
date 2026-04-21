"""
tests/test_features.py
----------------------
Unit tests for the feature engineering module.
"""
import numpy as np
import pandas as pd
import pytest
from features import add_features, load_and_merge, safe_string, first_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_minimal_df(**overrides):
    """Return a minimal DataFrame that satisfies add_features() requirements."""
    base = {
        "TransactionID": [1, 2, 3],
        "TransactionDT": [86400, 172800, 259200],   # 1d, 2d, 3d
        "TransactionAmt": [100.0, 0.99, 1000.0],
        "P_emaildomain": ["gmail.com", "yahoo.com", "gmail.com"],
        "R_emaildomain": ["gmail.com", "gmail.com", "gmail.com"],
        "card1": ["1001", "2002", "3003"],
        "card2": ["200", "400", "600"],
        "card3": ["150", "150", "150"],
        "card4": ["visa", "mastercard", "visa"],
        "card6": ["debit", "credit", "debit"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# safe_string
# ---------------------------------------------------------------------------

class TestSafeString:
    def test_fills_nan(self):
        s = pd.Series([None, "hello", float("nan")])
        result = safe_string(s)
        assert result[0] == "Unknown"
        assert result[1] == "hello"
        assert result[2] == "Unknown"

    def test_normal_values_unchanged(self):
        s = pd.Series(["visa", "mastercard"])
        assert list(safe_string(s)) == ["visa", "mastercard"]


# ---------------------------------------------------------------------------
# first_token
# ---------------------------------------------------------------------------

class TestFirstToken:
    def test_splits_on_space(self):
        s = pd.Series(["Windows 10", "Mac OS X"])
        result = first_token(s)
        assert result[0] == "Windows"
        assert result[1] == "Mac"

    def test_no_separator(self):
        s = pd.Series(["Chrome", "Firefox"])
        result = first_token(s)
        assert list(result) == ["Chrome", "Firefox"]

    def test_nan_becomes_unknown(self):
        s = pd.Series([None, "Safari"])
        result = first_token(s)
        assert result[0] == "Unknown"


# ---------------------------------------------------------------------------
# add_features
# ---------------------------------------------------------------------------

class TestAddFeatures:
    def setup_method(self):
        self.df = make_minimal_df()
        self.out = add_features(self.df)

    def test_original_columns_preserved(self):
        for col in self.df.columns:
            assert col in self.out.columns

    def test_time_features_created(self):
        assert "Transaction_hour" in self.out.columns
        assert "Transaction_day" in self.out.columns
        assert "TransactionDT_days" in self.out.columns

    def test_amount_features_created(self):
        assert "TransactionAmt_log1p" in self.out.columns
        assert "TransactionAmt_mod_1" in self.out.columns
        assert "is_round_amount" in self.out.columns

    def test_missingness_features_created(self):
        assert "row_missing_count" in self.out.columns
        assert "v_missing_count" in self.out.columns
        assert "d_missing_count" in self.out.columns

    def test_categorical_interactions(self):
        assert "email_match" in self.out.columns
        assert "card4_card6" in self.out.columns
        assert "card1_card2" in self.out.columns

    def test_email_match_correct(self):
        # Row 0: both gmail → match (1); Row 1: yahoo vs gmail → no match (0)
        assert self.out["email_match"].iloc[0] == 1
        assert self.out["email_match"].iloc[1] == 0

    def test_log_amount_nonnegative(self):
        assert (self.out["TransactionAmt_log1p"] >= 0).all()

    def test_is_round_amount(self):
        # 100.0 is round, 0.99 is not
        assert self.out["is_round_amount"].iloc[0] == 1
        assert self.out["is_round_amount"].iloc[1] == 0

    def test_transaction_hour_range(self):
        assert self.out["Transaction_hour"].between(0, 23).all()

    def test_does_not_mutate_input(self):
        original_cols = set(self.df.columns)
        add_features(self.df)
        assert set(self.df.columns) == original_cols

    def test_with_id_columns(self):
        df = make_minimal_df()
        df["id_30"] = ["Windows 10", "Mac OS X", None]
        df["id_31"] = ["Chrome 88", "Firefox 80", "Safari"]
        df["DeviceInfo"] = ["Samsung SM-G950F", "Apple iPhone", None]
        df["id_33"] = ["1920x1080", "1280x720", None]
        out = add_features(df)
        assert "id_30_os" in out.columns
        assert "id_31_browser" in out.columns
        assert "DeviceInfo_brand" in out.columns
        assert "id_33_ratio" in out.columns

    def test_returns_dataframe(self):
        assert isinstance(self.out, pd.DataFrame)

    def test_row_count_unchanged(self):
        assert len(self.out) == len(self.df)


# ---------------------------------------------------------------------------
# load_and_merge (uses tmp files)
# ---------------------------------------------------------------------------

class TestLoadAndMerge:
    def test_merge_and_features(self, tmp_path):
        trans = make_minimal_df()
        trans["isFraud"] = [0, 1, 0]
        identity = pd.DataFrame({
            "TransactionID": [1, 2],
            "id_30": ["Windows 10", "Mac OS"],
        })
        trans_path = tmp_path / "trans.csv"
        id_path = tmp_path / "id.csv"
        trans.to_csv(trans_path, index=False)
        identity.to_csv(id_path, index=False)

        merged = load_and_merge(str(trans_path), str(id_path))

        # All transaction rows must be present (left join)
        assert len(merged) == 3
        # Feature columns must be present
        assert "Transaction_hour" in merged.columns
        assert "email_match" in merged.columns
        # TransactionID must be preserved for joining
        assert "TransactionID" in merged.columns

    def test_missing_identity_rows_are_nan(self, tmp_path):
        """Row 3 has no identity match — numeric identity cols should be NaN."""
        trans = make_minimal_df()
        trans["isFraud"] = [0, 1, 0]
        identity = pd.DataFrame({
            "TransactionID": [1],
            "id_02": [1000.0],
        })
        trans_path = tmp_path / "trans.csv"
        id_path = tmp_path / "id.csv"
        trans.to_csv(trans_path, index=False)
        identity.to_csv(id_path, index=False)

        merged = load_and_merge(str(trans_path), str(id_path))
        # Rows 2 and 3 should have NaN for id_02
        assert merged.loc[merged["TransactionID"] == 2, "id_02"].isna().all()
