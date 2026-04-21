"""
tests/test_predict.py
---------------------
Unit tests for the predict / save / score pipeline.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from predict import save_best_model, score_new_data
from features import add_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mini_transaction_csv(tmp_path, name="trans.csv"):
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [86400, 172800, 259200],
        "TransactionAmt": [50.0, 200.0, 10.0],
        "P_emaildomain": ["gmail.com", "yahoo.com", "gmail.com"],
        "R_emaildomain": ["gmail.com", "gmail.com", "gmail.com"],
        "card1": ["1001", "2002", "3003"],
        "card2": ["200", "400", "600"],
        "card3": ["150", "150", "150"],
        "card4": ["visa", "mastercard", "visa"],
        "card6": ["debit", "credit", "debit"],
    })
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def make_mini_identity_csv(tmp_path, name="id.csv"):
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "id_30": ["Windows 10", "Mac OS"],
    })
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# save_best_model
# ---------------------------------------------------------------------------

class TestSaveBestModel:
    def test_saves_logistic_artifacts(self, tmp_path):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        model = Pipeline([
            ("imp", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression()),
        ])
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]
        model.fit(X, y)

        save_best_model(
            best_model_name="Logistic Regression",
            models={"Logistic Regression": model},
            threshold=0.4,
            categorical_cols=[],
            numeric_cols=["a", "b"],
            artifact_dir=tmp_path,
        )

        assert (tmp_path / "logit_fraud_model.pkl").exists()
        assert (tmp_path / "logit_metadata.pkl").exists()

    def test_metadata_contents(self, tmp_path):
        import joblib
        from sklearn.dummy import DummyClassifier

        dummy = DummyClassifier()
        dummy.fit([[0]], [0])

        save_best_model(
            best_model_name="Dummy",
            models={"Dummy": dummy},
            threshold=0.3,
            categorical_cols=["card4"],
            numeric_cols=["TransactionAmt"],
            artifact_dir=tmp_path,
        )

        meta_path = tmp_path / "best_metadata.pkl"  # fallback name for Dummy
        # Check either the fallback or a named file was saved
        meta_files = list(tmp_path.glob("*_metadata.pkl"))
        assert len(meta_files) == 1
        meta = joblib.load(meta_files[0])
        assert meta["threshold"] == 0.3
        assert "categorical_cols" in meta
        assert "numeric_cols" in meta


# ---------------------------------------------------------------------------
# score_new_data
# ---------------------------------------------------------------------------

class TestScoreNewData:
    def test_output_columns(self, tmp_path):
        """score_new_data must return a DataFrame with the two required columns."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        numeric_cols = ["TransactionAmt", "Transaction_hour", "TransactionAmt_log1p",
                        "TransactionAmt_mod_1", "TransactionAmt_mod_10", "is_round_amount",
                        "row_missing_count", "v_missing_count", "d_missing_count",
                        "Transaction_day", "TransactionDT_days", "card1_card2_ratio"]
        categorical_cols = ["card4_card6", "card1_card2", "card1_card3"]

        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])

        from sklearn.compose import make_column_selector
        prep = ColumnTransformer([
            ("num", num_pipe, numeric_cols),
        ], remainder="drop")

        model = Pipeline([("prep", prep), ("lr", LogisticRegression())])

        # Fit on dummy numeric data
        X_dummy = pd.DataFrame(np.random.rand(10, len(numeric_cols)), columns=numeric_cols)
        y_dummy = np.array([0, 1] * 5)
        model.fit(X_dummy, y_dummy)

        trans_path = make_mini_transaction_csv(tmp_path)
        id_path = make_mini_identity_csv(tmp_path)

        results = score_new_data(
            new_transaction_path=str(trans_path),
            new_identity_path=str(id_path),
            best_model_name="Logistic Regression",
            models={"Logistic Regression": model},
            threshold=0.5,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
        )

        assert "fraud_probability" in results.columns
        assert "fraud_prediction" in results.columns
        assert len(results) == 3  # matches transaction rows

    def test_probabilities_in_range(self, tmp_path):
        from sklearn.dummy import DummyClassifier
        trans_path = make_mini_transaction_csv(tmp_path)
        id_path = make_mini_identity_csv(tmp_path)

        # Use a pass-through model by scoring manually
        # Just test that the function runs and probabilities are [0,1]
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        numeric_cols = ["TransactionAmt"]
        model = Pipeline([
            ("imp", SimpleImputer()),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression()),
        ])
        model.fit([[50.0], [200.0], [10.0]], [0, 1, 0])

        # patch predict_proba to work with any DataFrame
        class WrapModel:
            def predict_proba(self, X):
                return np.column_stack([
                    np.full(len(X), 0.3),
                    np.full(len(X), 0.7),
                ])

        results = score_new_data(
            new_transaction_path=str(trans_path),
            new_identity_path=str(id_path),
            best_model_name="Logistic Regression",
            models={"Logistic Regression": WrapModel()},
            threshold=0.5,
            categorical_cols=[],
            numeric_cols=numeric_cols,
        )

        assert results["fraud_probability"].between(0, 1).all()

    def test_missing_transaction_id_raises(self, tmp_path):
        df = pd.DataFrame({"Amount": [100]})  # no TransactionID
        p = tmp_path / "bad_trans.csv"
        df.to_csv(p, index=False)
        id_path = make_mini_identity_csv(tmp_path)

        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.8, 0.2]] * len(X))

        with pytest.raises(ValueError, match="TransactionID"):
            score_new_data(
                new_transaction_path=str(p),
                new_identity_path=str(id_path),
                best_model_name="Logistic Regression",
                models={"Logistic Regression": DummyModel()},
                threshold=0.5,
                categorical_cols=[],
                numeric_cols=[],
            )
