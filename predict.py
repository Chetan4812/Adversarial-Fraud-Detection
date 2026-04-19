"""
predict.py
----------
Save/load trained models and score new transaction data.
"""

from pathlib import Path

import joblib
import pandas as pd

from features import add_features


ARTIFACT_DIR = Path("artifacts")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_best_model(
    best_model_name: str,
    models: dict,
    threshold: float,
    categorical_cols: list[str],
    numeric_cols: list[str],
    cat_num_medians=None,
    artifact_dir: Path = ARTIFACT_DIR,
) -> None:
    """
    Persist the best model and its metadata to *artifact_dir*.

    Parameters
    ----------
    models : dict
        Keys are model names (e.g. 'XGBoost'), values are fitted model objects.
    threshold : float
        Decision threshold determined by threshold tuning.
    """
    artifact_dir.mkdir(exist_ok=True)

    name_to_file = {
        "CatBoost": "catboost_fraud_model.pkl",
        "XGBoost": "xgb_fraud_model.pkl",
        "Logistic Regression": "logit_fraud_model.pkl",
        "Random Forest": "rf_fraud_model.pkl",
    }
    model_file = name_to_file.get(best_model_name, "best_fraud_model.pkl")
    model_stem = model_file.replace("_fraud_model.pkl", "")

    joblib.dump(models[best_model_name], artifact_dir / model_file)

    metadata = {
        "threshold": threshold,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }
    if cat_num_medians is not None and best_model_name == "CatBoost":
        metadata["cat_num_medians"] = cat_num_medians

    joblib.dump(metadata, artifact_dir / f"{model_stem}_metadata.pkl")
    print(f"Saved {best_model_name} and metadata to: {artifact_dir.resolve()}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def score_new_data(
    new_transaction_path: str,
    new_identity_path: str,
    best_model_name: str,
    models: dict,
    threshold: float,
    categorical_cols: list[str],
    numeric_cols: list[str],
    cat_num_medians=None,
) -> pd.DataFrame:
    """
    Merge new transaction + identity files, engineer features, and score.

    Returns a DataFrame with columns ['fraud_probability', 'fraud_prediction'].
    """
    new_tr = pd.read_csv(new_transaction_path)
    new_id = pd.read_csv(new_identity_path)
    new_df = new_tr.merge(new_id, on="TransactionID", how="left")
    new_df = add_features(new_df)
    new_df = new_df.drop(columns=["TransactionID"], errors="ignore")

    if best_model_name == "CatBoost":
        temp = new_df.copy()
        for c in categorical_cols:
            if c in temp.columns:
                temp[c] = temp[c].astype("string").fillna("Unknown")
        if cat_num_medians is not None:
            temp[numeric_cols] = temp[numeric_cols].fillna(cat_num_medians)
        proba = models["CatBoost"].predict_proba(temp)[:, 1]
    else:
        model_obj = models[best_model_name]
        proba = model_obj.predict_proba(new_df)[:, 1]

    return pd.DataFrame({
        "fraud_probability": proba,
        "fraud_prediction": (proba >= threshold).astype(int),
    })
