"""
score.py
--------
Score new transaction data using a previously saved model.

Usage
-----
    python score.py --transaction test_transaction.csv --identity test_identity.csv

The script loads the model + metadata from ./artifacts/ automatically.
"""

import argparse
import joblib
from pathlib import Path

from predict import score_new_data

ARTIFACT_DIR = Path("artifacts")

MODEL_FILES = {
    "catboost": ("catboost_fraud_model.pkl", "catboost_metadata.pkl", "CatBoost"),
    "xgb":      ("xgb_fraud_model.pkl",      "xgb_metadata.pkl",      "XGBoost"),
    "logit":    ("logit_fraud_model.pkl",     "logit_metadata.pkl",     "Logistic Regression"),
    "rf":       ("rf_fraud_model.pkl",        "rf_metadata.pkl",        "Random Forest"),
}


def find_saved_model(artifact_dir: Path):
    """Return (model_object, metadata, model_name) for the first artefact found."""
    for stem, (model_file, meta_file, name) in MODEL_FILES.items():
        model_path = artifact_dir / model_file
        meta_path  = artifact_dir / meta_file
        if model_path.exists() and meta_path.exists():
            print(f"Loading {name} from {model_path}")
            model    = joblib.load(model_path)
            metadata = joblib.load(meta_path)
            return model, metadata, name
    raise FileNotFoundError(
        f"No saved model found in {artifact_dir}. Run train.py first."
    )


def main():
    parser = argparse.ArgumentParser(description="Score new transaction data for fraud.")
    parser.add_argument("--transaction", required=True, help="Path to new transaction CSV")
    parser.add_argument("--identity",    required=True, help="Path to new identity CSV")
    parser.add_argument("--output",      default="scores.csv", help="Output CSV path")
    args = parser.parse_args()

    model, metadata, model_name = find_saved_model(ARTIFACT_DIR)

    scores = score_new_data(
        new_transaction_path=args.transaction,
        new_identity_path=args.identity,
        best_model_name=model_name,
        models={model_name: model},
        threshold=metadata["threshold"],
        categorical_cols=metadata["categorical_cols"],
        numeric_cols=metadata["numeric_cols"],
        cat_num_medians=metadata.get("cat_num_medians"),
    )

    scores.to_csv(args.output, index=False)
    print(f"Scores written to {args.output}")
    print(scores.head())


if __name__ == "__main__":
    main()
