import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

def create_dummy_model():
    """Create a mock model and metadata for testing the Streamlit app."""
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)
    
    # 1. Create dummy data to fit a pipeline
    X = pd.DataFrame({
        "TransactionAmt": [100.0, 20.0, 50.0, 300.0],
        "Transaction_hour": [10.0, 2.0, 15.0, 23.0],
        "card4_card6": ["visa_debit", "mastercard_credit", "visa_debit", "discover_credit"]
    })
    y = np.array([0, 1, 0, 1])
    
    numeric_cols = ["TransactionAmt", "Transaction_hour"]
    categorical_cols = ["card4_card6"]
    
    # 2. Build a simple preprocessor
    num_pipe = Pipeline([("imp", SimpleImputer()), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder())])
    
    prep = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])
    
    # 3. Build and fit pipeline
    model_pipeline = Pipeline([
        ("prep", prep),
        ("model", LogisticRegression())
    ])
    model_pipeline.fit(X, y)
    
    # 4. Save model
    model_path = artifact_dir / "logit_fraud_model.pkl"
    joblib.dump(model_pipeline, model_path)
    
    # 5. Save metadata
    metadata = {
        "threshold": 0.5,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }
    meta_path = artifact_dir / "logit_metadata.pkl"
    joblib.dump(metadata, meta_path)
    
    print(f"✅ Mock model created at: {model_path}")
    print(f"✅ Mock metadata created at: {meta_path}")

if __name__ == "__main__":
    create_dummy_model()
