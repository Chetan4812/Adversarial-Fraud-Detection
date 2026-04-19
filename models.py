"""
models.py
---------
Preprocessing pipelines and model definitions for fraud detection.

Models:
  - Dummy (baseline)
  - Logistic Regression
  - Random Forest
  - XGBoost
  - CatBoost (uses raw categoricals directly)
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ---------------------------------------------------------------------------
# Shared sklearn preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes + scales numeric columns
      - Imputes + ordinal-encodes categorical columns
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_dummy() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def build_logistic(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=None,
        )),
    ])


def build_random_forest(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_xgboost(preprocessor: ColumnTransformer, scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=3,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_catboost(class_weight_ratio: float) -> CatBoostClassifier:
    """
    CatBoost model that accepts raw categorical columns — no OrdinalEncoder needed.
    `class_weight_ratio` = count(negative) / count(positive).
    """
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=42,
        class_weights=[1.0, float(class_weight_ratio)],
        verbose=100,
        early_stopping_rounds=100,
    )
