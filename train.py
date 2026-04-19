"""
train.py
--------
End-to-end fraud detection training script.

Usage
-----
    python train.py

Expects train_transaction.csv and train_identity.csv in the working directory.
Outputs trained model + metadata to ./artifacts/.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from features import load_and_merge
from models import (
    build_catboost,
    build_dummy,
    build_logistic,
    build_preprocessor,
    build_random_forest,
    build_xgboost,
)
from evaluate import (
    compare_models,
    plot_catboost_importance,
    print_threshold_report,
    tune_threshold,
)
from predict import save_best_model


# ---------------------------------------------------------------------------
# 1. Load data + feature engineering
# ---------------------------------------------------------------------------
merged = load_and_merge("train_transaction.csv", "train_identity.csv")

TARGET = "isFraud"
df = merged.drop(columns=["TransactionID"])
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# ---------------------------------------------------------------------------
# 2. Time-based train / validation split (80 / 20)
# ---------------------------------------------------------------------------
order = merged["TransactionDT"].rank(method="first").astype(int)
cutoff = np.quantile(order, 0.8)
train_idx = order <= cutoff
valid_idx = order > cutoff

X_train, X_valid = X.loc[train_idx].copy(), X.loc[valid_idx].copy()
y_train, y_valid = y.loc[train_idx].copy(), y.loc[valid_idx].copy()

print("Train size:", X_train.shape, "| Fraud rate:", round(y_train.mean(), 4))
print("Valid size:", X_valid.shape, "| Fraud rate:", round(y_valid.mean(), 4))

# ---------------------------------------------------------------------------
# 3. Column type detection
# ---------------------------------------------------------------------------
categorical_cols = X_train.select_dtypes(
    include=["object", "string", "category", "bool"]
).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

print(f"\nNumeric columns:     {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# ---------------------------------------------------------------------------
# 4. Build + fit sklearn models
# ---------------------------------------------------------------------------
preprocessor = build_preprocessor(numeric_cols, categorical_cols)
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
class_weight_ratio = scale_pos_weight  # same value, different variable name for clarity

print("\n--- Fitting Dummy ---")
dummy = build_dummy()
dummy.fit(X_train.fillna(0), y_train)
dummy_pred = dummy.predict_proba(X_valid.fillna(0))[:, 1]

print("--- Fitting Logistic Regression ---")
logit = build_logistic(preprocessor)
logit.fit(X_train, y_train)
logit_pred = logit.predict_proba(X_valid)[:, 1]

print("--- Fitting Random Forest ---")
rf = build_random_forest(preprocessor)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_valid)[:, 1]

print("--- Fitting XGBoost ---")
xgb = build_xgboost(preprocessor, scale_pos_weight=scale_pos_weight)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict_proba(X_valid)[:, 1]

# ---------------------------------------------------------------------------
# 5. Prepare CatBoost data (fill missing values manually)
# ---------------------------------------------------------------------------
print("--- Fitting CatBoost ---")
cat_train = X_train.copy()
cat_valid = X_valid.copy()

for c in categorical_cols:
    cat_train[c] = cat_train[c].astype("string").fillna("Unknown")
    cat_valid[c] = cat_valid[c].astype("string").fillna("Unknown")

cat_num_medians = cat_train[numeric_cols].median(numeric_only=True)
cat_train[numeric_cols] = cat_train[numeric_cols].fillna(cat_num_medians)
cat_valid[numeric_cols] = cat_valid[numeric_cols].fillna(cat_num_medians)

cat_model = build_catboost(class_weight_ratio=class_weight_ratio)
cat_model.fit(
    cat_train, y_train,
    cat_features=categorical_cols,
    eval_set=(cat_valid, y_valid),
    use_best_model=True,
)
cat_pred = cat_model.predict_proba(cat_valid)[:, 1]

# ---------------------------------------------------------------------------
# 6. Compare models
# ---------------------------------------------------------------------------
all_preds = {
    "Dummy": dummy_pred,
    "Logistic Regression": logit_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
    "CatBoost": cat_pred,
}

results = compare_models(y_valid, all_preds)
print("\n--- Model Comparison ---")
print(results.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Threshold tuning on best model
# ---------------------------------------------------------------------------
best_model_name = results.iloc[0]["Model"]
best_proba = all_preds[best_model_name]
print(f"\nBest model by PR-AUC: {best_model_name}")

threshold_info = tune_threshold(y_valid, best_proba)
print_threshold_report(y_valid, best_proba, threshold_info)

# ---------------------------------------------------------------------------
# 8. Feature importance (CatBoost only)
# ---------------------------------------------------------------------------
if best_model_name == "CatBoost":
    plot_catboost_importance(cat_model, feature_names=cat_train.columns.tolist())
else:
    print("Feature importance chart is only shown when CatBoost is the best model.")

# ---------------------------------------------------------------------------
# 9. Save best model
# ---------------------------------------------------------------------------
fitted_models = {
    "Dummy": dummy,
    "Logistic Regression": logit,
    "Random Forest": rf,
    "XGBoost": xgb,
    "CatBoost": cat_model,
}

save_best_model(
    best_model_name=best_model_name,
    models=fitted_models,
    threshold=threshold_info["threshold"],
    categorical_cols=categorical_cols,
    numeric_cols=numeric_cols,
    cat_num_medians=cat_num_medians if best_model_name == "CatBoost" else None,
)
