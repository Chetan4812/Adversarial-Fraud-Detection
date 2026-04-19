"""
evaluate.py
-----------
Model evaluation, comparison, and decision-threshold tuning for fraud detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_proba) -> dict:
    """Return ROC-AUC and PR-AUC for a set of predictions."""
    return {
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
    }


def compare_models(y_valid, predictions: dict) -> pd.DataFrame:
    """
    Build a comparison table sorted by PR-AUC.

    Parameters
    ----------
    y_valid : array-like
        Ground-truth labels.
    predictions : dict
        Mapping of model name -> predicted probabilities array.

    Returns
    -------
    pd.DataFrame with columns ['Model', 'ROC-AUC', 'PR-AUC'].
    """
    rows = []
    for name, proba in predictions.items():
        m = compute_metrics(y_valid, proba)
        rows.append({"Model": name, **m})
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(y_valid, best_proba) -> dict:
    """
    Find the decision threshold that maximises F1 on the validation set.

    Returns a dict with keys: threshold, precision, recall, f1.
    """
    precision, recall, thresholds = precision_recall_curve(y_valid, best_proba)
    f1_scores = (
        2 * (precision[:-1] * recall[:-1])
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_idx = int(np.argmax(f1_scores))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1_scores[best_idx]),
    }


def print_threshold_report(y_valid, best_proba, threshold_info: dict) -> None:
    """Print precision/recall/F1 and confusion matrix at the tuned threshold."""
    threshold = threshold_info["threshold"]
    pred_labels = (best_proba >= threshold).astype(int)

    print(f"Best threshold : {threshold:.4f}")
    print(f"Precision      : {threshold_info['precision']:.4f}")
    print(f"Recall         : {threshold_info['recall']:.4f}")
    print(f"F1             : {threshold_info['f1']:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_valid, pred_labels, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_valid, pred_labels))


# ---------------------------------------------------------------------------
# Feature importance (CatBoost)
# ---------------------------------------------------------------------------

def plot_catboost_importance(cat_model, feature_names, top_n: int = 20) -> None:
    """Plot a horizontal bar chart of the top-N CatBoost feature importances."""
    importances = cat_model.get_feature_importance()
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    print(fi.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.title(f"Top {top_n} CatBoost Feature Importances")
    plt.tight_layout()
    plt.show()
