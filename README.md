# Adversarial Fraud Detection

A production-grade, multi-model fraud detection pipeline built on the IEEE-CIS Fraud Detection dataset. Combines behavioral feature engineering, adversarial-aware preprocessing, and threshold-optimized classification to catch fraud with high precision and recall.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-enabled-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-enabled-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Modeling Pipeline](#modeling-pipeline)
- [Feature Engineering](#feature-engineering)
- [Threshold Optimization](#threshold-optimization)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Dependencies](#dependencies)
- [Future Work](#future-work)

---

## Overview

Fraud detection is an adversarial problem — fraudsters actively adapt to evade detection systems. This project tackles the IEEE-CIS Fraud Detection challenge by combining:

- **Deep behavioral feature engineering** (time, device, email, card interaction signals)
- **A multi-model comparison framework** (Logistic Regression, Random Forest, XGBoost, CatBoost)
- **Precision-optimized threshold tuning** to maximize F1 under real-world class imbalance
- **Explainability** via CatBoost feature importance

The core insight: **user behavioral inconsistencies** are stronger fraud signals than raw transaction anomalies alone.

---

## Architecture

![High Level Design](HLD.png)

The pipeline follows a modular, production-ready design:

```
Raw CSVs -> Merge -> Feature Engineering -> Preprocessing -> Multi-Model Training
                                                                  |
                                                  Threshold Tuning -> Evaluation
                                                                  |
                                                         Artifact Persistence
                                                                  |
                                                       Inference (score.py)
```

---

## Dataset

The project uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) dataset (Kaggle), consisting of two tables:

### `train_transaction.csv`

| Column | Description |
|---|---|
| `TransactionID` | Unique transaction identifier |
| `TransactionAmt` | Transaction amount |
| `TransactionDT` | Time delta from a reference point (seconds) |
| `ProductCD` | Product category |
| `card1-card6` | Payment card attributes |
| `addr1, addr2` | Billing/shipping address features |
| `dist1, dist2` | Distance features |
| `P_emaildomain` | Purchaser email domain |
| `R_emaildomain` | Recipient email domain |
| `V1-V339` | Anonymized engineered features |
| `isFraud` | **Target** (0 = legitimate, 1 = fraud) |

### `train_identity.csv`

| Column | Description |
|---|---|
| `DeviceType` | Mobile or desktop |
| `DeviceInfo` | Device model/brand details |
| `id_30` | Operating system |
| `id_31` | Browser |
| `id_33` | Screen resolution |
| `id_12-id_38` | Anonymized identity features |

> Note: Not all transactions have identity records. **Missing identity data is treated as a fraud signal**, not a data quality issue.

---

## Project Structure

```
Adversarial-Fraud-Detection/
|
+-- Notebooks/
|   +-- Understanding Dataset and EDA for Fraud Detection.ipynb        # Baseline EDA
|   +-- Understanding Dataset and EDA for Fraud Detection - Updated.ipynb
|   +-- eda.ipynb                                                       # Extended EDA
|   +-- Precision-Optimized Adversarial Fraud Pipeline.ipynb           # Full pipeline
|
+-- Scripts/
|   +-- features.py          # Feature engineering & data loading
|   +-- models.py            # Model definitions & preprocessing pipelines
|   +-- train.py             # End-to-end training script
|   +-- evaluate.py          # Metrics, comparison, threshold tuning
|   +-- predict.py           # Save/load models, score new data
|   +-- score.py             # CLI scoring tool
|
+-- Dockerfile
+-- .dockerignore
+-- requirements.txt
+-- HLD.png                  # Architecture diagram
+-- README.md
```

---

## Modeling Pipeline

Five models are trained and compared on a **time-based 80/20 validation split** (respecting transaction chronology to prevent data leakage):

| Model | Key Characteristics |
|---|---|
| **Dummy Classifier** | Baseline — always predicts majority class |
| **Logistic Regression** | Linear baseline with balanced class weights |
| **Random Forest** | 300 trees, balanced subsampling |
| **XGBoost** | 500 estimators, `scale_pos_weight` for imbalance, `eval_metric=aucpr` |
| **CatBoost** | 1000 iterations, native categorical support, early stopping on PR-AUC |

Models are ranked by **PR-AUC** (Precision-Recall AUC), which is more appropriate than ROC-AUC for highly imbalanced datasets like fraud detection.

---

## Feature Engineering

Handled in `features.py`, the `add_features()` function creates interpretable fraud signals on top of raw columns:

### Time Features
- `Transaction_hour` — hour of day (cyclic fraud patterns)
- `Transaction_day` — day index from dataset start
- `TransactionDT_days` — float day representation

### Amount Features
- `TransactionAmt_log1p` — log-transformed amount (reduces skew)
- `TransactionAmt_mod_1` — cents portion (behavioral signal)
- `is_round_amount` — flag for suspiciously round amounts

### Missingness Features
- `row_missing_count` — total missing fields per row
- `v_missing_count` — missing V-columns (anonymized features)
- `d_missing_count` — missing D-columns (time delta features)

### Interaction Features
- `email_match` — whether purchaser and recipient email domains match
- `card4_card6`, `card1_card2`, `card1_card3` — card attribute combinations
- `card1_card2_ratio` — numeric card ratio

### Device and Browser Features
- `id_30_os` — extracted OS from `id_30`
- `id_31_browser` — extracted browser brand from `id_31`
- `DeviceInfo_brand` — extracted device brand
- `id_33_w`, `id_33_h`, `id_33_ratio` — screen resolution width/height/aspect ratio

---

## Threshold Optimization

Default 0.5 classification thresholds are suboptimal for fraud. The `tune_threshold()` function in `evaluate.py` scans the precision-recall curve to find the threshold that **maximises F1** on the validation set:

```python
def tune_threshold(y_valid, best_proba) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_valid, best_proba)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return {"threshold": float(thresholds[best_idx]), ...}
```

This threshold is saved alongside the model artifact and applied at inference time.

---

## Installation

### Prerequisites
- Python 3.10+
- IEEE-CIS dataset CSVs: `train_transaction.csv` and `train_identity.csv`

### Setup

```bash
git clone https://github.com/<your-username>/Adversarial-Fraud-Detection.git
cd Adversarial-Fraud-Detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train

Place `train_transaction.csv` and `train_identity.csv` in the project root, then run:

```bash
python train.py
```

This will:
- Load and merge transaction + identity data
- Engineer features
- Train all 5 models on an 80/20 time-based split
- Compare models by PR-AUC
- Tune the decision threshold on the best model
- Save the best model + metadata to `./artifacts/`

### 2. Score New Data

```bash
python score.py \
  --transaction test_transaction.csv \
  --identity test_identity.csv \
  --output predictions.csv
```

Output columns:

| Column | Description |
|---|---|
| `fraud_probability` | Raw model probability (0 to 1) |
| `fraud_prediction` | Binary flag (1 = predicted fraud) |

### 3. Notebooks

For exploratory analysis, open the notebooks in order:

```bash
jupyter notebook "Understanding Dataset and EDA for Fraud Detection.ipynb"
jupyter notebook "Precision-Optimized Adversarial Fraud Pipeline.ipynb"
```

---

## Docker

Build and run the full training pipeline in a container:

```bash
# Build
docker build -t fraud-detection .

# Train (mount your data directory)
docker run -v $(pwd)/data:/app/data fraud-detection python train.py

# Score
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts \
  fraud-detection python score.py \
  --transaction data/test_transaction.csv \
  --identity data/test_identity.csv \
  --output data/scores.csv
```

---

## Dependencies

```
numpy
pandas
scikit-learn
xgboost
catboost
matplotlib
seaborn
joblib
ipykernel
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## Future Work

| Area | Improvement |
|---|---|
| Feature Engineering | User-level aggregation features (velocity, historical fraud rate) |
| Temporal Modeling | Advanced rolling window and lag features |
| Graph Features | Graph-based fraud signals (card-device-email networks) |
| Deep Learning | Tabular deep learning (TabNet, FT-Transformer) |
| Real-Time Inference | Streaming pipeline with Kafka and model serving |
| Retraining Loop | Automated model refresh as fraud patterns shift |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

Built for the [IEEE-CIS Fraud Detection Challenge](https://www.kaggle.com/c/ieee-fraud-detection)
