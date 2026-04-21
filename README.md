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
- [Docker Support](#docker-support)
- [Testing](#testing)
- [Future Work](#future-work)

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
                                                   Inference (score.py / app.py)
```

---

## 📂 Project Structure

```text
├── artifacts/             # Persisted models and metadata
├── tests/                 # Unit tests for core modules
├── Dockerfile             # Containerization config
├── .dockerignore          # Docker exclusion rules
├── requirements.txt       # Project dependencies
├── train.py               # Model training script
├── score.py               # Inference script
├── features.py            # Feature engineering module
├── models.py              # Model definitions & pipelines
├── evaluate.py            # Metrics and PR-AUC tuning logic
├── predict.py             # Model persistence & scoring logic
├── app.py                 # Streamlit dashboard
└── README.md              # Documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Docker](https://www.docker.com/) (optional)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chetan4812/Adversarial-Fraud-Detection.git
   cd Adversarial-Fraud-Detection
   ```

2. **Setup environment:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Requirements:**
   Place `train_transaction.csv` and `train_identity.csv` in the root directory.

---

## 🛠 Usage

### 1. Training the Model
Executes feature engineering, trains multiple models, compares performance via PR-AUC, and saves the best model.
```bash
python train.py
```

### 2. Inference / Scoring
Score new transaction data using the best available model in `artifacts/`.
```bash
python score.py --transaction data/test_transaction.csv --identity data/test_identity.csv
```

### 3. Dashboard (Streamlit)
Launch the interactive web frontend to upload CSVs and visualize results.
```bash
streamlit run app.py
```

### 4. Running Tests
The project includes unit tests for the feature engineering and prediction modules.
```bash
python3 -m pytest
```

### 5. Docker Support
The project is fully containerized. To build and run the dashboard:
```bash
# Build the image
docker build -t fraud-detection .

# Run the Streamlit dashboard
docker run -p 8501:8501 fraud-detection
```

---

## 📈 Pipeline Highlights

- **Adversarial Awareness**: Treating missing identity data as a **strong fraud signal** rather than a data quality issue.
- **Behavioral Signals**: Focusing on **user behavioral inconsistencies** rather than raw transaction anomalies.
- **Model Portfolio**: Compares Dummy (baseline), Logistic Regression (scaled), Random Forest, XGBoost, and CatBoost.
- **Threshold Tuning**: Custom logic to optimize classification thresholds based on the Precision-Recall curve, prioritizing the detection of rare fraud cases.

---

## 📊 Key Insights
* Fraud is strongly linked to **behavioral inconsistencies**.
* Threshold optimization is critical; detecting a true fraud case is often 10x more valuable than a false alarm.
* CatBoost's native handling of categorical variables provides superior performance on sparse identity data.

---

## 📌 Future Roadmap
- [ ] Implement Target Encoding for high-cardinality features (`card1`, `addr1`).
- [ ] Add User-level aggregate features (rolling window counts).
- [ ] Deploy as a REST API using FastAPI.
- [ ] Integrate SHAP for better interpretability at the transaction level.

---

## License

This project is licensed under the [MIT License](LICENSE).
---
Built for the [IEEE-CIS Fraud Detection Challenge](https://www.kaggle.com/c/ieee-fraud-detection)
