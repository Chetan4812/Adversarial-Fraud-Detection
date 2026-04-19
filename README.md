# Adversarial Fraud Detection - Multi-Model Pipeline

This project implements a robust fraud detection pipeline using the IEEE-CIS Fraud Detection dataset. It combines extensive Exploratory Data Analysis (EDA) with a high-performance machine learning pipeline featuring CatBoost, XGBoost, and Random Forest.

The core philosophy is that fraud detection should not rely solely on transaction-level anomalies, but also incorporate **user behavior patterns** and **missingness signals**, as these often provide stronger predictive power than isolated transaction-level anomalies.

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

### 3. Docker Support
Run the entire pipeline in a containerized environment.
```bash
# Build
docker build -t fraud-detection .

# Run Training
docker run fraud-detection
```

---

## 📈 Pipeline Highlights

- **Feature Engineering**: Implements custom logic for time-based features, transaction amount log-scaling, missingness indicators (V and D columns), and device attribute parsing.
- **Model Portfolio**: Compares Dummy (baseline), Logistic Regression (scaled), Random Forest, XGBoost, and CatBoost.
- **CatBoost Integration**: Native handling of categorical variables for superior performance on sparse identity data.
- **Threshold Tuning**: Custom logic to optimize classification thresholds based on the Precision-Recall curve, prioritizing the detection of rare fraud cases.

---

## 📂 Project Structure
```text
├── artifacts/             # Persisted models and metadata
├── Dockerfile             # Containerization config
├── .dockerignore          # Docker exclusion rules
├── requirements.txt       # Project dependencies
├── train.py               # Model training script
├── score.py               # Inference script
├── features.py            # Feature engineering module
├── models.py              # Model definitions & pipelines
├── evaluate.py            # Metrics and PR-AUC tuning logic
├── predict.py             # Model persistence & scoring logic
└── README.md              # Documentation
```

## 📊 Key Insights
* Fraud is strongly linked to **behavioral inconsistencies**.
* Missing identity data is a **strong fraud indicator** rather than a problem to be solved.
* Threshold optimization is critical; detecting a true fraud case is often 10x more valuable than a false alarm.

---

## 📌 Future Roadmap
- [ ] Implement Target Encoding for high-cardinality features (`card1`, `addr1`).
- [ ] Add User-level aggregate features (rolling window counts).
- [ ] Deploy as a REST API using FastAPI.
- [ ] Integrate SHAP for better interpretability at the transaction level.
