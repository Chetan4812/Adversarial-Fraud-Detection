# Fraud Detection - EDA and Multi-Model Pipeline

## Overview

This project focuses on detecting fraudulent transactions using the IEEE dataset by combining **exploratory data analysis (EDA)** with a **multi-model machine learning pipeline**.

The core idea behind this work is that fraud detection should not rely solely on transaction-level anomalies, but also incorporate **user behavior patterns**, which often provide stronger predictive signals.

---

## Dataset Description

The dataset consists of two main components:

### 1. Transaction Data (`train_transaction`)

Contains transaction-level information

* `TransactionID`: Unique identifier
* `TransactionAmt`: Transaction amount
* `TransactionDT`: Time delta from a reference point
* `ProductCD`: Product category
* `card1–card6`: Payment card attributes
* `addr1, addr2`: Address-related features
* `dist1, dist2`: Distance features
* `isFraud`: Target variable (0 = non-fraud, 1 = fraud)
* `V1–V339`: Engineered anonymized features

---

### 2. Identity Data (`train_identity`)

Contains user/device-level attributes:

* `DeviceType`: Mobile or desktop
* `DeviceInfo`: Device details
* `id_31`: Browser information
* `id_33`: Screen resolution
* `id_12–id_38`: Identity-related anonymized features

---

## Key Observations

* Not all transactions include identity information
* Missing identity data can act as a **strong fraud indicator**
* The dataset is **highly imbalanced**, with very few fraud cases
* Many features contain **significant missing values**
* Behavioral inconsistencies often correlate with fraudulent activity

---

## Data Processing

### Merging Datasets

Transaction and identity datasets are merged to enrich feature space:

```python
df = df_trans.merge(df_id, on="TransactionID", how="left")
```

### Important Considerations

* Missing values are treated as **informative signals**, not simply removed
* Identity and transaction features are combined for better context
* Care is taken to preserve data integrity during merging

---

## Exploratory Data Analysis (EDA)

### Key Insights

* Fraud is strongly linked to **behavioral inconsistencies**
* `TransactionDT` represents **relative time**, not actual timestamps
* Certain devices, browsers, and missing identity patterns show higher fraud rates
* Feature interactions reveal stronger signals than individual variables

### Core Insight

Understanding **user behavior patterns** is more effective than focusing solely on transaction anomalies.

---

## Preprocessing Pipeline

A custom preprocessing function is implemented to standardize data preparation:

```python
def preprocess(data, is_train=True, label_encoders=None):
    ...
```

### Responsibilities

* Handle missing values (retain signal where useful)
* Encode categorical variables
* Align training and testing datasets
* Prepare features for model compatibility

---

## Modeling Approach

Multiple machine learning models are trained and compared:

* Random Forest
* Gradient Boosting-based models
* Additional aligned models for experimentation

```python
models = {
    ...
}
```

### Why Multiple Models?

* Improves robustness
* Captures different patterns in data
* Reduces reliance on a single model’s bias

---

## Threshold Optimization

A custom threshold tuning function is used:

```python
def find_optimal_threshold(y_true, y_prob, min_recall=0.30):
    ...
```

### Rationale

* Fraud detection prioritizes **recall over precision**
* Missing fraud cases is more costly than false positives
* Threshold tuning allows better control over model behavior

---

## Training and Evaluation Pipeline

### Workflow

1. Preprocess data
2. Train multiple models
3. Generate prediction probabilities
4. Optimize classification threshold
5. Evaluate performance

```python
results = {}
```

### Evaluation Focus

* Recall (primary metric)
* Precision
* F1-score
* Model stability

---

## Key Takeaways

* Fraud detection improves significantly with **feature understanding**
* Behavioral patterns provide stronger signals than isolated transactions
* Missing data can be **predictive rather than problematic**
* Combining multiple models increases reliability and performance

---

## Future Improvements

* User-level aggregation features
* Advanced time-based feature engineering
* Deep learning models
* Graph-based fraud detection approaches
* Real-time fraud detection systems

---

## Project Structure

```
├── EDA Notebook
│   └── Understanding Dataset and EDA for Fraud Detection.ipynb
│
├── Modeling Notebook
│   └── IEEE_Fraud_Multi_Model_Aligned.ipynb
│
└── README.md
```

---

## Conclusion

This project demonstrates that effective fraud detection depends on a combination of:

* Strong data understanding
* Thoughtful feature engineering
* Model selection and optimization

A **behavior-focused approach** provides significantly better predictive performance compared to traditional transaction-only methods.

