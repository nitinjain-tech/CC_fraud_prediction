# CC_fraud_prediction

# 💳 Credit Card Fraud Detection System

## 📌 Overview

This project builds a **production-ready machine learning system** to detect fraudulent credit card transactions from a highly imbalanced dataset.
It demonstrates **end-to-end ML engineering**, including modeling, evaluation, API deployment, containerization, and cloud-based experiment tracking.

---

## 🎯 Problem Statement

Credit card fraud detection is a classic **imbalanced classification problem**, where fraudulent transactions are extremely rare (~0.17%).
The goal is to **maximize fraud detection (recall)** while maintaining **low false positives (high precision)**.

---

## 📊 Dataset

* ~284,000 transactions
* Highly imbalanced:

  * Non-fraud: ~99.83%
  * Fraud: ~0.17%
* Features:

  * `V1–V28`: PCA-transformed anonymized features
  * `Time`, `Amount`

---

## ⚙️ Approach

### 1️⃣ Exploratory Data Analysis (EDA)

* Identified severe class imbalance
* Analyzed feature distributions and correlations
* Observed strong separation in key PCA components (e.g., V11, V14, V17)

---

### 2️⃣ Handling Imbalanced Data

* Applied **SMOTE** on training data only
* Used **stratified train-test split** to avoid data leakage

---

### 3️⃣ Model Training

* Model: **Random Forest Classifier**
* Evaluated using:

  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * PR-AUC (important for imbalanced data)

---

### 4️⃣ Threshold Tuning

* Optimized classification threshold to balance:

  * Fraud detection (recall)
  * Customer experience (precision)

---

## 📈 Results

| Metric            | Value    |
| ----------------- | -------- |
| Precision (Fraud) | **97%**  |
| Recall (Fraud)    | **78%**  |
| ROC-AUC           | **0.96** |
| PR-AUC            | **0.83** |

---

## 🚀 Deployment

### 🔹 FastAPI

* Built REST API for real-time predictions
* Endpoint: `/predict`
* Returns:

  * Fraud probability
  * Prediction (0/1)
  * Risk level (Low / Medium / High)

---

### 🔹 Docker

* Containerized the application for portability
* Enables deployment on any platform

```bash
# Build
docker build -t fraud-api .

# Run
docker run -p 8000:8000 fraud-api
```

---

## ☁️ Azure Integration

Integrated with **Azure AI Foundry using MLflow:

* Experiment tracking
* Parameter logging
* Metric tracking
* Model artifact storage

---

## 🧠 ML System Design

This project simulates a real-world ML system:

* Training pipeline
* Model serialization (`.pkl`)
* API-based inference
* Containerized deployment
* Experiment tracking

---

## 📁 Project Structure

```
fraud-detection-ml/
│
├── app/
│   └── main.py              # FastAPI app
│
├── src/
│   ├── train_pipeline.py   # Training + MLflow
│   └── predict.py          # Inference logic
│
├── artifacts/
│   ├── model.pkl
│   └── threshold.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* FastAPI
* Docker
* MLflow
* Azure AI Foundry

---

## 🔑 Key Learnings

* Handling **imbalanced datasets** using SMOTE
* Avoiding **data leakage**
* Importance of **precision vs recall trade-off**
* Building **production-ready ML APIs**
* Managing **ML lifecycle with MLflow**
* Resolving **real-world deployment issues (Docker, dependencies)**

---

## 📌 Future Improvements

* Real-time streaming (Kafka)
* Feature store integration
* Automated retraining pipeline
* Model drift monitoring
* CI/CD for ML deployment

---

## 💼 Resume Highlights

* Built end-to-end ML system for fraud detection
* Achieved high precision (97%) with strong recall (78%)
* Deployed model using FastAPI and Docker
* Integrated MLflow with Azure for experiment tracking

---

## 📎 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python -m uvicorn app.main:app --reload
```

Open: http://localhost:8000/docs

---

