# Enterprise AI Fraud Detection & Decision Intelligence Platform

An industry-grade end-to-end Machine Learning system designed to detect fraudulent financial transactions and support human decision-making through explainable AI and interactive dashboards.

---

## 🚀 Project Overview

This project simulates how real-world fintech companies build, deploy, and validate fraud detection systems.  
It covers the complete Machine Learning lifecycle — from raw data ingestion to model deployment and decision support.

The system is intentionally designed to be **production-oriented**, focusing on robustness, explainability, and realistic fraud behavior rather than just model accuracy.

---

## 🧠 Key Features

- Fraud transaction data ingestion and validation  
- Feature engineering pipeline with strict raw/processed data separation  
- Baseline and advanced ML models (Logistic Regression, XGBoost)  
- Proper evaluation for highly imbalanced datasets (ROC-AUC, Precision, Recall)  
- Explainable AI using SHAP for model interpretability  
- Real-time fraud prediction service (API-ready design)  
- Interactive decision dashboard with human-in-the-loop feedback  
- Prediction history logging for auditability and monitoring  

---

## 🏗️ System Architecture

Raw Transaction Data
↓
Data Validation & Ingestion
↓
Feature Engineering
↓
Model Training & Evaluation
↓
Explainability (SHAP)
↓
Prediction Logic
↓
Decision Dashboard + Feedback Loop


---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Explainability:** SHAP  
- **Model Serving:** FastAPI (local)  
- **Dashboard:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 📊 Dataset

- **Source:** Credit Card Transactions Dataset (European cardholders)  
- **Nature:** Highly imbalanced fraud detection dataset  
- **Note:** Raw dataset is not included in the repository due to licensing constraints.  
  Feature-engineered data and pipeline code are provided.

---

## 🧪 Model Behavior & Validation

- Legitimate transactions typically receive **near-zero fraud probability**  
- Fraud transactions produce **high-confidence risk scores**  
- The system was validated using:
  - Legitimate samples  
  - Real historical fraud cases  
  - Synthetic fraud-like stress tests  

This behavior reflects **real-world fraud detection systems**, where most transactions are safe and only a small fraction are high risk.

---

## 📊 Decision Dashboard

The Streamlit-based dashboard allows:
- Manual input of transaction features  
- Real-time fraud probability estimation  
- Viewing prediction history  
- Simulated human feedback for future retraining  

This demonstrates a **human-in-the-loop ML workflow**, commonly used in finance and risk systems.

---

