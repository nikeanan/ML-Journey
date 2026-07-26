# ML-Journey: Enterprise Machine Learning Engineering Studio & Monetizable API 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ML-Journey** is a full-stack, enterprise-grade Machine Learning Engineering repository and SaaS microservice platform. Built from numerical first-principles (vectorized NumPy algorithms) up to production REST API endpoints and interactive Streamlit Pro Dark UI/UX dashboards.

---

## 🌟 Key Features & Architecture

```text
ML-Journey Enterprise Stack
├── 🧠 Custom Algorithms (NumPy Neural Networks, Vectorized Cost Surfaces)
├── 📈 Financial Cost-Sensitive Learning (Campaign ROI Threshold Tuning)
├── 💳 Fintech Risk Scoring (FICO-style Credit Risk Scores 300-850)
├── 🔄 Churn & LTV Optimization (Customer Lifetime Value & Retention Offers)
├── 🔑 Monetizable REST API Microservice (FastAPI + API Key Auth)
└── 🚀 Interactive Pro Studio Web App (Streamlit Glassmorphism UI)
```

---

## 🚀 Interactive Web Application (Streamlit Studio)

Launch the interactive Pro Studio UI:

```bash
python -m streamlit run app.py
```

Open **`http://localhost:8501`** in your browser to access:
1. **📈 3D Cost Surface & Gradient Descent**: Interactive Plotly surface plots with real-time weight trajectory tracking.
2. **📊 Classifier Performance Studio**: Multi-model benchmarking leaderboards and feature importance charts.
3. **💳 Credit Risk Scoring & Explainable AI**: FICO-style credit risk scores (300 - 850) and risk tier meters.
4. **🔄 Customer Churn & LTV Optimizer**: Calculate Customer Lifetime Value and maximum profitable retention discount offers.
5. **💰 Financial ROI Campaign Optimizer**: Profit-maximizing decision threshold tuning for direct marketing campaigns.
6. **🔑 Commercial API & Developer Portal**: API key management and copy-paste integration code snippets.
7. **📁 Custom CSV Pipeline & Live Inference**: Upload any CSV dataset for automated preprocessing, model training, ROC curve plotting, and `.joblib` model downloading.
8. **🧠 Neural Network (Scratch Lab)**: Train custom Multi-Layer Perceptrons (MLPs) built from scratch using vectorized matrix propagation.

---

## 🔑 Commercial FastAPI REST API Microservice

Launch the production REST API server:

```bash
python -m uvicorn api:app --reload --port 8000
```

Access the interactive **Swagger / OpenAPI Documentation** at **`http://localhost:8000/docs`**.

### Commercial REST Endpoints:

#### 1. Compute FICO-style Credit Risk Score (`POST /v1/credit-score`)
```bash
curl -X POST "http://localhost:8000/v1/credit-score" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: sk_demo_free_123456789" \
     -d '{
       "credit_limit": 50000,
       "payment_delay_status": 1,
       "current_bill_amount": 25000,
       "last_payment_amount": 1500
     }'
```

**JSON Response**:
```json
{
  "credit_score": 642,
  "risk_tier": "High Risk ⚠️",
  "estimated_default_probability": 0.32,
  "underwriting_decision": "REJECTED (High Default Risk)"
}
```

#### 2. Calculate Customer LTV & Retention Offer (`POST /v1/churn-incentive`)
```bash
curl -X POST "http://localhost:8000/v1/churn-incentive" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: sk_demo_free_123456789" \
     -d '{
       "monthly_spend": 85.0,
       "tenure_months": 18,
       "predicted_churn_risk": 0.72
     }'
```

**JSON Response**:
```json
{
  "customer_ltv": 1147.5,
  "max_retention_discount": 229.5,
  "retention_priority": "High Risk (Aggressive Retention Offer)"
}
```

---

## 📁 Repository Structure

```text
ML-Journey-main/
├── api.py                            # Production Commercial FastAPI REST Server
├── app.py                            # Streamlit Pro Dark Web Dashboard
├── setup.py                          # Package setup and metadata (v1.0.0)
├── requirements.txt                  # Production dependencies
│
├── ml_journey/                       # Core Python Machine Learning Package
│   ├── __init__.py                   # Package exports (v1.0.0)
│   ├── metrics.py                    # Vectorized cost & gradient routines
│   ├── preprocessing.py              # Zero-variance protected normalization
│   ├── visualization.py              # 3D surface, ROC, Confusion matrix & Gauge charts
│   ├── evaluation.py                 # Stratified K-Fold CV & joblib model persistence
│   ├── tuning.py                     # Automated GridSearchCV hyperparameter tuning
│   ├── neural_net.py                 # Vectorized NumPy Multi-Layer Perceptron (MLP)
│   ├── financial.py                  # Cost-sensitive learning & profit threshold optimizer
│   ├── risk_scoring.py               # FICO Credit Score (300-850) engine
│   ├── churn.py                      # Customer Lifetime Value (LTV) & Retention optimizer
│   └── api_auth.py                   # API Key authentication & quota metering
│
├── data/                             # Real-world benchmark datasets
│   ├── bank-full.csv                 # 45,211 direct marketing records
│   └── UCI_Credit_Card.csv           # 30,000 credit applicant records
│
├── notebooks/                        # Numbered progressive case study notebooks
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_decision_trees_random_forest.ipynb
│   ├── 04_svm_classification.ipynb
│   ├── 05_neural_network_from_scratch.ipynb
│   ├── 06_real_world_bank_marketing_roi.ipynb
│   ├── 07_credit_card_risk_scoring.ipynb
│   └── 08_customer_churn_ltv_optimization.ipynb
│
└── tests/                            # Automated Pytest Suite
    ├── test_metrics.py
    ├── test_visualization.py
    ├── test_evaluation.py
    ├── test_neural_net.py
    ├── test_financial.py
    ├── test_risk_scoring.py
    ├── test_churn.py
    └── test_api.py
```

---

## 🧪 Running Automated Tests

Run the full Pytest test suite:

```bash
pytest
```

---

## 📜 License

Distributed under the MIT License. Created by **Niketan Anand**.
