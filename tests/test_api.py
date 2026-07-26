"""
Unit tests for FastAPI REST API Microservice endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)
VALID_KEY = "sk_demo_free_123456789"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Operational"


def test_credit_score_unauthorized():
    response = client.post("/v1/credit-score", json={
        "credit_limit": 50000,
        "payment_delay_status": 0,
        "current_bill_amount": 10000,
        "last_payment_amount": 2000
    })
    assert response.status_code == 401


def test_credit_score_authorized():
    headers = {"X-API-Key": VALID_KEY}
    payload = {
        "credit_limit": 50000.0,
        "payment_delay_status": 0,
        "current_bill_amount": 10000.0,
        "last_payment_amount": 2000.0
    }
    response = client.post("/v1/credit-score", headers=headers, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 300 <= data["credit_score"] <= 850
    assert "underwriting_decision" in data


def test_churn_incentive_authorized():
    headers = {"X-API-Key": VALID_KEY}
    payload = {
        "monthly_spend": 100.0,
        "tenure_months": 24,
        "predicted_churn_risk": 0.80
    }
    response = client.post("/v1/churn-incentive", headers=headers, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_ltv"] > 0
    assert data["max_retention_discount"] > 0
