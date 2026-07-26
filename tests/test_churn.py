"""
Unit tests for churn risk and LTV incentive module.
"""

import pytest
from ml_journey.churn import calculate_customer_ltv, calculate_max_retention_discount


def test_calculate_customer_ltv():
    ltv = calculate_customer_ltv(monthly_spend=100.0, tenure_months=24, margin_rate=0.75)
    assert ltv == 1800.0


def test_calculate_max_retention_discount():
    ltv = 2000.0
    res_high = calculate_max_retention_discount(churn_prob=0.80, ltv=ltv)
    assert res_high["Max Retention Discount ($)"] == 400.0
    assert "High Risk" in res_high["Retention Priority"]

    res_low = calculate_max_retention_discount(churn_prob=0.10, ltv=ltv)
    assert res_low["Max Retention Discount ($)"] == 0.0
    assert "Low Risk" in res_low["Retention Priority"]
