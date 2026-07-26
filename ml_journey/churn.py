"""
Customer Churn Risk and Lifetime Value (LTV) incentive optimization utilities.
"""

from typing import Dict


def calculate_customer_ltv(
    monthly_spend: float, tenure_months: int, margin_rate: float = 0.75
) -> float:
    """
    Computes estimated Customer Lifetime Value (LTV).

    Args:
        monthly_spend: Average monthly bill amount ($).
        tenure_months: Customer tenure in months.
        margin_rate: Operating profit margin percentage (default 75%).

    Returns:
        ltv: Estimated Customer Lifetime Value ($).
    """
    expected_future_months = max(12, tenure_months)
    return float(monthly_spend * expected_future_months * margin_rate)


def calculate_max_retention_discount(
    churn_prob: float, ltv: float, max_discount_pct: float = 0.20
) -> Dict[str, float]:
    """
    Calculates the optimal retention discount incentive offer to prevent customer churn.

    Args:
        churn_prob: Predicted probability of churn (0.0 to 1.0).
        ltv: Calculated Customer Lifetime Value ($).
        max_discount_pct: Maximum allowed discount percentage of LTV.

    Returns:
        metrics: Dictionary containing 'Max Retention Discount ($)', 'Retention Priority', 'Net Retained LTV ($)'.
    """
    if churn_prob < 0.30:
        recommended_discount = 0.0
        priority = "Low Risk (No Action Needed)"
    elif churn_prob < 0.65:
        recommended_discount = ltv * (max_discount_pct * 0.5)
        priority = "Moderate Risk (Standard Discount Offer)"
    else:
        recommended_discount = ltv * max_discount_pct
        priority = "High Risk (Aggressive Retention Offer)"

    net_retained_ltv = ltv - recommended_discount

    return {
        "Max Retention Discount ($)": float(recommended_discount),
        "Retention Priority": priority,
        "Net Retained LTV ($)": float(net_retained_ltv),
    }
