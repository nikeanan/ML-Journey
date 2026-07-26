"""
Credit Risk Scoring and Explainable AI utilities for financial default risk modeling.
"""

from typing import Dict, Tuple, Any
import numpy as np


def probability_to_credit_score(
    default_prob: float, base_score: int = 600, pdo: int = 50
) -> int:
    """
    Converts a predicted default probability into a standard Credit Risk Score (300 to 850).

    Args:
        default_prob: Predicted default probability (0.0 to 1.0).
        base_score: Base credit score anchor (default 600).
        pdo: Points to Double Odds (PDO).

    Returns:
        score: Calculated Credit Score integer bounded between 300 and 850.
    """
    prob_clamped = np.clip(default_prob, 0.0001, 0.9999)
    odds = (1.0 - prob_clamped) / prob_clamped
    factor = pdo / np.log(2.0)
    score = int(base_score + factor * np.log(odds))
    return int(np.clip(score, 300, 850))


def get_credit_risk_tier(credit_score: int) -> Tuple[str, str]:
    """
    Categorizes credit score into financial risk tier and color code.

    Returns:
        tier_label: Category ('Very High Risk', 'High Risk', 'Moderate Risk', 'Good', 'Excellent').
        color_code: Hex color code.
    """
    if credit_score < 580:
        return "Very High Risk 🚨", "#ef4444"
    elif credit_score < 670:
        return "High Risk ⚠️", "#f97316"
    elif credit_score < 740:
        return "Moderate Risk 🟡", "#eab308"
    elif credit_score < 800:
        return "Good Credit 🟢", "#22c55e"
    else:
        return "Excellent Credit 🌟", "#3b82f6"
