"""
Financial cost-sensitive learning and profit threshold optimization routines.
"""

from typing import Tuple, Dict
import numpy as np


def compute_financial_profit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    cost_per_call: float = 10.0,
    value_per_conversion: float = 100.0,
) -> Dict[str, float]:
    """
    Computes net financial profit for a direct marketing campaign based on decision threshold.

    Args:
        y_true: Ground truth binary labels (1 = subscribed, 0 = did not subscribe).
        y_proba: Predicted probability of conversion.
        threshold: Decision cutoff threshold.
        cost_per_call: Cost ($) of making a marketing call to a client.
        value_per_conversion: Net revenue ($) generated from a successful conversion.

    Returns:
        metrics: Dictionary containing 'Net Profit', 'Total Calls', 'Conversions', 'ROI %'.
    """
    y_pred = (y_proba >= threshold).astype(int)

    # True Positives: Called & Converted
    tp = np.sum((y_pred == 1) & (y_true == 1))
    # False Positives: Called & Did Not Convert
    fp = np.sum((y_pred == 1) & (y_true == 0))

    total_calls = tp + fp
    total_cost = total_calls * cost_per_call
    total_revenue = tp * value_per_conversion
    net_profit = total_revenue - total_cost

    roi = (net_profit / total_cost * 100.0) if total_cost > 0 else 0.0

    return {
        "Net Profit": float(net_profit),
        "Total Calls": int(total_calls),
        "Conversions": int(tp),
        "Total Cost": float(total_cost),
        "Total Revenue": float(total_revenue),
        "ROI %": float(roi),
    }


def find_optimal_profit_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_per_call: float = 10.0,
    value_per_conversion: float = 100.0,
    steps: int = 100,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Scans decision thresholds from 0.0 to 1.0 to find the profit-maximizing cutoff.

    Returns:
        best_threshold: Optimal decision threshold.
        max_profit: Maximum achievable net profit ($).
        profit_curve: Dict mapping threshold values to net profit.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    profit_curve = {}

    best_threshold = 0.5
    max_profit = -float("inf")

    for th in thresholds:
        res = compute_financial_profit(
            y_true, y_proba, threshold=th, cost_per_call=cost_per_call, value_per_conversion=value_per_conversion
        )
        p = res["Net Profit"]
        profit_curve[float(th)] = p

        if p > max_profit:
            max_profit = p
            best_threshold = float(th)

    return best_threshold, max_profit, profit_curve
