"""
Unit tests for financial cost-sensitive optimization module.
"""

import numpy as np
import pytest
from ml_journey.financial import compute_financial_profit, find_optimal_profit_threshold


def test_compute_financial_profit():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.4, 0.1])

    # With threshold 0.5: Calls points 0, 1 -> 2 calls ($20 cost), 2 conversions ($200 revenue) -> $180 net profit
    res = compute_financial_profit(y_true, y_proba, threshold=0.5, cost_per_call=10.0, value_per_conversion=100.0)

    assert res["Net Profit"] == 180.0
    assert res["Total Calls"] == 2
    assert res["Conversions"] == 2


def test_find_optimal_profit_threshold():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.95, 0.85, 0.70, 0.60, 0.30, 0.10])

    best_th, max_p, curve = find_optimal_profit_threshold(
        y_true, y_proba, cost_per_call=10.0, value_per_conversion=100.0
    )

    assert 0.60 < best_th < 0.75  # Optimal cutoff excludes false positive at 0.60
    assert max_p > 0
    assert len(curve) > 0
