"""
Unit tests for credit risk scoring module.
"""

import pytest
from ml_journey.risk_scoring import probability_to_credit_score, get_credit_risk_tier


def test_probability_to_credit_score():
    # Low default probability -> high credit score
    score_low_risk = probability_to_credit_score(0.01)
    assert score_low_risk > 750

    # High default probability -> low credit score
    score_high_risk = probability_to_credit_score(0.80)
    assert score_high_risk < 550

    # Clamping boundaries
    assert 300 <= probability_to_credit_score(1.0) <= 850
    assert 300 <= probability_to_credit_score(0.0) <= 850


def test_get_credit_risk_tier():
    label, color = get_credit_risk_tier(500)
    assert "Very High" in label

    label_good, _ = get_credit_risk_tier(780)
    assert "Good" in label_good
