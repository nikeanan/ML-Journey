"""
Unit tests for numerical algorithms in ml_journey package.
"""

import numpy as np
import pytest
from ml_journey.metrics import (
    compute_cost,
    compute_cost_matrix,
    compute_gradient,
    compute_gradient_matrix,
)
from ml_journey.preprocessing import zscore_normalize_features


@pytest.fixture
def sample_regression_data():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    true_w = np.array([1.5, -2.0, 0.5, 3.0])
    true_b = 0.5
    y = X @ true_w + true_b + np.random.randn(100) * 0.1
    w_init = np.zeros(4)
    b_init = 0.0
    return X, y, w_init, b_init


def test_cost_computation_equivalence(sample_regression_data):
    X, y, w, b = sample_regression_data
    cost_loop = compute_cost(X, y, w, b)
    cost_matrix = compute_cost_matrix(X, y, w, b)
    assert np.isclose(cost_loop, cost_matrix, atol=1e-6)


def test_gradient_computation_equivalence(sample_regression_data):
    X, y, w, b = sample_regression_data
    db_loop, dw_loop = compute_gradient(X, y, w, b)
    db_matrix, dw_matrix = compute_gradient_matrix(X, y, w, b)

    assert np.isclose(db_loop, db_matrix, atol=1e-6)
    assert np.allclose(dw_loop, dw_matrix, atol=1e-6)


def test_zscore_normalization():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    X_norm, mu, sigma = zscore_normalize_features(X, return_stats=True)

    assert np.allclose(mu, np.array([2.0, 20.0]))
    assert np.allclose(np.mean(X_norm, axis=0), 0.0, atol=1e-6)
    assert np.allclose(np.std(X_norm, axis=0), 1.0, atol=1e-6)
