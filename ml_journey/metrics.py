"""
Cost and gradient calculation routines for linear regression.
Includes both vectorized (matrix ops) and explicit loop implementations.
"""

from typing import Tuple
import numpy as np


def compute_cost_matrix(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, verbose: bool = False
) -> float:
    """
    Computes the Mean Squared Error cost for linear regression using vectorized operations.

    Args:
        X: (ndarray Shape (m, n)) Matrix of m examples with n features.
        y: (ndarray Shape (m,)) Target values vector.
        w: (ndarray Shape (n,)) Parameter weights vector.
        b: Model bias scalar.
        verbose: If True, prints intermediate predictions.

    Returns:
        total_cost: Calculated MSE cost (scalar).
    """
    m = X.shape[0]
    f_wb = X @ w + b
    total_cost = float((1.0 / (2.0 * m)) * np.sum((f_wb - y) ** 2))

    if verbose:
        print("f_wb:", f_wb)

    return total_cost


def compute_gradient_matrix(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> Tuple[float, np.ndarray]:
    """
    Computes the gradient of the cost function w.r.t parameters w and b using vectorization.

    Args:
        X: (ndarray Shape (m, n)) Matrix of m examples with n features.
        y: (ndarray Shape (m,)) Target values vector.
        w: (ndarray Shape (n,)) Parameter weights vector.
        b: Model bias scalar.

    Returns:
        dj_db: Gradient with respect to bias parameter b.
        dj_dw: Gradient array with respect to weights parameters w.
    """
    m, _ = X.shape
    f_wb = X @ w + b
    err = f_wb - y
    dj_dw = (1.0 / m) * (X.T @ err)
    dj_db = float((1.0 / m) * np.sum(err))

    return dj_db, dj_dw


def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Computes cost using explicit loop iterations for educational/comparison benchmarking.
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = float(np.dot(X[i], w) + b)
        cost += (f_wb_i - y[i]) ** 2
    return float(cost / (2.0 * m))


def compute_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> Tuple[float, np.ndarray]:
    """
    Computes gradient using explicit loop iterations for educational benchmarking.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,), dtype=float)
    dj_db = 0.0

    for i in range(m):
        err = float((np.dot(X[i], w) + b) - y[i])
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    return float(dj_db / m), dj_dw / m
