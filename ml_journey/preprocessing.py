"""
Preprocessing routines for feature scaling and normalization.
"""

from typing import Tuple, Union
import numpy as np


def zscore_normalize_features(
    X: np.ndarray, return_stats: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes Z-score feature normalization on input matrix X.

    Args:
        X: (ndarray Shape (m, n)) Feature matrix.
        return_stats: If True, returns (X_norm, mu, sigma).

    Returns:
        X_norm: Normalized feature array.
        mu (optional): Computed column means.
        sigma (optional): Computed column standard deviations.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    # Avoid division by zero for constant features
    sigma_adj = np.where(sigma == 0, 1e-8, sigma)
    X_norm = (X - mu) / sigma_adj

    if return_stats:
        return X_norm, mu, sigma
    return X_norm
