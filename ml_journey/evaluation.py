"""
Model evaluation, cross-validation benchmarking, and model persistence utilities.
"""

from typing import Dict, Any
import os
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_classifier_cv(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Performs Stratified K-Fold cross-validation and computes mean evaluation metrics.

    Args:
        estimator: Scikit-Learn classifier or pipeline.
        X: Feature array or dataframe.
        y: Target values array.
        cv_splits: Number of cross-validation folds.
        random_state: Random state seed.

    Returns:
        metrics: Dictionary containing mean 'ROC-AUC', 'F1 Score', 'Precision', 'Recall', and 'Accuracy'.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    }

    scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        "ROC-AUC": float(np.mean(scores["test_roc_auc"])),
        "F1 Score": float(np.mean(scores["test_f1"])),
        "Precision": float(np.mean(scores["test_precision"])),
        "Recall": float(np.mean(scores["test_recall"])),
        "Accuracy": float(np.mean(scores["test_accuracy"])),
    }


def save_model(model: Any, filepath: str) -> str:
    """
    Serializes and saves a trained Scikit-Learn model/pipeline to disk using joblib.

    Args:
        model: Trained model object.
        filepath: Save path (e.g. 'models/random_forest.joblib').

    Returns:
        filepath: Confirmed file save path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    return filepath


def load_model(filepath: str) -> Any:
    """
    Loads a saved model object from disk using joblib.

    Args:
        filepath: Path to saved joblib model artifact.

    Returns:
        model: Loaded model instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)
