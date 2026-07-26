"""
Unit tests for evaluation and model persistence module.
"""

import os
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ml_journey.evaluation import evaluate_classifier_cv, save_model, load_model


def test_evaluate_classifier_cv():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    clf = LogisticRegression()
    metrics = evaluate_classifier_cv(clf, X, y, cv_splits=3)

    assert "ROC-AUC" in metrics
    assert "F1 Score" in metrics
    assert 0.0 <= metrics["ROC-AUC"] <= 1.0


def test_model_persistence(tmp_path):
    X, y = make_classification(n_samples=50, n_features=3, random_state=42)
    clf = LogisticRegression()
    clf.fit(X, y)

    filepath = str(tmp_path / "models" / "test_model.joblib")
    save_model(clf, filepath)

    assert os.path.exists(filepath)
    loaded_clf = load_model(filepath)
    assert np.allclose(clf.predict(X), loaded_clf.predict(X))
