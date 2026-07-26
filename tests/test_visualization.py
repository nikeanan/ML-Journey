"""
Unit tests for visualization routines.
"""

import numpy as np
import plotly.graph_objects as go
from ml_journey.visualization import plot_cost_surface_3d, plot_model_comparison_bar


def test_plot_cost_surface_3d():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    fig = plot_cost_surface_3d(X, y, w_range=[-5, 5, 10], b_range=[-5, 5, 10])
    assert isinstance(fig, go.Figure)


def test_plot_model_comparison_bar():
    metrics = {
        "RandomForest": {"ROC-AUC": 0.88, "F1": 0.82},
        "LogisticRegression": {"ROC-AUC": 0.81, "F1": 0.75},
    }
    fig = plot_model_comparison_bar(metrics)
    assert isinstance(fig, go.Figure)
