"""
Interactive visualization routines using Plotly for cost surfaces and model evaluation metrics.
"""

from typing import Dict, List, Optional
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .metrics import compute_cost_matrix



def plot_cost_surface_3d(
    X: np.ndarray,
    y: np.ndarray,
    w_range: List[float] = [-100, 500, 50],
    b_range: List[float] = [-200, 200, 50],
    history: Optional[Dict] = None,
) -> go.Figure:
    """
    Generates an interactive 3D surface plot of the cost function J(w,b) with optional gradient descent trajectory.

    Args:
        X: Feature matrix.
        y: Target array.
        w_range: [w_min, w_max, steps] range for w grid.
        b_range: [b_min, b_max, steps] range for b grid.
        history: Optional gradient descent history dictionary containing 'params'.

    Returns:
        fig: Interactive Plotly Figure object.
    """
    w_vals = np.linspace(w_range[0], w_range[1], int(w_range[2]))
    b_vals = np.linspace(b_range[0], b_range[1], int(b_range[2]))
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = compute_cost_matrix(X, y, np.array([W[i, j]]), B[i, j])

    fig = go.Figure(
        data=[
            go.Surface(
                x=W,
                y=B,
                z=Z,
                colorscale="Viridis",
                opacity=0.8,
                name="Cost Surface J(w,b)",
            )
        ]
    )

    if history and "params" in history:
        w_hist = [p[0][0] if isinstance(p[0], np.ndarray) else p[0] for p in history["params"]]
        b_hist = [p[1] for p in history["params"]]
        cost_hist = history.get("cost", [])

        fig.add_trace(
            go.Scatter3d(
                x=w_hist,
                y=b_hist,
                z=cost_hist,
                mode="lines+markers",
                marker=dict(size=4, color="red"),
                line=dict(color="red", width=4),
                name="Gradient Descent Path",
            )
        )

    fig.update_layout(
        title="Interactive 3D Cost Surface J(w,b)",
        scene=dict(
            xaxis_title="Weight (w)",
            yaxis_title="Bias (b)",
            zaxis_title="Cost J(w,b)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def plot_model_comparison_bar(
    metrics_dict: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Generates an interactive grouped bar chart comparing evaluation metrics across models.

    Args:
        metrics_dict: Dict mapping model names to metric dictionaries, e.g.:
            {'RandomForest': {'ROC-AUC': 0.88, 'F1': 0.82}, 'LogisticRegression': {...}}

    Returns:
        fig: Interactive Plotly Figure object.
    """
    models = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())

    fig = go.Figure()
    for metric in metric_names:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=models,
                y=[metrics_dict[m][metric] for m in models],
            )
        )

    fig.update_layout(
        barmode="group",
        title="Classifier Benchmark Performance Comparison",
        xaxis_title="Machine Learning Model",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )
    return fig


def plot_feature_importance(
    feature_names: List[str], importances: np.ndarray, top_n: int = 15
) -> go.Figure:
    """
    Generates an interactive Plotly horizontal bar chart of top predictive feature importances.

    Args:
        feature_names: List of feature names.
        importances: Array of feature importance values.
        top_n: Number of top features to display.

    Returns:
        fig: Interactive Plotly Figure.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices][::-1]
    top_vals = importances[indices][::-1]

    fig = go.Figure(
        go.Bar(
            x=top_vals,
            y=top_names,
            orientation="h",
            marker=dict(color="teal"),
        )
    )

    fig.update_layout(
        title=f"Top {top_n} Predictive Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_white",
        margin=dict(l=150),
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = ["Negative", "Positive"]) -> go.Figure:
    """
    Generates an interactive Plotly heatmap visualization of a confusion matrix.

    Args:
        cm: (2, 2) or (k, k) confusion matrix array.
        labels: Class display labels.

    Returns:
        fig: Interactive Plotly heatmap Figure.
    """
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
    )
    fig.update_layout(
        title="Interactive Confusion Matrix Heatmap",
        template="plotly_dark",
    )
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> go.Figure:
    """
    Generates an interactive Plotly ROC curve plot.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.
        auc_score: Calculated ROC-AUC score.

    Returns:
        fig: Interactive Plotly line plot Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {auc_score:.4f})",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        template="plotly_dark",
    )
    return fig


def plot_profit_vs_threshold(
    profit_curve: Dict[float, float], best_threshold: float, max_profit: float
) -> go.Figure:
    """
    Generates an interactive Plotly curve of Net Campaign Profit ($) vs Decision Threshold.

    Args:
        profit_curve: Dict mapping threshold float to profit float.
        best_threshold: Optimal decision cutoff.
        max_profit: Maximum net profit value ($).

    Returns:
        fig: Interactive Plotly Figure.
    """
    th_vals = list(profit_curve.keys())
    p_vals = list(profit_curve.values())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=th_vals,
            y=p_vals,
            mode="lines",
            name="Net Campaign Profit ($)",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_threshold],
            y=[max_profit],
            mode="markers+text",
            name=f"Optimal Cutoff ({best_threshold:.2f}, ${max_profit:,.0f})",
            marker=dict(size=12, color="#c084fc"),
            text=[f"  Max Profit: ${max_profit:,.0f}"],
            textposition="top right",
        )
    )
    fig.update_layout(
        title="Campaign Net Profit ($) vs Decision Threshold",
        xaxis_title="Decision Cutoff Threshold",
        yaxis_title="Net Campaign Profit ($)",
        template="plotly_dark",
    )
    return fig


def plot_credit_score_gauge(score: int, risk_label: str) -> go.Figure:
    """
    Generates an interactive Plotly Credit Risk Score Gauge Chart (300 to 850).

    Args:
        score: Calculated credit score (300-850).
        risk_label: Risk tier label.

    Returns:
        fig: Interactive Plotly Gauge Figure.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Credit Risk Score: {risk_label}", "font": {"size": 20, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [300, 850], "tickwidth": 2, "tickcolor": "#94a3b8"},
                "bar": {"color": "#38bdf8", "thickness": 0.3},
                "bgcolor": "#0f172a",
                "borderwidth": 2,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [300, 580], "color": "#ef4444"},
                    {"range": [580, 670], "color": "#f97316"},
                    {"range": [670, 740], "color": "#eab308"},
                    {"range": [740, 800], "color": "#22c55e"},
                    {"range": [800, 850], "color": "#3b82f6"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig




