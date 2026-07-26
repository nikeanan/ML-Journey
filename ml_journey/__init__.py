"""
ML-Journey Package: Core numerical algorithms, ML utilities, and visualization routines.
"""

from .metrics import compute_cost_matrix, compute_gradient_matrix, compute_cost, compute_gradient
from .preprocessing import zscore_normalize_features
from .visualization import plot_cost_surface_3d, plot_model_comparison_bar, plot_feature_importance, plot_confusion_matrix, plot_roc_curve, plot_profit_vs_threshold, plot_credit_score_gauge
from .evaluation import evaluate_classifier_cv, save_model, load_model
from .tuning import tune_random_forest
from .neural_net import DenseLayer, NeuralNetwork
from .financial import compute_financial_profit, find_optimal_profit_threshold
from .risk_scoring import probability_to_credit_score, get_credit_risk_tier
from .churn import calculate_customer_ltv, calculate_max_retention_discount

__all__ = [
    "compute_cost_matrix",
    "compute_gradient_matrix",
    "compute_cost",
    "compute_gradient",
    "zscore_normalize_features",
    "plot_cost_surface_3d",
    "plot_model_comparison_bar",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_profit_vs_threshold",
    "plot_credit_score_gauge",
    "evaluate_classifier_cv",
    "save_model",
    "load_model",
    "tune_random_forest",
    "DenseLayer",
    "NeuralNetwork",
    "compute_financial_profit",
    "find_optimal_profit_threshold",
    "probability_to_credit_score",
    "get_credit_risk_tier",
    "calculate_customer_ltv",
    "calculate_max_retention_discount",
]
__version__ = "0.8.0"







