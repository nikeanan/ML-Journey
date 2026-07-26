import os
import numpy as np
from ml_journey.visualization import plot_cost_surface_3d, plot_model_comparison_bar

def main():
    # 1. Generate demo linear regression data
    np.random.seed(42)
    X = np.random.randn(100, 1) * 2.0
    true_w = np.array([3.5])
    true_b = 1.2
    y = (X @ true_w + true_b).flatten() + np.random.randn(100) * 0.5

    # Simulated gradient descent history
    history = {
        "params": [
            [np.array([-50.0]), -100.0],
            [np.array([0.0]), -20.0],
            [np.array([2.5]), 0.5],
            [np.array([3.5]), 1.2],
        ],
        "cost": [12000.0, 1500.0, 50.0, 0.25],
    }

    # Generate 3D Cost Surface plot
    fig_3d = plot_cost_surface_3d(X, y, w_range=[-10, 20, 30], b_range=[-50, 50, 30], history=history)
    fig_3d.write_html("cost_surface_3d.html")
    print("✓ Saved 3D Cost Surface plot to 'cost_surface_3d.html'")

    # 2. Model Comparison Metrics
    benchmark_metrics = {
        "Random Forest": {
            "ROC-AUC": 0.912,
            "F1 Score": 0.865,
            "Precision": 0.842,
            "Recall": 0.890,
        },
        "Logistic Regression": {
            "ROC-AUC": 0.835,
            "F1 Score": 0.771,
            "Precision": 0.790,
            "Recall": 0.753,
        },
        "Support Vector Machine": {
            "ROC-AUC": 0.884,
            "F1 Score": 0.829,
            "Precision": 0.830,
            "Recall": 0.828,
        },
        "Gradient Boosting": {
            "ROC-AUC": 0.925,
            "F1 Score": 0.881,
            "Precision": 0.865,
            "Recall": 0.898,
        },
    }

    fig_bar = plot_model_comparison_bar(benchmark_metrics)
    fig_bar.write_html("model_comparison_bar.html")
    print("✓ Saved Model Comparison plot to 'model_comparison_bar.html'")

if __name__ == "__main__":
    main()
