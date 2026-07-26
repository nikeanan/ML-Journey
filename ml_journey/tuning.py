"""
Hyperparameter tuning routines using GridSearchCV.
"""

from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def tune_random_forest(
    X_train,
    y_train,
    param_grid: Dict[str, list] = None,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Performs grid search cross-validation hyperparameter tuning for Random Forest.

    Returns:
        best_estimator: Optimal RandomForestClassifier instance.
        best_params: Dictionary of tuned hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5],
        }

    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    grid = GridSearchCV(
        rf, param_grid, cv=cv_splits, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_
