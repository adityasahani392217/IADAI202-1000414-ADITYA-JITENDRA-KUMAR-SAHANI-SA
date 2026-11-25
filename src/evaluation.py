"""Model evaluation, permutation importance, and fairness audit utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 for regression predictions."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def compute_subgroup_fairness(
    df_test_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_cols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate MAE, RMSE, mean bias, and R2 across demographic subgroups."""
    results: dict[str, list[dict[str, Any]]] = {}
    eval_df = df_test_meta.copy()
    eval_df["y_true"] = y_true
    eval_df["y_pred"] = y_pred
    eval_df["residual"] = y_true - y_pred

    for col in group_cols:
        if col not in eval_df.columns:
            continue
        rows = []
        for grp, sub in eval_df.groupby(col, observed=True):
            if len(sub) < 20:
                continue
            metrics = compute_regression_metrics(
                sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
            )
            rows.append(
                {
                    "subgroup": str(grp),
                    "count": int(len(sub)),
                    "mean_actual_los": round(float(sub["y_true"].mean()), 3),
                    "mean_pred_los": round(float(sub["y_pred"].mean()), 3),
                    "mean_error_bias": round(float((sub["y_pred"] - sub["y_true"]).mean()), 3),
                    **metrics,
                }
            )
        results[col] = rows
    return results


def compute_feature_importance(
    estimator: Any,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    """Compute permutation importance (MAE drop) on an unseen evaluation slice."""
    perm = permutation_importance(
        estimator,
        X_eval,
        y_eval,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    rows = []
    for idx, fname in enumerate(feature_names):
        rows.append(
            {
                "feature": fname,
                "importance_mean": round(float(perm.importances_mean[idx]), 4),
                "importance_std": round(float(perm.importances_std[idx]), 4),
            }
        )
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    return rows
