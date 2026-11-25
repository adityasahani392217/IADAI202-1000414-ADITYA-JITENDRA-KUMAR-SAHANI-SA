"""Reproducible ML training and comparison script for MediScope AI."""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.config import (
    FAIRNESS_ATTRIBUTES,
    ML_FEATURES,
    ML_TARGET,
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    MODELS_DIR,
    PROCESSED_PARQUET_PATH,
    REPORTS_DIR,
)
from src.evaluation import (
    compute_feature_importance,
    compute_regression_metrics,
    compute_subgroup_fairness,
)


def build_linear_preprocessor(features: list[str]) -> ColumnTransformer:
    """One-hot encode categoricals with frequency threshold to control memory."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=100,
                    sparse_output=True,
                ),
                features,
            )
        ],
        remainder="drop",
    )


def build_ordinal_preprocessor(features: list[str], dtype: Any = np.float64) -> ColumnTransformer:
    """Ordinal encode categoricals cleanly inside pipeline (no target leakage)."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                    dtype=dtype,
                ),
                features,
            )
        ],
        remainder="drop",
    )


def train_and_evaluate_models() -> dict[str, Any]:
    """Train Dummy, Ridge, Random Forest, and CatBoost on full 80:20 split and save best model."""
    t0 = time.time()
    if not PROCESSED_PARQUET_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run `python -m src.preprocessing` first."
        )

    needed_cols = sorted(set(ML_FEATURES + [ML_TARGET] + FAIRNESS_ATTRIBUTES))
    df = pd.read_parquet(PROCESSED_PARQUET_PATH, columns=needed_cols)

    # Convert ML features to plain strings so encoders behave consistently at train & inference time
    X = pd.DataFrame({c: df[c].astype(str) for c in ML_FEATURES})
    y = df[ML_TARGET].to_numpy(dtype=np.float32)
    meta_fairness = pd.DataFrame({c: df[c].astype(str) for c in FAIRNESS_ATTRIBUTES})

    # Reproducible 80:20 stratified split (stratified by severity code)
    X_train, X_test, y_train, y_test, _, meta_test = train_test_split(
        X,
        y,
        meta_fairness,
        test_size=0.20,
        random_state=42,
        stratify=X["apr_severity_code"],
    )
    del df, X, y, meta_fairness

    print(f"[train_model] Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # Define candidate pipelines (all transformations fitted strictly on X_train)
    candidates: dict[str, Pipeline] = {
        "Dummy Regressor (Mean Baseline)": Pipeline(
            [
                ("prep", build_ordinal_preprocessor(ML_FEATURES)),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("prep", build_linear_preprocessor(ML_FEATURES)),
                ("model", Ridge(alpha=5.0, random_state=42)),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("prep", build_ordinal_preprocessor(ML_FEATURES)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=40,
                        max_depth=14,
                        min_samples_leaf=50,
                        max_samples=200_000,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "CatBoost Regressor": Pipeline(
            [
                ("prep", build_ordinal_preprocessor(ML_FEATURES, dtype=np.int32)),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=350,
                        depth=7,
                        learning_rate=0.08,
                        loss_function="RMSE",
                        cat_features=list(range(len(ML_FEATURES))),
                        random_seed=42,
                        verbose=0,
                        thread_count=-1,
                    ),
                ),
            ]
        ),
    }

    comparison_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    test_preds: dict[str, np.ndarray] = {}

    for name, pipe in candidates.items():
        mt0 = time.time()
        print(f"[train_model] Fitting {name}...")
        pipe.fit(X_train, y_train)
        pred_train = np.clip(pipe.predict(X_train), 1.0, 120.0)
        pred_test = np.clip(pipe.predict(X_test), 1.0, 120.0)
        train_metrics = compute_regression_metrics(y_train, pred_train)
        test_metrics = compute_regression_metrics(y_test, pred_test)
        elapsed = round(time.time() - mt0, 2)

        row = {
            "model": name,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "train_time_s": elapsed,
        }
        comparison_rows.append(row)
        fitted_models[name] = pipe
        test_preds[name] = pred_test
        print(
            f"[train_model] {name} -> Test MAE={test_metrics['mae']:.3f}d, "
            f"RMSE={test_metrics['rmse']:.3f}d, R2={test_metrics['r2']:.4f} ({elapsed}s)"
        )

    # Select best model by lowest Test RMSE (and verify generalization gap)
    best_row = min(comparison_rows, key=lambda r: r["test_rmse"])
    selected_name = best_row["model"]
    selected_pipe = fitted_models[selected_name]
    selected_pred_test = test_preds[selected_name]

    # Sensitivity experiment: Admission-only Ridge (excluding APR severity & risk of mortality)
    admission_only_features = [
        "age_group",
        "gender",
        "type_of_admission",
        "emergency_department_indicator",
        "ccsr_diagnosis_code",
        "apr_mdc_code",
        "apr_medical_surgical_description",
    ]
    adm_pipe = Pipeline(
        [
            ("prep", build_linear_preprocessor(admission_only_features)),
            ("model", Ridge(alpha=5.0, random_state=42)),
        ]
    )
    adm_pipe.fit(X_train[admission_only_features], y_train)
    adm_pred = np.clip(adm_pipe.predict(X_test[admission_only_features]), 1.0, 120.0)
    adm_metrics = compute_regression_metrics(y_test, adm_pred)

    # Permutation importance on 25,000 unseen test rows for fast, reliable interpretability
    rng = np.random.default_rng(42)
    eval_idx = rng.choice(len(X_test), size=min(25_000, len(X_test)), replace=False)
    perm_importance = compute_feature_importance(
        selected_pipe,
        X_test.iloc[eval_idx],
        y_test[eval_idx],
        ML_FEATURES,
        n_repeats=5,
        random_state=42,
    )

    # Fairness audit on full test set across Race, Ethnicity, Gender, Age Group
    fairness_audit = compute_subgroup_fairness(
        meta_test, y_test, selected_pred_test, FAIRNESS_ATTRIBUTES
    )

    # Save model artifact and test evaluation scatter sample (15,000 unseen rows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_pipe, MODEL_PIPELINE_PATH, compress=3)

    sample_idx = rng.choice(len(X_test), size=min(15_000, len(X_test)), replace=False)
    eval_sample_df = pd.DataFrame(
        {
            "actual_los": y_test[sample_idx],
            "predicted_los": np.round(selected_pred_test[sample_idx], 3),
            "residual": np.round(y_test[sample_idx] - selected_pred_test[sample_idx], 3),
            "apr_severity_code": X_test["apr_severity_code"].iloc[sample_idx].values,
            "age_group": meta_test["age_group"].iloc[sample_idx].values,
            "gender": meta_test["gender"].iloc[sample_idx].values,
            "race": meta_test["race"].iloc[sample_idx].values,
        }
    )
    eval_sample_path = MODELS_DIR / "test_eval_sample.parquet"
    eval_sample_df.to_parquet(eval_sample_path, index=False)

    # Allowed categorical options for Page 8 interactive predictor
    # Map CCSR diagnosis code -> description from processed data for readable UI dropdowns
    diag_lookup_df = (
        pd.read_parquet(
            PROCESSED_PARQUET_PATH,
            columns=["ccsr_diagnosis_code", "ccsr_diagnosis_description", "apr_mdc_code", "apr_mdc_description"],
        )
        .drop_duplicates(subset=["ccsr_diagnosis_code"])
    )
    diag_options = sorted(
        [
            {
                "code": str(r["ccsr_diagnosis_code"]),
                "description": str(r["ccsr_diagnosis_description"]),
                "mdc_code": str(r["apr_mdc_code"]),
                "mdc_description": str(r["apr_mdc_description"]),
            }
            for _, r in diag_lookup_df.iterrows()
        ],
        key=lambda d: d["code"],
    )

    metadata = {
        "selected_model": selected_name,
        "validation_strategy": "Stratified 80:20 train-test split (stratified on APR Severity Code, random_state=42)",
        "chronological_validation_note": "Dataset contains a single discharge year (2021); cross-year temporal splitting is not applicable.",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": ML_FEATURES,
        "target": ML_TARGET,
        "excluded_leakage_columns": [
            "total_charges",
            "total_costs",
            "patient_disposition",
            "ccsr_procedure_code",
            "apr_drg_code",
            "cost_per_day",
            "charge_per_day",
            "charge_to_cost_ratio",
        ],
        "model_comparison": comparison_rows,
        "admission_only_ridge_sensitivity": {
            "features": admission_only_features,
            "test_metrics": adm_metrics,
        },
        "permutation_importance": perm_importance,
        "fairness_audit": fairness_audit,
        "feature_options": {
            "age_group": sorted(X_train["age_group"].unique().tolist()),
            "gender": sorted(X_train["gender"].unique().tolist()),
            "type_of_admission": sorted(X_train["type_of_admission"].unique().tolist()),
            "emergency_department_indicator": sorted(
                X_train["emergency_department_indicator"].unique().tolist()
            ),
            "apr_medical_surgical_description": sorted(
                X_train["apr_medical_surgical_description"].unique().tolist()
            ),
            "apr_severity_code": sorted(X_train["apr_severity_code"].unique().tolist()),
            "apr_risk_of_mortality": sorted(X_train["apr_risk_of_mortality"].unique().tolist()),
            "apr_mdc_code": sorted(X_train["apr_mdc_code"].unique().tolist()),
            "diagnoses": diag_options,
        },
        "model_size_mb": round(MODEL_PIPELINE_PATH.stat().st_size / (1024 * 1024), 2),
        "total_training_elapsed_s": round(time.time() - t0, 2),
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[train_model] Selected: {selected_name} ({metadata['model_size_mb']} MB). "
        f"Metadata saved to {MODEL_METADATA_PATH}."
    )
    return metadata


if __name__ == "__main__":
    train_and_evaluate_models()
