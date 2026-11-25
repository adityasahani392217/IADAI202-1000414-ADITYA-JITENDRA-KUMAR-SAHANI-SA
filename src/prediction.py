"""Inference and input-validation utilities for the MediScope AI LOS Predictor."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ML_FEATURES, MODEL_METADATA_PATH, MODEL_PIPELINE_PATH


def load_model_metadata() -> dict[str, Any]:
    """Load saved ML metadata JSON."""
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model metadata not found (`models/model_metadata.json`). Run `python -m src.train_model` first."
        )
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trained_pipeline() -> Any:
    """Load the saved preprocessing + regression pipeline."""
    if not MODEL_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            "Trained model artifact not found (`models/los_model_pipeline.joblib`). "
            "Run `python -m src.train_model` first."
        )
    return joblib.load(MODEL_PIPELINE_PATH)


def validate_prediction_inputs(
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Validate that all required prediction features are present and non-empty."""
    errors: list[str] = []
    for feat in ML_FEATURES:
        val = payload.get(feat)
        if val is None or str(val).strip() == "":
            errors.append(f"Missing required input field: `{feat}`")

    if metadata and "feature_options" in metadata:
        opts = metadata["feature_options"]
        for feat in (
            "age_group",
            "gender",
            "type_of_admission",
            "emergency_department_indicator",
            "apr_medical_surgical_description",
            "apr_severity_code",
            "apr_risk_of_mortality",
        ):
            allowed = set(opts.get(feat, []))
            if allowed and str(payload.get(feat, "")) not in allowed:
                errors.append(
                    f"Invalid value '{payload.get(feat)}' for `{feat}`. Allowed: {sorted(allowed)}"
                )

    return len(errors) == 0, errors


def predict_length_of_stay(
    payload: dict[str, Any],
    pipeline: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Validate inputs, transform via saved pipeline, and return predicted LOS in days."""
    meta = metadata if metadata is not None else load_model_metadata()
    is_valid, errors = validate_prediction_inputs(payload, meta)
    if not is_valid:
        raise ValueError("; ".join(errors))

    model = pipeline if pipeline is not None else load_trained_pipeline()
    input_df = pd.DataFrame([{feat: str(payload[feat]).strip() for feat in ML_FEATURES}])
    pred = float(model.predict(input_df)[0])
    # Clamp to valid hospital inpatient stay domain [1.0, 120.0]
    return round(float(np.clip(pred, 1.0, 120.0)), 2)
