"""Unit and integration tests for ML prediction validation and artifact loading."""

from __future__ import annotations

import pytest

from src.config import MODEL_METADATA_PATH, MODEL_PIPELINE_PATH
from src.prediction import (
    load_model_metadata,
    load_trained_pipeline,
    predict_length_of_stay,
    validate_prediction_inputs,
)


def test_validate_prediction_inputs_rejects_missing_fields() -> None:
    ok, errs = validate_prediction_inputs({"age_group": "50 to 69"})
    assert ok is False
    assert len(errs) >= 1


def test_validate_prediction_inputs_rejects_invalid_category() -> None:
    fake_meta = {"feature_options": {"age_group": ["0 to 17", "50 to 69"]}}
    payload = {
        "age_group": "INVALID_AGE",
        "gender": "M",
        "type_of_admission": "Emergency",
        "emergency_department_indicator": "Y",
        "ccsr_diagnosis_code": "INF012",
        "apr_mdc_code": "04",
        "apr_medical_surgical_description": "Medical",
        "apr_severity_code": "3",
        "apr_risk_of_mortality": "Major",
    }
    ok, errs = validate_prediction_inputs(payload, metadata=fake_meta)
    assert ok is False
    assert any("age_group" in e for e in errs)


@pytest.mark.skipif(
    not (MODEL_PIPELINE_PATH.exists() and MODEL_METADATA_PATH.exists()),
    reason="Trained model artifacts not yet generated.",
)
def test_end_to_end_prediction_with_saved_artifact() -> None:
    meta = load_model_metadata()
    pipe = load_trained_pipeline()
    diag = meta["feature_options"]["diagnoses"][0]
    payload = {
        "age_group": "50 to 69",
        "gender": "M",
        "type_of_admission": "Emergency",
        "emergency_department_indicator": "Y",
        "ccsr_diagnosis_code": diag["code"],
        "apr_mdc_code": diag["mdc_code"],
        "apr_medical_surgical_description": "Medical",
        "apr_severity_code": "3",
        "apr_risk_of_mortality": "Major",
    }
    pred = predict_length_of_stay(payload, pipeline=pipe, metadata=meta)
    assert isinstance(pred, float)
    assert 1.0 <= pred <= 120.0
