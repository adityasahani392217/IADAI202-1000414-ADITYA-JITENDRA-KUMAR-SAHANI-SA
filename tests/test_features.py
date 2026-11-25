"""Unit tests for feature engineering and zero-division guards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import add_engineered_features


def test_feature_engineering_ratios_and_flags() -> None:
    df = pd.DataFrame(
        {
            "age_group": ["18 to 29", "70 or Older"],
            "apr_severity_description": ["Minor", "Extreme"],
            "emergency_department_indicator": ["N", "Y"],
            "type_of_admission": ["Newborn", "Emergency"],
            "birth_weight_grams": [3100.0, np.nan],
            "payment_typology_1": ["Medicaid", "Medicare"],
            "payment_typology_2": ["None", "Medicaid"],
            "payment_typology_3": ["None", "Self-Pay"],
            "length_of_stay": [2.0, 20.0],
            "total_charges": [10000.0, 100000.0],
            "total_costs": [2500.0, 50000.0],
        }
    )
    out = add_engineered_features(df)
    assert out["age_ordinal"].tolist() == [1, 4]
    assert out["severity_numeric"].tolist() == [1, 4]
    assert out["is_ed_admission"].tolist() == [0, 1]
    assert out["is_newborn"].tolist() == [True, False]
    assert out["payer_count"].tolist() == [1, 3]
    assert np.isclose(out["cost_per_day"].iloc[0], 1250.0)
    assert np.isclose(out["charge_to_cost_ratio"].iloc[1], 2.0)


def test_feature_engineering_zero_division_protection() -> None:
    df = pd.DataFrame(
        {
            "age_group": ["50 to 69"],
            "apr_severity_description": ["Moderate"],
            "emergency_department_indicator": ["Y"],
            "type_of_admission": ["Emergency"],
            "payment_typology_1": ["Medicare"],
            "payment_typology_2": ["None"],
            "payment_typology_3": ["None"],
            "length_of_stay": [0.0],
            "total_charges": [1000.0],
            "total_costs": [0.0],
        }
    )
    out = add_engineered_features(df)
    assert pd.isna(out["cost_per_day"].iloc[0])
    assert pd.isna(out["charge_per_day"].iloc[0])
    assert pd.isna(out["charge_to_cost_ratio"].iloc[0])
