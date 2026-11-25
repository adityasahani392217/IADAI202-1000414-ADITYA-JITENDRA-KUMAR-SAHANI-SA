"""Feature engineering helpers for MediScope AI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AGE_GROUP_ORDER, SEVERITY_ORDER


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-grounded analytical features with safe numeric guards."""
    out = df.copy()

    # 1. Age ordinal mapping (0 to 4)
    age_map = {label: idx for idx, label in enumerate(AGE_GROUP_ORDER)}
    if "age_group" in out.columns:
        out["age_ordinal"] = (
            out["age_group"].astype(str).map(age_map).fillna(-1).astype("int8")
        )

    # 2. Severity numeric mapping (1=Minor .. 4=Extreme, 0=Not Applicable)
    sev_map = {"Not Applicable": 0, "Minor": 1, "Moderate": 2, "Major": 3, "Extreme": 4}
    if "apr_severity_description" in out.columns:
        out["severity_numeric"] = (
            out["apr_severity_description"].astype(str).map(sev_map).fillna(0).astype("int8")
        )

    # 3. Emergency Department binary indicator
    if "emergency_department_indicator" in out.columns:
        out["is_ed_admission"] = (
            out["emergency_department_indicator"].astype(str).str.upper() == "Y"
        ).astype("int8")

    # 4. Newborn admission indicator
    if "type_of_admission" in out.columns:
        is_nb = out["type_of_admission"].astype(str) == "Newborn"
        if "birth_weight_grams" in out.columns:
            is_nb = is_nb | out["birth_weight_grams"].notna()
        out["is_newborn"] = is_nb.astype("bool")

    # 5. Payer count (1, 2, or 3 active payment typologies)
    if all(c in out.columns for c in ("payment_typology_1", "payment_typology_2", "payment_typology_3")):
        p1 = (~out["payment_typology_1"].astype(str).isin(["None", "Unknown", "nan", ""])).astype("int8")
        p2 = (~out["payment_typology_2"].astype(str).isin(["None", "Unknown", "nan", ""])).astype("int8")
        p3 = (~out["payment_typology_3"].astype(str).isin(["None", "Unknown", "nan", ""])).astype("int8")
        out["payer_count"] = (p1 + p2 + p3).astype("int8")

    # 6. Financial efficiency ratios (protected against division by zero)
    if {"total_costs", "length_of_stay"}.issubset(out.columns):
        los_safe = np.where(out["length_of_stay"] > 0, out["length_of_stay"], np.nan)
        out["cost_per_day"] = (out["total_costs"] / los_safe).astype("float32")
    if {"total_charges", "length_of_stay"}.issubset(out.columns):
        los_safe = np.where(out["length_of_stay"] > 0, out["length_of_stay"], np.nan)
        out["charge_per_day"] = (out["total_charges"] / los_safe).astype("float32")
    if {"total_charges", "total_costs"}.issubset(out.columns):
        cost_safe = np.where(out["total_costs"] > 0, out["total_costs"], np.nan)
        out["charge_to_cost_ratio"] = (out["total_charges"] / cost_safe).astype("float32")

    # 7. Tukey IQR outlier flag on Length of Stay (retained, not deleted)
    if "length_of_stay" in out.columns and len(out) > 0:
        q1 = float(out["length_of_stay"].quantile(0.25))
        q3 = float(out["length_of_stay"].quantile(0.75))
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        out["los_iqr_outlier"] = (out["length_of_stay"] > upper_fence).astype("bool")

    return out


def add_benchmark_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute diagnosis- and facility-level contextual benchmarks for EDA only.

    WARNING: Do not pass these group-aggregated columns into ML model training
    on the same split, as they encode target statistics.
    """
    out = df.copy()
    if "ccsr_diagnosis_code" in out.columns:
        diag_stats = out.groupby("ccsr_diagnosis_code", observed=True).agg(
            diag_avg_los=("length_of_stay", "mean"),
            diag_avg_charges=("total_charges", "mean"),
            diag_avg_costs=("total_costs", "mean"),
        )
        out = out.merge(diag_stats, on="ccsr_diagnosis_code", how="left")

    if "permanent_facility_id" in out.columns:
        fac_stats = out.groupby("permanent_facility_id", observed=True).agg(
            facility_avg_los=("length_of_stay", "mean"),
            facility_discharge_volume=("length_of_stay", "count"),
        )
        out = out.merge(fac_stats, on="permanent_facility_id", how="left")

    return out
