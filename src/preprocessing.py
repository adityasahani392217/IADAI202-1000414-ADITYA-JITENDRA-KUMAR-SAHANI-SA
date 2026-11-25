"""Modular data cleaning and preprocessing pipeline for MediScope AI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    AGE_GROUP_ORDER,
    COLUMN_RENAME_MAP,
    PROCESSED_DATA_DIR,
    PROCESSED_PARQUET_PATH,
    REPORTS_DIR,
    RISK_ORDER,
    SEVERITY_ORDER,
    resolve_raw_csv_path,
)
from src.data_loader import iter_raw_chunks


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw SPARCS columns to consistent snake_case identifiers."""
    renamed = df.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns})
    # Fallback normalization for any unmapped columns
    renamed.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") for c in renamed.columns
    ]
    return renamed


def parse_monetary_series(series: pd.Series) -> pd.Series:
    """Parse comma/currency-formatted strings into positive float64 values."""
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = (
            series.astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        vals = pd.to_numeric(cleaned, errors="coerce")
    return vals.astype("float64")


def parse_length_of_stay(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse Length of Stay strings, handling NYS SPARCS '120 +' top-coding.

    Returns:
        los_numeric: float32 array where '120 +' is mapped to 120.0.
        is_topcoded: boolean Series flagging top-coded stays (>= 120 days).
    """
    s_str = series.astype("string").str.strip()
    is_topcoded = (s_str == "120 +").fillna(False)
    mapped = s_str.where(~is_topcoded, "120")
    los_num = pd.to_numeric(mapped, errors="coerce").astype("float32")
    # Invalidate impossible non-positive values
    los_num = los_num.where(los_num > 0)
    return los_num, is_topcoded.astype("bool")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented, domain-appropriate missing-value imputation and flagging."""
    out = df.copy()

    # Facility / geography fields redacted for confidentiality (10,642 rows)
    for col in ("hospital_service_area", "hospital_county", "permanent_facility_id"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().fillna("Redacted / Unknown")

    for col in ("facility_name",):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().fillna("Unknown Facility")

    for col in ("operating_certificate_number", "zip_code_3"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().fillna("Unknown")

    # Diagnosis & procedure fields
    if "ccsr_diagnosis_code" in out.columns:
        out["ccsr_diagnosis_code"] = (
            out["ccsr_diagnosis_code"].astype("string").str.strip().fillna("UNSPECIFIED")
        )
    if "ccsr_diagnosis_description" in out.columns:
        out["ccsr_diagnosis_description"] = (
            out["ccsr_diagnosis_description"]
            .astype("string")
            .str.strip()
            .fillna("Unspecified Diagnosis")
        )

    if "ccsr_procedure_code" in out.columns:
        out["ccsr_procedure_code"] = (
            out["ccsr_procedure_code"].astype("string").str.strip().fillna("NONE")
        )
    if "ccsr_procedure_description" in out.columns:
        out["ccsr_procedure_description"] = (
            out["ccsr_procedure_description"]
            .astype("string")
            .str.strip()
            .fillna("No Procedure Recorded")
        )

    # APR Severity & Mortality Risk (Code '0' corresponds to 'Not Applicable')
    if "apr_severity_description" in out.columns:
        out["apr_severity_description"] = (
            out["apr_severity_description"]
            .astype("string")
            .str.strip()
            .fillna("Not Applicable")
        )
    if "apr_risk_of_mortality" in out.columns:
        out["apr_risk_of_mortality"] = (
            out["apr_risk_of_mortality"].astype("string").str.strip().fillna("Not Applicable")
        )

    # Secondary/tertiary payment typologies (51% and 84% missing — legitimately absent)
    for col in ("payment_typology_2", "payment_typology_3"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().fillna("None")

    if "payment_typology_1" in out.columns:
        out["payment_typology_1"] = (
            out["payment_typology_1"].astype("string").str.strip().fillna("Unknown")
        )

    # Birth weight: ~90.16% missing (adult/non-newborn discharges); convert UNKN -> NaN
    if "birth_weight" in out.columns:
        bw = pd.to_numeric(out["birth_weight"].astype("string").str.strip(), errors="coerce")
        out["birth_weight_grams"] = bw.astype("float32")
        out = out.drop(columns=["birth_weight"])

    return out


def validate_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Clean whitespace and standardize key categorical fields."""
    out = df.copy()
    str_cols = [
        "age_group",
        "gender",
        "race",
        "ethnicity",
        "type_of_admission",
        "patient_disposition",
        "apr_drg_code",
        "apr_drg_description",
        "apr_mdc_code",
        "apr_mdc_description",
        "apr_severity_code",
        "apr_medical_surgical_description",
        "emergency_department_indicator",
    ]
    for col in str_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    if "discharge_year" in out.columns:
        out["discharge_year"] = (
            pd.to_numeric(out["discharge_year"], errors="coerce").fillna(2021).astype("int16")
        )

    return out


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric and categorical columns to minimize RAM usage."""
    out = df.copy()
    categorical_candidates = [
        "hospital_service_area",
        "hospital_county",
        "operating_certificate_number",
        "permanent_facility_id",
        "facility_name",
        "age_group",
        "zip_code_3",
        "gender",
        "race",
        "ethnicity",
        "type_of_admission",
        "patient_disposition",
        "ccsr_diagnosis_code",
        "ccsr_diagnosis_description",
        "ccsr_procedure_code",
        "ccsr_procedure_description",
        "apr_drg_code",
        "apr_drg_description",
        "apr_mdc_code",
        "apr_mdc_description",
        "apr_severity_code",
        "apr_severity_description",
        "apr_risk_of_mortality",
        "apr_medical_surgical_description",
        "payment_typology_1",
        "payment_typology_2",
        "payment_typology_3",
        "emergency_department_indicator",
    ]
    for col in categorical_candidates:
        if col in out.columns:
            if col == "age_group":
                cats = [c for c in AGE_GROUP_ORDER if c in set(out[col].dropna())]
                extras = sorted(set(out[col].dropna()) - set(cats))
                out[col] = pd.Categorical(out[col], categories=cats + extras, ordered=True)
            elif col == "apr_severity_description":
                cats = [c for c in SEVERITY_ORDER if c in set(out[col].dropna())]
                extras = sorted(set(out[col].dropna()) - set(cats))
                out[col] = pd.Categorical(out[col], categories=cats + extras, ordered=True)
            elif col == "apr_risk_of_mortality":
                cats = [c for c in RISK_ORDER if c in set(out[col].dropna())]
                extras = sorted(set(out[col].dropna()) - set(cats))
                out[col] = pd.Categorical(out[col], categories=cats + extras, ordered=True)
            else:
                out[col] = out[col].astype("category")

    return out


def clean_dataframe(
    df: pd.DataFrame,
    deduplicate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full cleaning pipeline on a raw DataFrame and return (cleaned_df, audit_dict)."""
    initial_rows = len(df)
    out = standardize_columns(df)

    # Exact duplicate removal before imputation so original duplicates are caught accurately
    duplicates_removed = 0
    if deduplicate and initial_rows > 0:
        dup_mask = out.duplicated(keep="first")
        duplicates_removed = int(dup_mask.sum())
        if duplicates_removed > 0:
            out = out.loc[~dup_mask].reset_index(drop=True)

    # Parse Length of Stay
    los_num, is_topcoded = parse_length_of_stay(out["length_of_stay"])
    out["length_of_stay"] = los_num
    out["is_los_topcoded"] = is_topcoded

    # Parse monetary columns
    out["total_charges"] = parse_monetary_series(out["total_charges"])
    out["total_costs"] = parse_monetary_series(out["total_costs"])

    # Drop rows with invalid/missing core numeric targets or negative/zero financial entries
    valid_mask = (
        out["length_of_stay"].notna()
        & (out["length_of_stay"] > 0)
        & out["total_charges"].notna()
        & (out["total_charges"] > 0)
        & out["total_costs"].notna()
        & (out["total_costs"] > 0)
    )
    invalid_numeric_dropped = int((~valid_mask).sum())
    if invalid_numeric_dropped > 0:
        out = out.loc[valid_mask].reset_index(drop=True)

    out = handle_missing_values(out)
    out = validate_categoricals(out)

    audit = {
        "initial_rows": initial_rows,
        "exact_duplicates_removed": duplicates_removed,
        "invalid_numeric_rows_dropped": invalid_numeric_dropped,
        "final_rows": len(out),
        "los_topcoded_120plus_count": int(out["is_los_topcoded"].sum()) if len(out) else 0,
    }
    return out, audit


def run_full_preprocessing(
    csv_path: Path | str | None = None,
    output_parquet: Path = PROCESSED_PARQUET_PATH,
    deduplicate: bool = True,
) -> dict[str, Any]:
    """Stream-process the full SPARCS CSV, deduplicate globally, engineer core features, and save Parquet."""
    from src.feature_engineering import add_engineered_features

    t0 = time.time()
    resolved = Path(csv_path) if csv_path is not None else resolve_raw_csv_path()
    if resolved is None or not resolved.exists():
        raise FileNotFoundError("Raw CSV file not found.")

    cleaned_chunks: list[pd.DataFrame] = []
    seen_hashes: set[int] = set()
    initial_rows = 0
    duplicates_removed = 0
    invalid_numeric_dropped = 0

    for idx, chunk in enumerate(iter_raw_chunks(resolved, chunksize=250_000), start=1):
        initial_rows += len(chunk)
        std_chunk = standardize_columns(chunk)

        if deduplicate:
            # Global deduplication across chunks using 64-bit row hash + verification within chunk
            row_hashes = pd.util.hash_pandas_object(std_chunk, index=False).values
            keep_mask = np.ones(len(std_chunk), dtype=bool)
            for i, h in enumerate(row_hashes):
                h_int = int(h)
                if h_int in seen_hashes:
                    keep_mask[i] = False
                else:
                    seen_hashes.add(h_int)
            duplicates_removed += int((~keep_mask).sum())
            std_chunk = std_chunk.loc[keep_mask].reset_index(drop=True)

        cleaned_chunk, chunk_audit = clean_dataframe(std_chunk, deduplicate=False)
        invalid_numeric_dropped += chunk_audit["invalid_numeric_rows_dropped"]
        cleaned_chunk = add_engineered_features(cleaned_chunk)
        cleaned_chunks.append(cleaned_chunk)
        print(f"[preprocessing] Chunk {idx}: accumulated {sum(len(c) for c in cleaned_chunks):,} rows")

    full_df = pd.concat(cleaned_chunks, ignore_index=True)
    del cleaned_chunks
    # Recompute global IQR outlier flag accurately across the full dataset
    q1 = float(full_df["length_of_stay"].quantile(0.25))
    q3 = float(full_df["length_of_stay"].quantile(0.75))
    upper_fence = q3 + 1.5 * (q3 - q1)
    full_df["los_iqr_outlier"] = (full_df["length_of_stay"] > upper_fence).astype("bool")
    full_df = optimize_dtypes(full_df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(output_parquet, index=False, compression="snappy")

    audit = {
        "source_file": str(resolved.name),
        "initial_rows": initial_rows,
        "exact_duplicates_removed": duplicates_removed,
        "invalid_numeric_rows_dropped": invalid_numeric_dropped,
        "final_rows": int(len(full_df)),
        "los_topcoded_120plus_count": int(full_df["is_los_topcoded"].sum()),
        "newborn_admissions_count": int(full_df["is_newborn"].sum()),
        "los_iqr_outlier_count": int(full_df["los_iqr_outlier"].sum()),
        "parquet_size_mb": round(output_parquet.stat().st_size / (1024 * 1024), 2),
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = REPORTS_DIR / "preprocessing_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"[preprocessing] Saved {output_parquet} ({audit['parquet_size_mb']} MB) in {audit['elapsed_seconds']}s")
    return audit


if __name__ == "__main__":
    run_full_preprocessing()
