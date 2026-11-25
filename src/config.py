"""Central configuration, design tokens, paths, and schema definitions for MediScope AI."""

from __future__ import annotations

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AGGREGATES_DIR = DATA_DIR / "aggregates"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

# Primary raw dataset filename
DEFAULT_RAW_FILENAME = (
    "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"
)

# Standardised processed parquet path (full dataset)
PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "discharges_cleaned.parquet"

# Compact cloud-ready artifacts (safe for GitHub & Streamlit Community Cloud)
CLOUD_SAMPLE_PARQUET_PATH = AGGREGATES_DIR / "discharges_sample.parquet"
CLOUD_SUMMARY_JSON_PATH = AGGREGATES_DIR / "eda_summary.json"
CLOUD_DIAGNOSIS_AGG_PATH = AGGREGATES_DIR / "agg_diagnosis.parquet"
CLOUD_FACILITY_COUNTY_AGG_PATH = AGGREGATES_DIR / "agg_facility_county.parquet"
CLOUD_SEVERITY_AGG_PATH = AGGREGATES_DIR / "agg_severity.parquet"
CLOUD_PAYMENT_AGG_PATH = AGGREGATES_DIR / "agg_payment.parquet"
CLOUD_AGE_GENDER_AGG_PATH = AGGREGATES_DIR / "agg_age_gender.parquet"
CLOUD_LOS_HIST_AGG_PATH = AGGREGATES_DIR / "agg_los_histogram.parquet"

# Model artifacts
MODEL_PIPELINE_PATH = MODELS_DIR / "los_model_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# LogiSight / MediScope AI Design System Tokens
PALETTE = {
    "teal": "#0D9488",
    "teal_dark": "#0F766E",
    "navy": "#10234B",
    "navy_light": "#1E3A6E",
    "light_teal": "#E6F5F4",
    "white": "#FFFFFF",
    "grey": "#64748B",
    "bg_neutral": "#F8FAFC",
    "border": "#E2E8F0",
    "text_primary": "#0F172A",
    "amber": "#D97706",
    "coral": "#E11D48",
}

CHART_COLORWAY = [
    "#0D9488",  # Teal
    "#10234B",  # Navy
    "#2DD4BF",  # Mint Teal
    "#3B82F6",  # Clinical Blue
    "#64748B",  # Slate Grey
    "#0F766E",  # Deep Teal
    "#F59E0B",  # Amber
    "#E11D48",  # Rose
]

# Column standardization mapping (Original SPARCS CSV -> snake_case)
COLUMN_RENAME_MAP = {
    "Hospital Service Area": "hospital_service_area",
    "Hospital County": "hospital_county",
    "Operating Certificate Number": "operating_certificate_number",
    "Permanent Facility Id": "permanent_facility_id",
    "Facility Name": "facility_name",
    "Age Group": "age_group",
    "Zip Code - 3 digits": "zip_code_3",
    "Gender": "gender",
    "Race": "race",
    "Ethnicity": "ethnicity",
    "Length of Stay": "length_of_stay",
    "Type of Admission": "type_of_admission",
    "Patient Disposition": "patient_disposition",
    "Discharge Year": "discharge_year",
    "CCSR Diagnosis Code": "ccsr_diagnosis_code",
    "CCSR Diagnosis Description": "ccsr_diagnosis_description",
    "CCSR Procedure Code": "ccsr_procedure_code",
    "CCSR Procedure Description": "ccsr_procedure_description",
    "APR DRG Code": "apr_drg_code",
    "APR DRG Description": "apr_drg_description",
    "APR MDC Code": "apr_mdc_code",
    "APR MDC Description": "apr_mdc_description",
    "APR Severity of Illness Code": "apr_severity_code",
    "APR Severity of Illness Description": "apr_severity_description",
    "APR Risk of Mortality": "apr_risk_of_mortality",
    "APR Medical Surgical Description": "apr_medical_surgical_description",
    "Payment Typology 1": "payment_typology_1",
    "Payment Typology 2": "payment_typology_2",
    "Payment Typology 3": "payment_typology_3",
    "Birth Weight": "birth_weight",
    "Emergency Department Indicator": "emergency_department_indicator",
    "Total Charges": "total_charges",
    "Total Costs": "total_costs",
}

# Ordered categorical definitions
AGE_GROUP_ORDER = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"]
SEVERITY_ORDER = ["Minor", "Moderate", "Major", "Extreme", "Not Applicable"]
RISK_ORDER = ["Minor", "Moderate", "Major", "Extreme", "Not Applicable"]

# Machine Learning feature configuration (strictly admission/triage-time safe vs documented post-grouping)
# Note: Total Charges, Total Costs, Patient Disposition, and Procedures are strictly EXCLUDED to prevent target leakage.
ML_FEATURES = [
    "age_group",
    "gender",
    "type_of_admission",
    "emergency_department_indicator",
    "ccsr_diagnosis_code",
    "apr_mdc_code",
    "apr_medical_surgical_description",
    "apr_severity_code",
    "apr_risk_of_mortality",
]
ML_TARGET = "length_of_stay"

# Protected attributes excluded from predictive features (used only for fairness audit)
FAIRNESS_ATTRIBUTES = ["race", "ethnicity", "gender", "age_group"]


def resolve_raw_csv_path() -> Path | None:
    """Locate the raw SPARCS CSV file across env var, data/raw, data/, and project root."""
    env_path = os.environ.get("MEDISCOPE_DATA_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists() and p.is_file():
            return p

    candidates = [
        RAW_DATA_DIR / DEFAULT_RAW_FILENAME,
        DATA_DIR / DEFAULT_RAW_FILENAME,
        PROJECT_ROOT / DEFAULT_RAW_FILENAME,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Fallback: check any CSV matching SPARCS pattern in raw or root
    for search_dir in (RAW_DATA_DIR, PROJECT_ROOT):
        if search_dir.exists():
            csvs = sorted(search_dir.glob("*SPARCS*.csv"))
            if csvs:
                return csvs[0]
    return None
