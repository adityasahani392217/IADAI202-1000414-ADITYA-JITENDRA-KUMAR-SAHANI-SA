"""Shared cached data loading, global sidebar filtering, and aggregate resolution."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.config import (
    AGE_GROUP_ORDER,
    CLOUD_SUMMARY_JSON_PATH,
    SEVERITY_ORDER,
)
from src.data_loader import load_aggregate_table, load_analysis_dataset

DASHBOARD_COLUMNS = [
    "hospital_service_area",
    "hospital_county",
    "permanent_facility_id",
    "facility_name",
    "age_group",
    "gender",
    "length_of_stay",
    "type_of_admission",
    "discharge_year",
    "ccsr_diagnosis_code",
    "ccsr_diagnosis_description",
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
    "total_charges",
    "total_costs",
    "is_ed_admission",
    "is_newborn",
    "payer_count",
    "cost_per_day",
    "charge_per_day",
    "charge_to_cost_ratio",
]


@st.cache_data(show_spinner="Loading MediScope AI inpatient dataset...")
def get_dashboard_dataframe() -> tuple[pd.DataFrame, str]:
    """Load and cache the cleaned dataset (full local parquet or cloud sample)."""
    df, mode = load_analysis_dataset(columns=DASHBOARD_COLUMNS, prefer_full=True)
    return df, mode


@st.cache_data(show_spinner=False)
def get_full_eda_summary() -> dict[str, Any] | None:
    """Load exact precomputed full-dataset EDA summary JSON if available."""
    if CLOUD_SUMMARY_JSON_PATH.exists():
        with open(CLOUD_SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def get_preaggregated_table(filename: str) -> pd.DataFrame | None:
    """Cached wrapper for loading pre-aggregated Parquet tables."""
    return load_aggregate_table(filename)


def _cat_options(series: pd.Series, ordered_ref: list[str] | None = None) -> list[str]:
    """Extract sorted unique values in O(1) for Categorical dtypes."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        opts = [str(c) for c in series.cat.categories]
    else:
        opts = [str(c) for c in series.dropna().unique().tolist()]

    if ordered_ref is not None:
        opt_set = set(opts)
        ordered = [x for x in ordered_ref if x in opt_set]
        extras = sorted(opt_set - set(ordered))
        return ordered + extras
    return sorted(opts)


def _reset_filters() -> None:
    """Clear all global sidebar filter keys in Streamlit session_state."""
    for key in (
        "flt_year",
        "flt_county",
        "flt_facility",
        "flt_age",
        "flt_gender",
        "flt_severity",
        "flt_mdc",
    ):
        if key in st.session_state:
            del st.session_state[key]


def apply_global_filters(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, bool]:
    """Render global sidebar filters and return (filtered_df, filters_active)."""
    with st.sidebar:
        st.markdown("### MediScope AI")
        st.caption("Hospital Inpatient Analytics & LOS Platform")

        if mode == "full_local":
            st.success(f"Full Local Dataset ({len(df):,} rows)")
        else:
            st.info(
                f"Cloud Mode: Pre-aggregated 2.09M metrics + {len(df):,}-row stratified sample"
            )

        st.divider()
        st.markdown("#### Global Filters")

        if st.button("Reset All Filters", width="stretch"):
            _reset_filters()
            st.rerun()

        years = sorted(df["discharge_year"].dropna().unique().tolist())
        sel_years = st.multiselect(
            "Discharge Year",
            options=years,
            default=years,
            key="flt_year",
            help="2021 SPARCS De-Identified Inpatient Discharges",
        )

        counties = _cat_options(df["hospital_county"])
        sel_counties = st.multiselect(
            "Hospital County",
            options=counties,
            default=[],
            key="flt_county",
            placeholder="All counties",
        )

        if sel_counties:
            fac_series = df.loc[df["hospital_county"].isin(sel_counties), "facility_name"]
            facilities = sorted(fac_series.dropna().astype(str).unique().tolist())
        else:
            facilities = _cat_options(df["facility_name"])

        sel_facilities = st.multiselect(
            "Facility Name",
            options=facilities,
            default=[],
            key="flt_facility",
            placeholder="All facilities",
        )

        age_opts = _cat_options(df["age_group"], ordered_ref=AGE_GROUP_ORDER)
        sel_ages = st.multiselect(
            "Age Group",
            options=age_opts,
            default=[],
            key="flt_age",
            placeholder="All age groups",
        )

        gender_opts = _cat_options(df["gender"])
        sel_genders = st.multiselect(
            "Gender",
            options=gender_opts,
            default=[],
            key="flt_gender",
            placeholder="All genders",
        )

        sev_opts = _cat_options(df["apr_severity_description"], ordered_ref=SEVERITY_ORDER)
        sel_sevs = st.multiselect(
            "Illness Severity",
            options=sev_opts,
            default=[],
            key="flt_severity",
            placeholder="All severity levels",
        )

        mdc_opts = _cat_options(df["apr_mdc_description"])
        sel_mdcs = st.multiselect(
            "Diagnosis Category (MDC)",
            options=mdc_opts,
            default=[],
            key="flt_mdc",
            placeholder="All diagnostic categories",
        )

    filters_active = bool(
        (sel_years and len(sel_years) < len(years))
        or sel_counties
        or sel_facilities
        or sel_ages
        or sel_genders
        or sel_sevs
        or sel_mdcs
    )

    if not filters_active:
        filtered = df
    else:
        mask = pd.Series(True, index=df.index)
        if sel_years:
            mask &= df["discharge_year"].isin(sel_years)
        if sel_counties:
            mask &= df["hospital_county"].isin(sel_counties)
        if sel_facilities:
            mask &= df["facility_name"].isin(sel_facilities)
        if sel_ages:
            mask &= df["age_group"].isin(sel_ages)
        if sel_genders:
            mask &= df["gender"].isin(sel_genders)
        if sel_sevs:
            mask &= df["apr_severity_description"].isin(sel_sevs)
        if sel_mdcs:
            mask &= df["apr_mdc_description"].isin(sel_mdcs)
        filtered = df.loc[mask]

    with st.sidebar:
        st.divider()
        st.caption(f"Active slice: **{len(filtered):,}** records")
        if not filtered.empty:
            export_cap = min(len(filtered), 10_000)
            csv_bytes = filtered.head(export_cap).to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download Filtered CSV ({export_cap:,} rows)",
                data=csv_bytes,
                file_name="mediscope_filtered_discharges.csv",
                mime="text/csv",
                width="stretch",
            )

    return filtered, filters_active
