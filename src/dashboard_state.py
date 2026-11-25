"""Shared cached data loading, persistent global sidebar filtering, and aggregate resolution."""

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

FILTER_STATE_KEYS = (
    "flt_year",
    "flt_county",
    "flt_facility",
    "flt_age",
    "flt_gender",
    "flt_severity",
    "flt_mdc",
)


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


def _init_persistent_filter_state(years: list[int]) -> None:
    """Restore persisted filter values into widget keys so page switches never drop selections."""
    saved = st.session_state.setdefault("_saved_filters", {})
    for key in FILTER_STATE_KEYS:
        if key not in st.session_state:
            if key == "flt_year":
                st.session_state[key] = saved.get(key, list(years))
            else:
                st.session_state[key] = list(saved.get(key, []))


def _persist_filter_state() -> None:
    """Sync current widget filter selections into non-widget session storage."""
    saved = st.session_state.setdefault("_saved_filters", {})
    for key in FILTER_STATE_KEYS:
        if key in st.session_state:
            saved[key] = list(st.session_state[key])


def _reset_filters() -> None:
    """Reset all global sidebar filter keys cleanly."""
    st.session_state["_saved_filters"] = {}
    for key in FILTER_STATE_KEYS:
        if key in st.session_state:
            del st.session_state[key]


def build_filter_summary_text(filtered_count: int) -> str | None:
    """Return a readable string describing active filters, or None if unfiltered."""
    parts: list[str] = []
    label_map = [
        ("flt_county", "County"),
        ("flt_facility", "Facility"),
        ("flt_age", "Age"),
        ("flt_gender", "Gender"),
        ("flt_severity", "Severity"),
        ("flt_mdc", "Category"),
    ]
    for key, label in label_map:
        vals = st.session_state.get(key) or []
        if vals:
            shown = ", ".join(str(v) for v in vals[:2])
            if len(vals) > 2:
                shown += f" +{len(vals) - 2} more"
            parts.append(f"{label}: {shown}")
    if not parts:
        return None
    return " | ".join(parts) + f" — Showing {filtered_count:,} records"


def apply_global_filters(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, bool]:
    """Render persistent global sidebar filters and return (filtered_df, filters_active)."""
    years = sorted(df["discharge_year"].dropna().unique().tolist())
    _init_persistent_filter_state(years)

    # Sanitize year selection against valid options
    st.session_state["flt_year"] = [
        y for y in (st.session_state.get("flt_year") or years) if y in years
    ] or list(years)

    with st.sidebar:
        if mode == "full_local":
            st.success(f"Full Local Data: {len(df):,} rows")
        else:
            st.info(f"Cloud Mode: 2.09M Aggregates + {len(df):,} Sample")

        st.markdown("#### Global Filters")

        if st.button("Reset All Filters", width="stretch", key="btn_reset_global_filters"):
            _reset_filters()
            st.rerun()

        sel_years = st.multiselect(
            "Discharge Year",
            options=years,
            key="flt_year",
            on_change=_persist_filter_state,
            help="2021 NYS SPARCS Inpatient Discharges",
        )

        counties = _cat_options(df["hospital_county"])
        st.session_state["flt_county"] = [
            c for c in st.session_state.get("flt_county", []) if c in counties
        ]
        sel_counties = st.multiselect(
            "Hospital County",
            options=counties,
            key="flt_county",
            on_change=_persist_filter_state,
            placeholder="All counties",
        )

        if sel_counties:
            fac_series = df.loc[df["hospital_county"].isin(sel_counties), "facility_name"]
            facilities = sorted(fac_series.dropna().astype(str).unique().tolist())
        else:
            facilities = _cat_options(df["facility_name"])

        # Sanitize facility selection when county filter narrows options
        st.session_state["flt_facility"] = [
            f for f in st.session_state.get("flt_facility", []) if f in facilities
        ]
        sel_facilities = st.multiselect(
            "Facility Name",
            options=facilities,
            key="flt_facility",
            on_change=_persist_filter_state,
            placeholder="All facilities",
        )

        age_opts = _cat_options(df["age_group"], ordered_ref=AGE_GROUP_ORDER)
        sel_ages = st.multiselect(
            "Age Group",
            options=age_opts,
            key="flt_age",
            on_change=_persist_filter_state,
            placeholder="All age groups",
        )

        gender_opts = _cat_options(df["gender"])
        sel_genders = st.multiselect(
            "Gender",
            options=gender_opts,
            key="flt_gender",
            on_change=_persist_filter_state,
            placeholder="All genders",
        )

        sev_opts = _cat_options(df["apr_severity_description"], ordered_ref=SEVERITY_ORDER)
        sel_sevs = st.multiselect(
            "Illness Severity",
            options=sev_opts,
            key="flt_severity",
            on_change=_persist_filter_state,
            placeholder="All severity levels",
        )

        mdc_opts = _cat_options(df["apr_mdc_description"])
        sel_mdcs = st.multiselect(
            "Diagnosis Category (MDC)",
            options=mdc_opts,
            key="flt_mdc",
            on_change=_persist_filter_state,
            placeholder="All diagnostic categories",
        )

        _persist_filter_state()

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
                key="btn_download_filtered_csv",
            )

    # Cache active slice in session_state for pages running inside st.navigation shell
    st.session_state["_active_filtered_df"] = filtered
    st.session_state["_active_filters_flag"] = filters_active
    st.session_state["_active_data_mode"] = mode
    st.session_state["_active_filter_summary"] = (
        build_filter_summary_text(len(filtered)) if filters_active else None
    )
    return filtered, filters_active


def get_page_context() -> tuple[pd.DataFrame, bool, str, str | None]:
    """Return (filtered_df, filters_active, mode, filter_summary_text) safely in both shell and standalone modes."""
    if st.session_state.get("_nav_shell_active") and "_active_filtered_df" in st.session_state:
        return (
            st.session_state["_active_filtered_df"],
            bool(st.session_state.get("_active_filters_flag", False)),
            str(st.session_state.get("_active_data_mode", "cloud_sample")),
            st.session_state.get("_active_filter_summary"),
        )

    # Standalone page execution fallback (e.g., direct AppTest on pages/*.py)
    df, mode = get_dashboard_dataframe()
    filtered_df, filters_active = apply_global_filters(df, mode)
    summary_text = build_filter_summary_text(len(filtered_df)) if filters_active else None
    return filtered_df, filters_active, mode, summary_text
