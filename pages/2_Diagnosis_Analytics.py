"""Page 2: Diagnosis Analytics (RQ1)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

if not st.session_state.get("_nav_shell_active"):
    st.set_page_config(page_title="Diagnosis Analytics | MediScope AI", page_icon="assets/mediscope_icon.svg", layout="wide")

from src.config import PALETTE
from src.dashboard_state import get_page_context, get_preaggregated_table
from src.eda import summarize_diagnosis
from src.utils import (
    inject_custom_css,
    render_active_filter_banner,
    render_page_header,
    render_plain_english_guide,
)
from src.visualizations import (
    _apply_mediscope_layout,
    plot_avg_stay_by_diagnosis,
    plot_top_volume_diagnoses,
)


def main() -> None:
    if not st.session_state.get("_nav_shell_active"):
        inject_custom_css()

    filtered_df, filters_active, mode, filter_summary = get_page_context()

    render_page_header(
        "Diagnosis Analytics (RQ1)",
        "Which medical diagnoses lead to the longest hospital stays, highest volume, and highest charges?",
    )
    render_active_filter_banner(filter_summary)

    if filtered_df.empty:
        st.warning("No discharge records match the selected filters.")
        return

    col_ctl1, col_ctl2, col_ctl3 = st.columns([1, 1, 2])
    with col_ctl1:
        top_n = st.slider("Top N Diagnoses", min_value=5, max_value=30, value=15, step=1)
    with col_ctl2:
        default_min_n = 500 if (not filters_active and mode == "full_local") else 25
        min_discharges = st.number_input(
            "Min. Discharges Threshold",
            min_value=1,
            max_value=5000,
            value=default_min_n,
            step=25,
            help="Hides rare 1-patient codes so average stay rankings are clinically reliable.",
        )
    with col_ctl3:
        mdc_list = ["All Categories"] + sorted(
            filtered_df["apr_mdc_description"].dropna().astype(str).unique().tolist()
        )
        sel_mdc_local = st.selectbox("Filter by Major Diagnostic Category (MDC)", mdc_list)

    work_df = filtered_df
    if sel_mdc_local != "All Categories":
        work_df = filtered_df[filtered_df["apr_mdc_description"] == sel_mdc_local]

    if not filters_active and sel_mdc_local == "All Categories":
        agg_diag = get_preaggregated_table("agg_diagnosis.parquet")
        if agg_diag is not None:
            diag_df = agg_diag[agg_diag["discharges"] >= min_discharges].copy()
        else:
            diag_df = summarize_diagnosis(work_df, min_discharges=int(min_discharges))
    else:
        diag_df = summarize_diagnosis(work_df, min_discharges=int(min_discharges))

    if diag_df.empty:
        st.warning("No diagnoses meet the minimum discharge threshold for your filter selection. Lower the threshold.")
        return

    render_plain_english_guide(
        "Key Clinical Takeaway",
        "Blood cancers (Acute Myeloid & Lymphoblastic Leukemia), stroke/brain injury sequelae, and severe psychiatric conditions "
        "(Schizophrenia Spectrum Disorders) require the longest inpatient stays (14–23 days on average). Meanwhile, childbirth (Liveborn), "
        "Septicemia, and COVID-19 account for the largest total number of hospital admissions.",
    )

    st.plotly_chart(plot_avg_stay_by_diagnosis(diag_df, top_n=top_n), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_top_volume_diagnoses(diag_df, top_n=min(top_n, 15)), width="stretch")
    with c2:
        top_chg = diag_df.sort_values("avg_charges", ascending=False).head(min(top_n, 15)).copy()
        top_chg = top_chg.sort_values("avg_charges", ascending=True)
        top_chg["label"] = (
            top_chg["ccsr_diagnosis_code"].astype(str)
            + " — "
            + top_chg["ccsr_diagnosis_description"].astype(str).str.slice(0, 34)
        )
        fig_chg = px.bar(
            top_chg,
            x="avg_charges",
            y="label",
            orientation="h",
            color_discrete_sequence=[PALETTE["teal"]],
        )
        fig_chg.update_xaxes(tickprefix="$")
        _apply_mediscope_layout(
            fig_chg,
            f"Top {min(top_n, 15)} Diagnoses by Average Hospital Charges",
            xaxis_title="Average Total Charges (USD)",
            yaxis_title="CCSR Diagnosis",
            height=max(400, min(top_n, 15) * 28),
        )
        st.plotly_chart(fig_chg, width="stretch")

    top10_codes = (
        diag_df.sort_values("discharges", ascending=False)
        .head(10)["ccsr_diagnosis_code"]
        .astype(str)
        .tolist()
    )
    sev_diag = (
        work_df[work_df["ccsr_diagnosis_code"].isin(top10_codes)]
        .groupby(["ccsr_diagnosis_code", "apr_severity_description"], observed=True)
        .size()
        .reset_index(name="discharges")
    )
    fig_sev = px.bar(
        sev_diag,
        x="ccsr_diagnosis_code",
        y="discharges",
        color="apr_severity_description",
        barmode="stack",
        color_discrete_sequence=[PALETTE["light_teal"], PALETTE["teal"], "#3B82F6", PALETTE["navy"], PALETTE["grey"]],
    )
    _apply_mediscope_layout(
        fig_sev,
        "Severity Mix Across Top 10 High-Volume Diagnoses",
        xaxis_title="CCSR Diagnosis Code",
        yaxis_title="Discharges",
        height=400,
    )
    st.plotly_chart(fig_sev, width="stretch")

    st.markdown("#### Diagnosis Summary Table")
    st.dataframe(diag_df.head(100).round(2), width="stretch")
    st.download_button(
        "Download Diagnosis Summary (CSV)",
        data=diag_df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_diagnosis_summary.csv",
        mime="text/csv",
    )


main()
