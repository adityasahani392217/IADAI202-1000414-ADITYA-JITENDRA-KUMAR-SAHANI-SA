"""Page 1: Executive Overview."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

if not st.session_state.get("_nav_shell_active"):
    st.set_page_config(page_title="Overview | MediScope AI", page_icon="assets/mediscope_icon.svg", layout="wide")

from src.config import PALETTE
from src.dashboard_state import get_full_eda_summary, get_page_context
from src.eda import compute_core_kpis
from src.utils import (
    format_currency,
    format_number,
    inject_custom_css,
    render_active_filter_banner,
    render_kpi_card,
    render_page_header,
    render_plain_english_guide,
)
from src.visualizations import _apply_mediscope_layout


def main() -> None:
    if not st.session_state.get("_nav_shell_active"):
        inject_custom_css()

    filtered_df, filters_active, _, filter_summary = get_page_context()

    render_page_header(
        "Executive Overview",
        "Statewide inpatient discharge volume, stay duration, cost structure, and clinical mix.",
        mode_label="Exact Full-Dataset KPIs" if not filters_active else "Filtered Slice",
    )
    render_active_filter_banner(filter_summary)

    if filtered_df.empty:
        st.warning("No discharge records match the selected filters. Click **Reset All Filters** in the sidebar.")
        return

    summary = get_full_eda_summary()
    kpis = summary["kpis"] if (not filters_active and summary) else compute_core_kpis(filtered_df)

    r1 = st.columns(4)
    with r1[0]:
        render_kpi_card("Total Discharge Records", format_number(kpis["total_discharges"]), "Cleaned inpatient stays")
    with r1[1]:
        render_kpi_card("Average Length of Stay", f"{kpis['avg_los']:.2f} days", f"Std Dev: {kpis['los_std']:.2f}d", navy=True)
    with r1[2]:
        render_kpi_card("Median Stay (Typical Patient)", f"{kpis['median_los']:.1f} days", f"Middle 50% span (IQR): {kpis['los_iqr']:.1f}d")
    with r1[3]:
        render_kpi_card("Unique Facilities / Counties", f"{kpis['unique_facilities']} / {kpis['unique_counties']}", "Acute & specialty hospitals", navy=True)

    st.write("")
    r2 = st.columns(4)
    with r2[0]:
        render_kpi_card("Average Hospital Charges", format_currency(kpis["avg_charges"]), f"Median bill: {format_currency(kpis['median_charges'])}", navy=True)
    with r2[1]:
        render_kpi_card("Average Hospital Costs", format_currency(kpis["avg_costs"]), f"Median cost: {format_currency(kpis['median_costs'])}")
    with r2[2]:
        render_kpi_card("Avg Cost per Inpatient Day", format_currency(kpis["avg_cost_per_day"]), f"Charge-to-Cost: {kpis['avg_charge_to_cost']:.2f}x", navy=True)
    with r2[3]:
        render_kpi_card("ED / Surgical Share", f"{kpis['ed_pct']:.1f}% / {kpis['surgical_pct']:.1f}%", "Emergency vs. surgical stays")

    render_plain_english_guide(
        "How to Read These Numbers",
        f"Across <strong>{format_number(kpis['total_discharges'])}</strong> inpatient stays, half of all patients go home within "
        f"<strong>{kpis['median_los']:.1f} days</strong> (median), while long complex hospitalizations pull the overall average to "
        f"<strong>{kpis['avg_los']:.2f} days</strong>. Roughly <strong>{kpis['ed_pct']:.0f}%</strong> of stays begin in the Emergency Department, "
        f"and hospitals report an average daily cost of <strong>{format_currency(kpis['avg_cost_per_day'])}</strong>.",
    )

    c1, c2 = st.columns(2)
    with c1:
        adm_counts = filtered_df["type_of_admission"].value_counts().reset_index()
        adm_counts.columns = ["type_of_admission", "discharges"]
        adm_counts["type_of_admission"] = adm_counts["type_of_admission"].astype(str)
        fig_adm = px.bar(
            adm_counts,
            x="type_of_admission",
            y="discharges",
            color_discrete_sequence=[PALETTE["teal"]],
        )
        _apply_mediscope_layout(
            fig_adm,
            "Discharges by Admission Type",
            xaxis_title="Admission Type",
            yaxis_title="Discharges",
        )
        st.plotly_chart(fig_adm, width="stretch")

    with c2:
        ms_counts = filtered_df["apr_medical_surgical_description"].value_counts().reset_index()
        ms_counts.columns = ["care_type", "discharges"]
        ms_counts["care_type"] = ms_counts["care_type"].astype(str)
        fig_ms = px.pie(
            ms_counts,
            names="care_type",
            values="discharges",
            hole=0.45,
            color_discrete_sequence=[PALETTE["navy"], PALETTE["teal"], PALETTE["grey"]],
        )
        _apply_mediscope_layout(fig_ms, "Medical vs. Surgical Inpatient Mix")
        st.plotly_chart(fig_ms, width="stretch")

    st.markdown("#### Summary KPIs Table")
    kpi_df = pd.DataFrame([kpis])
    st.dataframe(kpi_df, width="stretch")
    st.download_button(
        "Download KPI Summary (CSV)",
        data=kpi_df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_overview_kpis.csv",
        mime="text/csv",
    )


main()
