"""Page 1: Executive Overview."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Overview | MediScope AI", page_icon="📊", layout="wide")

from src.config import PALETTE
from src.dashboard_state import (
    apply_global_filters,
    get_dashboard_dataframe,
    get_full_eda_summary,
)
from src.eda import compute_core_kpis
from src.utils import (
    format_currency,
    format_number,
    inject_custom_css,
    render_kpi_card,
    render_page_header,
)
from src.visualizations import _apply_mediscope_layout


def main() -> None:
    inject_custom_css()
    df, mode = get_dashboard_dataframe()
    filtered_df, filters_active = apply_global_filters(df, mode)

    render_page_header(
        "Executive Overview",
        "Statewide inpatient discharge volume, stay duration, cost structure, and clinical mix.",
        mode_label="Exact Full-Dataset KPIs" if not filters_active else "Filtered Slice",
    )

    if filtered_df.empty:
        st.warning("No discharge records match the selected filters. Reset filters in the sidebar.")
        return

    summary = get_full_eda_summary()
    kpis = summary["kpis"] if (not filters_active and summary) else compute_core_kpis(filtered_df)

    # Row 1 KPIs
    r1 = st.columns(4)
    with r1[0]:
        render_kpi_card("Total Discharge Records", format_number(kpis["total_discharges"]), "Cleaned inpatient stays")
    with r1[1]:
        render_kpi_card("Average Length of Stay", f"{kpis['avg_los']:.2f} days", f"Std Dev: {kpis['los_std']:.2f}d", navy=True)
    with r1[2]:
        render_kpi_card("Median Length of Stay", f"{kpis['median_los']:.1f} days", f"IQR: {kpis['los_iqr']:.1f} days")
    with r1[3]:
        render_kpi_card("Unique Facilities / Counties", f"{kpis['unique_facilities']} / {kpis['unique_counties']}", "Acute & specialty hospitals", navy=True)

    st.write("")
    # Row 2 KPIs
    r2 = st.columns(4)
    with r2[0]:
        render_kpi_card("Average Hospital Charges", format_currency(kpis["avg_charges"]), f"Median: {format_currency(kpis['median_charges'])}", navy=True)
    with r2[1]:
        render_kpi_card("Average Hospital Costs", format_currency(kpis["avg_costs"]), f"Median: {format_currency(kpis['median_costs'])}")
    with r2[2]:
        render_kpi_card("Avg Cost per Inpatient Day", format_currency(kpis["avg_cost_per_day"]), f"Charge-to-Cost: {kpis['avg_charge_to_cost']:.2f}x", navy=True)
    with r2[3]:
        render_kpi_card("ED / Surgical Share", f"{kpis['ed_pct']:.1f}% / {kpis['surgical_pct']:.1f}%", "ED-originated vs. surgical")

    # Executive Summary generated dynamically from computed statistics
    st.markdown(
        f"""
        <div class="insight-box">
            <strong>Executive Summary:</strong> Across <strong>{format_number(kpis['total_discharges'])}</strong>
            inpatient discharges spanning <strong>{kpis['unique_facilities']}</strong> facilities and
            <strong>{kpis['unique_counties']}</strong> counties, the mean length of stay is
            <strong>{kpis['avg_los']:.2f} days</strong> while the median is <strong>{kpis['median_los']:.1f} days</strong>—reflecting
            strong right-skewness from complex long-stay admissions. Average hospital charges stand at
            <strong>{format_currency(kpis['avg_charges'])}</strong> against average reported costs of
            <strong>{format_currency(kpis['avg_costs'])}</strong> (mean charge-to-cost ratio:
            <strong>{kpis['avg_charge_to_cost']:.2f}x</strong>). Emergency Department admissions account for
            <strong>{kpis['ed_pct']:.1f}%</strong> of inpatient stays, while surgical procedures account for
            <strong>{kpis['surgical_pct']:.1f}%</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Breakdown charts: Admission Type and Medical vs Surgical
    c1, c2 = st.columns(2)
    with c1:
        adm_counts = (
            filtered_df["type_of_admission"]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        adm_counts.columns = ["type_of_admission", "discharges"]
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
        st.plotly_chart(fig_adm, width='stretch')

    with c2:
        ms_counts = (
            filtered_df["apr_medical_surgical_description"]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        ms_counts.columns = ["care_type", "discharges"]
        fig_ms = px.pie(
            ms_counts,
            names="care_type",
            values="discharges",
            hole=0.45,
            color_discrete_sequence=[PALETTE["navy"], PALETTE["teal"], PALETTE["grey"]],
        )
        _apply_mediscope_layout(fig_ms, "Medical vs. Surgical Inpatient Mix")
        st.plotly_chart(fig_ms, width='stretch')

    # Downloadable summary snapshot
    st.markdown("#### Summary KPIs Table")
    kpi_df = pd.DataFrame([kpis])
    st.dataframe(kpi_df, width='stretch')
    st.download_button(
        "Download KPI Summary (CSV)",
        data=kpi_df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_overview_kpis.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
