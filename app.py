"""MediScope AI — Main Streamlit Application Entry Point."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="MediScope AI | Hospital Inpatient Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def main() -> None:
    inject_custom_css()
    df, mode = get_dashboard_dataframe()
    filtered_df, filters_active = apply_global_filters(df, mode)

    mode_label = (
        "Full Local Dataset (2.09M Records)"
        if mode == "full_local"
        else "Cloud Ready (2.09M Pre-Aggregated + 80K Sample)"
    )
    render_page_header(
        "MediScope AI — Inpatient Discharge & Length-of-Stay Intelligence",
        "New York State SPARCS De-Identified Inpatient Analytics Platform (2021)",
        mode_label=mode_label,
    )

    if filtered_df.empty:
        st.warning(
            "No discharge records match your active sidebar filters. Click **Reset All Filters** in the sidebar."
        )
        return

    summary = get_full_eda_summary()
    if not filters_active and summary:
        kpis = summary["kpis"]
    else:
        kpis = compute_core_kpis(filtered_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Total Inpatient Discharges",
            format_number(kpis["total_discharges"]),
            f"ED Admissions: {kpis['ed_pct']:.1f}%",
        )
    with c2:
        render_kpi_card(
            "Length of Stay (Mean / Median)",
            f"{kpis['avg_los']:.2f}d / {kpis['median_los']:.1f}d",
            f"Range: {int(kpis['min_los'])}–{int(kpis['max_los'])} days",
            navy=True,
        )
    with c3:
        render_kpi_card(
            "Average Hospital Charges",
            format_currency(kpis["avg_charges"]),
            f"Median: {format_currency(kpis['median_charges'])}",
        )
    with c4:
        render_kpi_card(
            "Facilities & Counties",
            f"{kpis['unique_facilities']} / {kpis['unique_counties']}",
            f"Avg Cost/Day: {format_currency(kpis['avg_cost_per_day'])}",
            navy=True,
        )

    st.markdown("### Platform Modules")
    st.markdown(
        "Navigate across MediScope AI's eight analytical and machine-learning workspaces using the sidebar or the quick links below:"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.page_link("pages/1_Overview.py", label="1. Executive Overview — KPIs & Executive Summary", icon="📊")
        st.page_link("pages/2_Diagnosis_Analytics.py", label="2. Diagnosis Analytics — Stay Duration & Volume by CCSR", icon="🩺")
        st.page_link("pages/3_Financial_Analytics.py", label="3. Financial Analytics — Severity Box Plots, Costs & Demographics", icon="💳")
        st.page_link("pages/4_Hospital_Performance.py", label="4. Hospital Performance — Facility × County Heatmap", icon="🏥")
    with col_b:
        st.page_link("pages/5_Payment_Analytics.py", label="5. Payment Analytics — Payer Mix Pie Chart & Comparisons", icon="🧾")
        st.page_link("pages/6_Length_of_Stay.py", label="6. Length of Stay — Distribution & Skewness Analysis", icon="⏱️")
        st.page_link("pages/7_ML_Performance.py", label="7. ML Performance — Model Benchmarks, Residuals & Fairness", icon="📈")
        st.page_link("pages/8_LOS_Predictor.py", label="8. Length of Stay Predictor — Interactive Patient Forecast", icon="🤖")

    st.markdown(
        """
        <div class="insight-box">
            <strong>Governance & Clinical Disclaimer:</strong> All analytics are derived from de-identified
            2021 New York State SPARCS inpatient discharge records. Predictive Length of Stay estimates are
            provided solely for operational capacity planning and resource forecasting—not for individual
            clinical diagnosis or treatment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
