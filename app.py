"""MediScope AI — Main Streamlit Application Shell with Persistent Navigation & Filters."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="MediScope AI | Hospital Inpatient Analytics",
    page_icon="assets/mediscope_icon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard_state import (
    apply_global_filters,
    get_dashboard_dataframe,
    get_full_eda_summary,
    get_page_context,
)
from src.eda import compute_core_kpis
from src.utils import (
    LOGO_ICON_PATH,
    LOGO_SIDEBAR_PATH,
    format_currency,
    format_number,
    inject_custom_css,
    render_active_filter_banner,
    render_kpi_card,
    render_page_header,
    render_plain_english_guide,
)


def render_home_page() -> None:
    """Render the MediScope AI landing hub and executive launchpad."""
    filtered_df, filters_active, mode, filter_summary = get_page_context()

    mode_label = (
        "Full Local Dataset (2.09M Records)"
        if mode == "full_local"
        else "Cloud Mode (2.09M Pre-Aggregated + 80K Sample)"
    )
    render_page_header(
        "MediScope AI — Inpatient Discharge & Length-of-Stay Intelligence",
        "New York State SPARCS De-Identified Inpatient Analytics Platform (2021)",
        mode_label=mode_label,
    )
    render_active_filter_banner(filter_summary)

    if filtered_df.empty:
        st.warning(
            "No discharge records match your active sidebar filters. Click **Reset All Filters** in the sidebar."
        )
        return

    summary = get_full_eda_summary()
    kpis = summary["kpis"] if (not filters_active and summary) else compute_core_kpis(filtered_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Total Inpatient Discharges",
            format_number(kpis["total_discharges"]),
            f"ED Admissions: {kpis['ed_pct']:.1f}%",
        )
    with c2:
        render_kpi_card(
            "Avg vs. Typical Stay (Mean / Median)",
            f"{kpis['avg_los']:.2f}d / {kpis['median_los']:.1f}d",
            f"Range: {int(kpis['min_los'])}–{int(kpis['max_los'])} days",
            navy=True,
        )
    with c3:
        render_kpi_card(
            "Average Hospital Charges",
            format_currency(kpis["avg_charges"]),
            f"Typical (Median): {format_currency(kpis['median_charges'])}",
        )
    with c4:
        render_kpi_card(
            "Hospitals & Counties",
            f"{kpis['unique_facilities']} / {kpis['unique_counties']}",
            f"Avg Cost/Day: {format_currency(kpis['avg_cost_per_day'])}",
            navy=True,
        )

    render_plain_english_guide(
        "Quick Guide to MediScope AI",
        "Use the persistent <strong>Global Filters</strong> on the left sidebar to slice data by county, hospital, age, gender, illness severity, or diagnosis category—your filters stay active across every analytics page. "
        "Note that <strong>Median</strong> represents the typical patient stay (3 days), while <strong>Mean</strong> (5.75 days) is higher because complex ICU, oncology, and behavioral health stays extend for weeks.",
    )

    st.markdown("### Explore Platform Modules")
    col_a, col_b = st.columns(2)
    with col_a:
        st.page_link("pages/1_Overview.py", label="1. Executive Overview — Key Hospital Metrics & Summary", icon=":material/bar_chart:")
        st.page_link("pages/2_Diagnosis_Analytics.py", label="2. Diagnosis Analytics — Longest Stays & Top Conditions", icon=":material/medical_services:")
        st.page_link("pages/3_Financial_Analytics.py", label="3. Financial Analytics — Charges by Severity, Age & Gender", icon=":material/payments:")
        st.page_link("pages/4_Hospital_Performance.py", label="4. Hospital Performance — Facility × County Stay Heatmap", icon=":material/local_hospital:")
    with col_b:
        st.page_link("pages/5_Payment_Analytics.py", label="5. Payment Analytics — Insurance Mix & Payer Costs", icon=":material/receipt_long:")
        st.page_link("pages/6_Length_of_Stay.py", label="6. Length of Stay — Stay Duration Curve & Severity Breakdown", icon=":material/schedule:")
        st.page_link("pages/7_ML_Performance.py", label="7. ML Performance — Model Accuracy, Residuals & Fairness", icon=":material/monitoring:")
        st.page_link("pages/8_LOS_Predictor.py", label="8. Length of Stay Predictor — Interactive Bed-Day Forecast", icon=":material/model_training:")

    st.markdown(
        """
        <div class="insight-box">
            <strong>Governance & Clinical Disclaimer:</strong> All analytics use de-identified
            2021 New York State SPARCS inpatient records. Predicted Length of Stay values support
            hospital bed capacity and staffing planning only—never individual clinical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()
    if LOGO_SIDEBAR_PATH.exists() and LOGO_ICON_PATH.exists():
        st.logo(str(LOGO_SIDEBAR_PATH), icon_image=str(LOGO_ICON_PATH))

    # Mark shell active so subpages reuse persistent filters and skip duplicate page_config
    st.session_state["_nav_shell_active"] = True

    df, mode = get_dashboard_dataframe()
    apply_global_filters(df, mode)

    pages = [
        st.Page(render_home_page, title="Home", icon=":material/home:", default=True),
        st.Page("pages/1_Overview.py", title="1. Executive Overview", icon=":material/bar_chart:"),
        st.Page("pages/2_Diagnosis_Analytics.py", title="2. Diagnosis Analytics", icon=":material/medical_services:"),
        st.Page("pages/3_Financial_Analytics.py", title="3. Financial Analytics", icon=":material/payments:"),
        st.Page("pages/4_Hospital_Performance.py", title="4. Hospital Performance", icon=":material/local_hospital:"),
        st.Page("pages/5_Payment_Analytics.py", title="5. Payment Analytics", icon=":material/receipt_long:"),
        st.Page("pages/6_Length_of_Stay.py", title="6. Length of Stay", icon=":material/schedule:"),
        st.Page("pages/7_ML_Performance.py", title="7. ML Performance", icon=":material/monitoring:"),
        st.Page("pages/8_LOS_Predictor.py", title="8. LOS Predictor", icon=":material/model_training:"),
    ]
    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
