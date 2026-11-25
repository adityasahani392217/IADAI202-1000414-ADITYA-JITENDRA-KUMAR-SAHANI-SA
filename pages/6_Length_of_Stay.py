"""Page 6: Length of Stay Distribution & Acuity Analysis."""

from __future__ import annotations

import streamlit as st

if not st.session_state.get("_nav_shell_active"):
    st.set_page_config(page_title="Length of Stay | MediScope AI", page_icon="⏱️", layout="wide")

from src.dashboard_state import (
    get_full_eda_summary,
    get_page_context,
    get_preaggregated_table,
)
from src.eda import compute_core_kpis, summarize_severity
from src.utils import (
    inject_custom_css,
    render_active_filter_banner,
    render_kpi_card,
    render_page_header,
    render_plain_english_guide,
)
from src.visualizations import plot_los_by_severity, plot_los_histogram


def main() -> None:
    if not st.session_state.get("_nav_shell_active"):
        inject_custom_css()

    filtered_df, filters_active, _, filter_summary = get_page_context()

    render_page_header(
        "Length of Stay Analytics",
        "Stay-duration curve, right-skewness, typical vs. average stay gaps, and illness severity breakdown.",
    )
    render_active_filter_banner(filter_summary)

    if filtered_df.empty:
        st.warning("No discharge records match the selected filters.")
        return

    summary = get_full_eda_summary()
    kpis = summary["kpis"] if (not filters_active and summary) else compute_core_kpis(filtered_df)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Mean Length of Stay", f"{kpis['avg_los']:.2f} days", f"Std Dev: {kpis['los_std']:.2f}d")
    with k2:
        render_kpi_card("Median Stay (Typical Patient)", f"{kpis['median_los']:.1f} days", f"IQR: {kpis['los_iqr']:.1f}d", navy=True)
    with k3:
        render_kpi_card("Valid Stay Range", f"{int(kpis['min_los'])}–{int(kpis['max_los'])} days", "120 = SPARCS 120+ top-code")
    with k4:
        skew_val = summary["skewness"]["length_of_stay"] if (not filters_active and summary) else float(filtered_df["length_of_stay"].skew())
        render_kpi_card("LOS Skewness", f"{skew_val:.2f}", "Strongly right-skewed", navy=True)

    render_plain_english_guide(
        "Why Both Mean and Median Matter for Bed Planning",
        "Most patients stay only <strong>2 to 4 days</strong> (the tall bars on the left, median = 3.0 days). However, the long right tail "
        "of severe illnesses pulls the arithmetic mean up to <strong>5.75 days</strong>. Use the slider below to zoom in or inspect the full 120-day range.",
    )

    max_view = st.slider(
        "Histogram Display Window (Max Days Shown)",
        min_value=14,
        max_value=120,
        value=45,
        step=5,
        help="Adjust horizontal window to inspect core stays (1–45d) or full range up to 120+ days.",
    )

    los_hist_df = None if filters_active else get_preaggregated_table("agg_los_histogram.parquet")
    st.plotly_chart(
        plot_los_histogram(los_hist_df=los_hist_df, df=filtered_df, max_los_view=max_view),
        width="stretch",
    )

    sev_df = None if filters_active else get_preaggregated_table("agg_severity.parquet")
    if sev_df is None:
        sev_df = summarize_severity(filtered_df)

    st.plotly_chart(plot_los_by_severity(sev_df), width="stretch")


main()
