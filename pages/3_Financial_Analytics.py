"""Page 3: Financial Analytics (RQ2 & RQ3)."""

from __future__ import annotations

import streamlit as st

if not st.session_state.get("_nav_shell_active"):
    st.set_page_config(page_title="Financial Analytics | MediScope AI", page_icon="💳", layout="wide")

from src.dashboard_state import get_page_context, get_preaggregated_table
from src.eda import summarize_age_gender, summarize_payment, summarize_severity
from src.utils import (
    inject_custom_css,
    render_active_filter_banner,
    render_page_header,
    render_plain_english_guide,
)
from src.visualizations import (
    plot_charges_boxplot_by_severity,
    plot_charges_by_age_gender,
    plot_charges_by_payment,
    plot_charges_vs_costs_by_severity,
)


def main() -> None:
    if not st.session_state.get("_nav_shell_active"):
        inject_custom_css()

    filtered_df, filters_active, _, filter_summary = get_page_context()

    render_page_header(
        "Financial Analytics (RQ2 & RQ3)",
        "How illness severity, patient age, gender, and insurance type shape hospital charges and costs.",
    )
    render_active_filter_banner(filter_summary)

    if filtered_df.empty:
        st.warning("No discharge records match the selected filters.")
        return

    render_plain_english_guide(
        "How to Read the Severity Box Plot",
        "Each box shows the middle 50% of patient bills (from the 25th to 75th percentile), the center line marks the typical (median) charge, "
        "and dots above the whisker represent rare multi-million-dollar ICU or surgical outliers. Notice how Extreme severity bills are roughly <strong>5–6× higher</strong> than Minor cases.",
    )

    use_log = st.toggle(
        "Use Logarithmic Y-Axis for Box Plot (recommended so $1M+ outliers don't compress the boxes)",
        value=True,
    )

    st.plotly_chart(
        plot_charges_boxplot_by_severity(filtered_df, use_log_scale=use_log),
        width="stretch",
    )

    if not filters_active:
        sev_df = get_preaggregated_table("agg_severity.parquet")
        age_gender_df = get_preaggregated_table("agg_age_gender.parquet")
        pay_df = get_preaggregated_table("agg_payment.parquet")
    else:
        sev_df = None
        age_gender_df = None
        pay_df = None

    if sev_df is None:
        sev_df = summarize_severity(filtered_df)
    if age_gender_df is None:
        age_gender_df = summarize_age_gender(filtered_df)
    if pay_df is None:
        pay_df = summarize_payment(filtered_df)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_charges_vs_costs_by_severity(sev_df), width="stretch")
    with c2:
        st.plotly_chart(plot_charges_by_age_gender(age_gender_df), width="stretch")

    st.plotly_chart(plot_charges_by_payment(pay_df), width="stretch")

    st.markdown("#### Severity Financial Summary (Exact Quartiles & Means)")
    st.dataframe(sev_df.round(2), width="stretch")
    st.download_button(
        "Download Severity Financial Table (CSV)",
        data=sev_df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_severity_financials.csv",
        mime="text/csv",
    )


main()
