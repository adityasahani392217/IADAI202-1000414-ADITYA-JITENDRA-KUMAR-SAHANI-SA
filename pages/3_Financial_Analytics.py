"""Page 3: Financial Analytics (RQ2 & RQ3)."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Financial Analytics | MediScope AI", page_icon="💳", layout="wide")

from src.dashboard_state import (
    apply_global_filters,
    get_dashboard_dataframe,
    get_preaggregated_table,
)
from src.eda import summarize_age_gender, summarize_payment, summarize_severity
from src.utils import inject_custom_css, render_page_header
from src.visualizations import (
    plot_charges_boxplot_by_severity,
    plot_charges_by_age_gender,
    plot_charges_by_payment,
    plot_charges_vs_costs_by_severity,
)


def main() -> None:
    inject_custom_css()
    df, mode = get_dashboard_dataframe()
    filtered_df, filters_active = apply_global_filters(df, mode)

    render_page_header(
        "Financial Analytics (RQ2 & RQ3)",
        "Hospital charges and costs segmented by illness severity, age group, gender, and payer category.",
    )

    if filtered_df.empty:
        st.warning("No discharge records match the selected filters.")
        return

    use_log = st.toggle(
        "Use Logarithmic Y-Axis for Box Plot (recommended due to extreme $1M+ outliers)",
        value=True,
    )

    # Compulsory Chart 2: Box Plot of Total Charges segmented by Severity
    st.plotly_chart(
        plot_charges_boxplot_by_severity(filtered_df, use_log_scale=use_log),
        width='stretch',
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
        st.plotly_chart(plot_charges_vs_costs_by_severity(sev_df), width='stretch')
    with c2:
        st.plotly_chart(plot_charges_by_age_gender(age_gender_df), width='stretch')

    st.plotly_chart(plot_charges_by_payment(pay_df), width='stretch')

    st.markdown("#### Severity Financial Summary (Exact Quartiles & Means)")
    st.dataframe(sev_df.round(2), width='stretch')
    st.download_button(
        "Download Severity Financial Table (CSV)",
        data=sev_df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_severity_financials.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
