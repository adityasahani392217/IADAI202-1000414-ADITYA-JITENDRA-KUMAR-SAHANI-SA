"""Page 5: Payment Analytics (RQ5)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

if not st.session_state.get("_nav_shell_active"):
    st.set_page_config(page_title="Payment Analytics | MediScope AI", page_icon="assets/mediscope_icon.svg", layout="wide")

from src.config import PALETTE
from src.dashboard_state import get_page_context, get_preaggregated_table
from src.eda import summarize_payment
from src.utils import (
    inject_custom_css,
    render_active_filter_banner,
    render_page_header,
    render_plain_english_guide,
)
from src.visualizations import (
    _apply_mediscope_layout,
    plot_charges_by_payment,
    plot_payment_type_pie,
)


def main() -> None:
    if not st.session_state.get("_nav_shell_active"):
        inject_custom_css()

    filtered_df, filters_active, _, filter_summary = get_page_context()

    render_page_header(
        "Payment Analytics (RQ5)",
        "Primary payer mix, average hospital charges, length of stay, and secondary coverage patterns.",
    )
    render_active_filter_banner(filter_summary)

    if filtered_df.empty:
        st.warning("No discharge records match the selected filters.")
        return

    render_plain_english_guide(
        "Understanding Payer Categories (No Double-Counting)",
        "<code>Payment Typology 1</code> is each patient's <strong>primary insurance</strong> so shares always add up to 100%. "
        "Public programs (<strong>Medicare 39.4% + Medicaid 30.7%</strong>) cover ~70% of all stays. Secondary insurance "
        "(<code>Payment Typology 2</code>) is shown separately on the bottom-right chart.",
    )

    if not filters_active:
        pay_df = get_preaggregated_table("agg_payment.parquet")
        if pay_df is None:
            pay_df = summarize_payment(filtered_df)
    else:
        pay_df = summarize_payment(filtered_df)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_payment_type_pie(pay_df), width="stretch")
    with c2:
        st.plotly_chart(plot_charges_by_payment(pay_df), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        ordered_cnt = pay_df.sort_values("discharges", ascending=True)
        fig_cnt = px.bar(
            ordered_cnt,
            x="discharges",
            y="payment_typology_1",
            orientation="h",
            color_discrete_sequence=[PALETTE["teal"]],
        )
        _apply_mediscope_layout(
            fig_cnt,
            "Discharge Counts by Primary Payment Category",
            xaxis_title="Discharges",
            yaxis_title="Primary Payer",
            height=410,
        )
        st.plotly_chart(fig_cnt, width="stretch")

    with c4:
        sec_counts = filtered_df["payment_typology_2"].value_counts().reset_index()
        sec_counts.columns = ["secondary_payer", "discharges"]
        sec_counts["secondary_payer"] = sec_counts["secondary_payer"].astype(str)
        fig_sec = px.bar(
            sec_counts.head(10),
            x="secondary_payer",
            y="discharges",
            color_discrete_sequence=[PALETTE["navy"]],
        )
        _apply_mediscope_layout(
            fig_sec,
            "Secondary Payment Field (Payment Typology 2)",
            xaxis_title="Secondary Payer ('None' = Single-Payer Stay)",
            yaxis_title="Discharges",
            height=410,
        )
        st.plotly_chart(fig_sec, width="stretch")

    st.markdown("#### Primary Payment Category Metrics")
    st.dataframe(pay_df.round(2), width="stretch")
    st.download_button(
        "Download Payment Summary (CSV)",
        data=pay_df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_payment_summary.csv",
        mime="text/csv",
    )


main()
