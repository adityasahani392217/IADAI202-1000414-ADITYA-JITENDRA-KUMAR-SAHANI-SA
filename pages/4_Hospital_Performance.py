"""Page 4: Hospital Performance (RQ4)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

if not st.session_state.get("_nav_shell_active"):
    st.set_page_config(page_title="Hospital Performance | MediScope AI", page_icon="🏥", layout="wide")

from src.config import PALETTE
from src.dashboard_state import get_page_context, get_preaggregated_table
from src.eda import summarize_facility_county
from src.utils import (
    inject_custom_css,
    render_active_filter_banner,
    render_page_header,
    render_plain_english_guide,
)
from src.visualizations import (
    _apply_mediscope_layout,
    plot_county_utilisation,
    plot_facility_county_heatmap,
)


def main() -> None:
    if not st.session_state.get("_nav_shell_active"):
        inject_custom_css()

    filtered_df, filters_active, _, filter_summary = get_page_context()

    render_page_header(
        "Hospital & County Performance (RQ4)",
        "Facility utilisation, regional inpatient bed-days, emergency admission share, and average stay benchmarks.",
    )
    render_active_filter_banner(filter_summary)

    if filtered_df.empty:
        st.warning("No discharge records match the selected filters.")
        return

    render_plain_english_guide(
        "Interpreting Hospital Stay Benchmarks",
        "Darker navy cells in the heatmap indicate longer average patient stays. Switch <em>Select Top Facilities By</em> to "
        "<strong>Longest Average Stay</strong> to see why specialty rehabilitation, palliative, and psychiatric centers average 15–46 days, "
        "whereas high-volume acute medical centers average 4–7 days.",
    )

    if not filters_active:
        fac_county_df = get_preaggregated_table("agg_facility_county.parquet")
        if fac_county_df is None:
            fac_county_df = summarize_facility_county(filtered_df)
    else:
        fac_county_df = summarize_facility_county(filtered_df)

    ctl1, ctl2, ctl3 = st.columns(3)
    with ctl1:
        top_facs = st.slider("Top N Facilities in Heatmap", min_value=8, max_value=35, value=20)
    with ctl2:
        top_counties = st.slider("Top N Counties in Heatmap", min_value=5, max_value=20, value=12)
    with ctl3:
        rank_metric = st.selectbox(
            "Select Top Facilities By",
            options=["discharges", "avg_los"],
            format_func=lambda x: "Highest Discharge Volume" if x == "discharges" else "Longest Average Stay",
        )

    st.plotly_chart(
        plot_facility_county_heatmap(
            fac_county_df,
            top_n_facilities=top_facs,
            top_n_counties=top_counties,
            sort_by=rank_metric,
        ),
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_county_utilisation(fac_county_df, top_n=15), width="stretch")
    with c2:
        top_vol = fac_county_df.head(25)
        fig_ed = px.scatter(
            top_vol,
            x="discharges",
            y="avg_los",
            size="total_patient_days",
            color="ed_share_pct",
            hover_name="facility_name",
            hover_data=["hospital_county", "median_los", "avg_charges"],
            color_continuous_scale=[[0.0, PALETTE["light_teal"]], [0.5, PALETTE["teal"]], [1.0, PALETTE["navy"]]],
        )
        fig_ed.update_coloraxes(colorbar_title="ED %")
        _apply_mediscope_layout(
            fig_ed,
            "Top 25 Facilities: Volume vs. Avg Stay & ED Share (%)",
            xaxis_title="Total Discharges",
            yaxis_title="Average Length of Stay (Days)",
            height=430,
        )
        st.plotly_chart(fig_ed, width="stretch")

    st.markdown("#### Facility Utilisation Leaderboard")
    st.dataframe(fac_county_df.head(100).round(2), width="stretch")
    st.download_button(
        "Download Facility Performance Table (CSV)",
        data=fac_county_df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_facility_performance.csv",
        mime="text/csv",
    )


main()
