"""Page 7: Machine Learning Performance, Residual Diagnostics & Fairness."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ML Performance | MediScope AI", page_icon="📈", layout="wide")

from src.config import MODELS_DIR, PALETTE
from src.prediction import load_model_metadata
from src.utils import format_number, inject_custom_css, render_kpi_card, render_page_header
from src.visualizations import (
    _apply_mediscope_layout,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals_distribution,
    plot_residuals_vs_predicted,
)


def main() -> None:
    inject_custom_css()
    render_page_header(
        "Machine Learning Performance & Interpretability",
        "Supervised Length-of-Stay regression evaluation, leakage controls, residual analysis, and fairness audit.",
    )

    try:
        meta = load_model_metadata()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `python -m src.train_model` in the project root to generate trained model artifacts.")
        return

    comp_df = pd.DataFrame(meta["model_comparison"])
    selected_name = meta["selected_model"]
    best_row = comp_df[comp_df["model"] == selected_name].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Selected Production Model", selected_name.replace(" Regressor", ""), f"Artifact: {meta['model_size_mb']} MB")
    with k2:
        render_kpi_card("Test MAE (Hospital Days)", f"{best_row['test_mae']:.3f} d", f"Train MAE: {best_row['train_mae']:.3f} d", navy=True)
    with k3:
        render_kpi_card("Test RMSE (Hospital Days)", f"{best_row['test_rmse']:.3f} d", f"Train RMSE: {best_row['train_rmse']:.3f} d")
    with k4:
        render_kpi_card("Test R² Score", f"{best_row['test_r2']:.4f}", f"Unseen Test Rows: {format_number(meta['test_rows'])}", navy=True)

    st.markdown(
        f"""
        <div class="insight-box">
            <strong>Validation Protocol & Hospital-Day Interpretation:</strong> Models were trained on
            <strong>{format_number(meta['train_rows'])}</strong> records and evaluated on
            <strong>{format_number(meta['test_rows'])}</strong> strictly unseen records via
            <em>{meta['validation_strategy']}</em>. On average, the selected model's Length-of-Stay predictions
            differ from true inpatient stays by <strong>{best_row['test_mae']:.2f} hospital days</strong> (MAE),
            while RMSE of <strong>{best_row['test_rmse']:.2f} days</strong> penalises rare multi-week outliers more heavily.
            Note: {meta['chronological_validation_note']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Model Comparison Table (Train vs. Unseen Test)")
    st.dataframe(comp_df, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_model_comparison(comp_df), width='stretch')
    with c2:
        st.plotly_chart(plot_feature_importance(meta["permutation_importance"]), width='stretch')

    eval_sample_path = MODELS_DIR / "test_eval_sample.parquet"
    if eval_sample_path.exists():
        eval_df = pd.read_parquet(eval_sample_path)
        st.plotly_chart(plot_actual_vs_predicted(eval_df), width='stretch')
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(plot_residuals_distribution(eval_df), width='stretch')
        with r2:
            st.plotly_chart(plot_residuals_vs_predicted(eval_df), width='stretch')

    st.markdown("#### Fairness & Subgroup Error Analysis (Protected Attributes Excluded From Training)")
    fairness_attr = st.selectbox(
        "Inspect Fairness Audit Across Demographic Subgroup",
        options=list(meta.get("fairness_audit", {}).keys()),
    )
    if fairness_attr and fairness_attr in meta.get("fairness_audit", {}):
        fair_df = pd.DataFrame(meta["fairness_audit"][fairness_attr])
        f1, f2 = st.columns([1.2, 1])
        with f1:
            st.dataframe(fair_df, width='stretch')
        with f2:
            fig_fair = px.bar(
                fair_df,
                x="subgroup",
                y="mae",
                color="mean_error_bias",
                color_continuous_scale=[[0.0, PALETTE["teal"]], [0.5, PALETTE["light_teal"]], [1.0, PALETTE["navy"]]],
            )
            _apply_mediscope_layout(
                fig_fair,
                f"Test MAE Across `{fairness_attr}` Subgroups",
                xaxis_title=fairness_attr,
                yaxis_title="MAE (Hospital Days)",
                height=380,
            )
            st.plotly_chart(fig_fair, width='stretch')

    adm_sens = meta.get("admission_only_ridge_sensitivity", {})
    if adm_sens:
        m = adm_sens["test_metrics"]
        st.info(
            f"**Admission-Only Sensitivity Experiment (excluding APR Severity & Mortality Risk):** "
            f"Ridge Regression achieved Test MAE = {m['mae']:.3f}d, RMSE = {m['rmse']:.3f}d, R² = {m['r2']:.4f}. "
            f"This demonstrates how much predictive signal comes from early clinical acuity staging."
        )


if __name__ == "__main__":
    main()
