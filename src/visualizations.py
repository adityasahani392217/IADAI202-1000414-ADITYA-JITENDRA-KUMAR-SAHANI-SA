"""Reusable Plotly visualizations styled with the MediScope AI (LogiSight) design system."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import AGE_GROUP_ORDER, CHART_COLORWAY, PALETTE, SEVERITY_ORDER


def _apply_mediscope_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 430,
) -> go.Figure:
    """Apply consistent typography, margins, gridlines, and palette to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=PALETTE["navy"], family="Inter, Segoe UI, sans-serif"),
            x=0.01,
        ),
        paper_bgcolor=PALETTE["white"],
        plot_bgcolor=PALETTE["white"],
        font=dict(color=PALETTE["text_primary"], size=12, family="Inter, Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=55, r=25, t=52, b=50),
        xaxis=dict(
            title=xaxis_title,
            gridcolor="#F1F5F9",
            linecolor=PALETTE["border"],
            zeroline=False,
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="#F1F5F9",
            linecolor=PALETTE["border"],
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=PALETTE["border"],
            borderwidth=1,
        ),
    )
    return fig


def plot_avg_stay_by_diagnosis(diag_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Compulsory Chart 1: Interactive Bar Chart of Average Hospital Stay per Diagnosis Code."""
    top = diag_df.sort_values("avg_los", ascending=False).head(top_n).copy()
    top = top.sort_values("avg_los", ascending=True)
    top["short_desc"] = top["ccsr_diagnosis_description"].astype(str).str.slice(0, 42)
    top["label"] = top["ccsr_diagnosis_code"].astype(str) + " — " + top["short_desc"]

    fig = px.bar(
        top,
        x="avg_los",
        y="label",
        orientation="h",
        color="avg_los",
        color_continuous_scale=[[0.0, PALETTE["light_teal"]], [0.5, PALETTE["teal"]], [1.0, PALETTE["navy"]]],
        custom_data=[
            "ccsr_diagnosis_code",
            "ccsr_diagnosis_description",
            "discharges",
            "median_los",
            "avg_charges",
        ],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b>: %{customdata[1]}<br>"
            "Average Stay: <b>%{x:.2f} days</b><br>"
            "Median Stay: %{customdata[3]:.1f} days<br>"
            "Discharges: %{customdata[2]:,}<br>"
            "Avg Charges: $%{customdata[4]:,.0f}<extra></extra>"
        )
    )
    fig.update_coloraxes(showscale=False)
    return _apply_mediscope_layout(
        fig,
        f"Top {top_n} Diagnoses by Average Length of Stay",
        xaxis_title="Average Length of Stay (Days)",
        yaxis_title="CCSR Diagnosis",
        height=max(420, top_n * 28),
    )


def plot_top_volume_diagnoses(diag_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Bar chart of highest-volume diagnoses by inpatient discharge count."""
    top = diag_df.sort_values("discharges", ascending=False).head(top_n).copy()
    top = top.sort_values("discharges", ascending=True)
    top["label"] = (
        top["ccsr_diagnosis_code"].astype(str)
        + " — "
        + top["ccsr_diagnosis_description"].astype(str).str.slice(0, 38)
    )
    fig = px.bar(
        top,
        x="discharges",
        y="label",
        orientation="h",
        color_discrete_sequence=[PALETTE["navy"]],
        custom_data=["ccsr_diagnosis_description", "avg_los", "avg_charges"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Discharges: <b>%{x:,}</b><br>"
            "Avg LOS: %{customdata[1]:.2f} days<br>"
            "Avg Charges: $%{customdata[2]:,.0f}<extra></extra>"
        )
    )
    return _apply_mediscope_layout(
        fig,
        f"Top {top_n} Diagnoses by Discharge Volume",
        xaxis_title="Number of Discharges",
        yaxis_title="CCSR Diagnosis",
        height=max(400, top_n * 28),
    )


def plot_charges_boxplot_by_severity(
    df: pd.DataFrame,
    use_log_scale: bool = True,
    max_points_per_group: int = 4000,
) -> go.Figure:
    """Compulsory Chart 2: Interactive Box Plot of Total Charges Segmented by Severity.

    Note: Uses stratified downsampling only if a group exceeds `max_points_per_group`
    so the browser stays responsive while accurately preserving quartiles/outliers.
    """
    sev_present = [s for s in SEVERITY_ORDER if s in set(df["apr_severity_description"].astype(str))]
    plot_parts = []
    for sev in sev_present:
        sub = df[df["apr_severity_description"].astype(str) == sev][
            ["apr_severity_description", "total_charges"]
        ]
        if len(sub) > max_points_per_group:
            # Always preserve top/bottom extremes alongside random core sample for faithful whiskers/outliers
            n_tail = min(200, len(sub) // 10)
            sorted_sub = sub.sort_values("total_charges")
            low_tail = sorted_sub.iloc[:n_tail]
            high_tail = sorted_sub.iloc[-n_tail:]
            mid = sorted_sub.iloc[n_tail:-n_tail].sample(
                n=min(max_points_per_group - 2 * n_tail, len(sorted_sub) - 2 * n_tail),
                random_state=42,
            )
            sub = pd.concat([low_tail, mid, high_tail], ignore_index=True)
        plot_parts.append(sub)

    plot_df = pd.concat(plot_parts, ignore_index=True) if plot_parts else df
    color_map = {
        "Minor": "#2DD4BF",
        "Moderate": PALETTE["teal"],
        "Major": "#3B82F6",
        "Extreme": PALETTE["navy"],
        "Not Applicable": PALETTE["grey"],
    }
    fig = px.box(
        plot_df,
        x="apr_severity_description",
        y="total_charges",
        color="apr_severity_description",
        category_orders={"apr_severity_description": sev_present},
        color_discrete_map=color_map,
        points="outliers",
    )
    if use_log_scale:
        fig.update_yaxes(type="log", tickprefix="$")
    else:
        fig.update_yaxes(tickprefix="$")
    fig.update_layout(showlegend=False)
    return _apply_mediscope_layout(
        fig,
        "Total Hospital Charges Distribution by Illness Severity",
        xaxis_title="APR Severity of Illness",
        yaxis_title="Total Charges (USD)" + (" [Log Scale]" if use_log_scale else ""),
        height=460,
    )


def plot_charges_vs_costs_by_severity(sev_df: pd.DataFrame) -> go.Figure:
    """Grouped bar comparing mean and median hospital charges vs costs across severity."""
    sev_present = [s for s in SEVERITY_ORDER if s in set(sev_df["apr_severity_description"].astype(str))]
    ordered = sev_df.set_index("apr_severity_description").reindex(sev_present).dropna(how="all").reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Average Charges",
            x=ordered["apr_severity_description"],
            y=ordered["avg_charges"],
            marker_color=PALETTE["navy"],
            hovertemplate="Severity: %{x}<br>Avg Charges: <b>$%{y:,.0f}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Average Costs",
            x=ordered["apr_severity_description"],
            y=ordered["avg_costs"],
            marker_color=PALETTE["teal"],
            hovertemplate="Severity: %{x}<br>Avg Costs: <b>$%{y:,.0f}</b><extra></extra>",
        )
    )
    fig.update_layout(barmode="group", yaxis_tickprefix="$")
    return _apply_mediscope_layout(
        fig,
        "Average Hospital Charges vs. Costs by Severity",
        xaxis_title="APR Severity of Illness",
        yaxis_title="Amount (USD)",
        height=430,
    )


def plot_charges_by_age_gender(age_gender_df: pd.DataFrame) -> go.Figure:
    """RQ2: Grouped bar chart of average hospital charges by Age Group and Gender."""
    filtered = age_gender_df[age_gender_df["gender"].astype(str).isin(["F", "M"])].copy()
    fig = px.bar(
        filtered,
        x="age_group",
        y="avg_charges",
        color="gender",
        barmode="group",
        category_orders={"age_group": AGE_GROUP_ORDER},
        color_discrete_map={"F": PALETTE["teal"], "M": PALETTE["navy"]},
        custom_data=["discharges", "median_charges", "avg_los"],
    )
    fig.update_traces(
        hovertemplate=(
            "Age Group: %{x}<br>"
            "Avg Charges: <b>$%{y:,.0f}</b><br>"
            "Median Charges: $%{customdata[1]:,.0f}<br>"
            "Discharges: %{customdata[0]:,}<br>"
            "Avg LOS: %{customdata[2]:.2f} days<extra></extra>"
        )
    )
    fig.update_yaxes(tickprefix="$")
    return _apply_mediscope_layout(
        fig,
        "Average Hospital Charges by Age Group and Gender",
        xaxis_title="Patient Age Group",
        yaxis_title="Average Total Charges (USD)",
        height=430,
    )


def plot_facility_county_heatmap(
    fac_county_df: pd.DataFrame,
    top_n_facilities: int = 20,
    top_n_counties: int = 12,
    sort_by: str = "discharges",
) -> go.Figure:
    """Compulsory Chart 3: Interactive Heatmap of Average Stay Duration by Facility & County."""
    top_counties = (
        fac_county_df.groupby("hospital_county", observed=True)["discharges"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_counties)
        .index.tolist()
    )
    sub = fac_county_df[fac_county_df["hospital_county"].isin(top_counties)].copy()
    top_facs = (
        sub.sort_values(sort_by, ascending=False)
        .head(top_n_facilities)["facility_name"]
        .astype(str)
        .tolist()
    )
    sub = sub[sub["facility_name"].astype(str).isin(top_facs)].copy()
    sub["fac_short"] = sub["facility_name"].astype(str).str.slice(0, 42)

    pivot_los = sub.pivot_table(
        index="fac_short",
        columns="hospital_county",
        values="avg_los",
        aggfunc="mean",
        observed=True,
    )
    pivot_vol = sub.pivot_table(
        index="fac_short",
        columns="hospital_county",
        values="discharges",
        aggfunc="sum",
        observed=True,
    )

    # Order rows by row mean so highest stay facilities sit clearly at top
    row_order = pivot_los.mean(axis=1).sort_values(ascending=True).index.tolist()
    pivot_los = pivot_los.loc[row_order]
    pivot_vol = pivot_vol.reindex(index=row_order, columns=pivot_los.columns)

    text_vals = np.where(
        pivot_los.notna(),
        np.char.add(np.round(pivot_los.fillna(0).to_numpy(), 1).astype(str), "d"),
        "",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_los.to_numpy(),
            x=pivot_los.columns.astype(str).tolist(),
            y=pivot_los.index.astype(str).tolist(),
            customdata=pivot_vol.fillna(0).to_numpy(),
            text=text_vals,
            texttemplate="%{text}",
            colorscale=[
                [0.0, PALETTE["light_teal"]],
                [0.45, PALETTE["teal"]],
                [1.0, PALETTE["navy"]],
            ],
            colorbar=dict(title="Avg LOS (Days)"),
            hovertemplate=(
                "Facility: <b>%{y}</b><br>"
                "County: %{x}<br>"
                "Avg LOS: <b>%{z:.2f} days</b><br>"
                "Discharges: %{customdata:,.0f}<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    return _apply_mediscope_layout(
        fig,
        f"Average Length of Stay (Days) — Top {len(pivot_los)} Facilities Across Top {len(pivot_los.columns)} Counties",
        xaxis_title="Hospital County",
        yaxis_title="Facility Name",
        height=max(460, len(pivot_los) * 26),
    )


def plot_county_utilisation(fac_county_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of total inpatient discharges and average LOS by county."""
    county_agg = (
        fac_county_df.groupby("hospital_county", observed=True)
        .agg(
            discharges=("discharges", "sum"),
            patient_days=("total_patient_days", "sum"),
            facilities=("facility_name", "nunique"),
        )
        .reset_index()
    )
    county_agg["avg_los"] = county_agg["patient_days"] / np.where(
        county_agg["discharges"] > 0, county_agg["discharges"], np.nan
    )
    top = county_agg.sort_values("discharges", ascending=False).head(top_n)
    fig = px.bar(
        top,
        x="hospital_county",
        y="discharges",
        color="avg_los",
        color_continuous_scale=[[0.0, PALETTE["light_teal"]], [0.5, PALETTE["teal"]], [1.0, PALETTE["navy"]]],
        custom_data=["avg_los", "facilities", "patient_days"],
    )
    fig.update_traces(
        hovertemplate=(
            "County: <b>%{x}</b><br>"
            "Discharges: <b>%{y:,}</b><br>"
            "Weighted Avg LOS: %{customdata[0]:.2f} days<br>"
            "Facilities: %{customdata[1]}<br>"
            "Total Inpatient Days: %{customdata[2]:,.0f}<extra></extra>"
        )
    )
    fig.update_coloraxes(colorbar_title="Avg LOS")
    return _apply_mediscope_layout(
        fig,
        f"Top {top_n} Counties by Inpatient Discharge Volume",
        xaxis_title="Hospital County",
        yaxis_title="Total Discharges",
        height=430,
    )


def plot_payment_type_pie(pay_df: pd.DataFrame) -> go.Figure:
    """Compulsory Chart 4: Interactive Pie Chart of Patient Distribution by Primary Payment Type."""
    df_p = pay_df.sort_values("discharges", ascending=False).copy()
    # Group very small (<1%) payers only visually in text if needed, keep all distinct in legend
    fig = px.pie(
        df_p,
        names="payment_typology_1",
        values="discharges",
        hole=0.38,
        color_discrete_sequence=CHART_COLORWAY + ["#94A3B8"],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=(
            "Primary Payer: <b>%{label}</b><br>"
            "Discharges: <b>%{value:,}</b> (%{percent})<extra></extra>"
        ),
    )
    return _apply_mediscope_layout(
        fig,
        "Primary Payment Type Mix (Typology 1)",
        height=460,
    )


def plot_charges_by_payment(pay_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing mean and median charges across primary payment types."""
    ordered = pay_df.sort_values("avg_charges", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Average Charges",
            y=ordered["payment_typology_1"],
            x=ordered["avg_charges"],
            orientation="h",
            marker_color=PALETTE["navy"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Median Charges",
            y=ordered["payment_typology_1"],
            x=ordered["median_charges"],
            orientation="h",
            marker_color=PALETTE["teal"],
        )
    )
    fig.update_layout(barmode="group", xaxis_tickprefix="$")
    return _apply_mediscope_layout(
        fig,
        "Mean vs. Median Charges by Primary Payer",
        xaxis_title="Total Charges (USD)",
        yaxis_title="Primary Payment Typology",
        height=440,
    )


def plot_los_histogram(
    los_hist_df: pd.DataFrame | None = None,
    df: pd.DataFrame | None = None,
    max_los_view: int = 45,
) -> go.Figure:
    """Compulsory Chart 5: Interactive Histogram of Overall Length of Stay Distribution."""
    if los_hist_df is not None and not los_hist_df.empty:
        agg = (
            los_hist_df.groupby("length_of_stay", observed=True)["discharges"]
            .sum()
            .reset_index()
            .sort_values("length_of_stay")
        )
        total_n = agg["discharges"].sum()
        mean_los = float((agg["length_of_stay"] * agg["discharges"]).sum() / max(total_n, 1))
        cum = agg["discharges"].cumsum()
        median_los = float(agg.loc[cum >= total_n / 2.0, "length_of_stay"].iloc[0])
        view_agg = agg[agg["length_of_stay"] <= max_los_view]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=view_agg["length_of_stay"],
                    y=view_agg["discharges"],
                    marker_color=PALETTE["teal"],
                    name="Discharges",
                    hovertemplate="LOS: <b>%{x} days</b><br>Discharges: <b>%{y:,}</b><extra></extra>",
                )
            ]
        )
    elif df is not None and not df.empty:
        los_series = df["length_of_stay"].astype(float)
        mean_los = float(los_series.mean())
        median_los = float(los_series.median())
        view_df = df[df["length_of_stay"] <= max_los_view]
        fig = px.histogram(
            view_df,
            x="length_of_stay",
            nbins=max_los_view,
            color_discrete_sequence=[PALETTE["teal"]],
        )
    else:
        return go.Figure()

    fig.add_vline(
        x=median_los,
        line_dash="dash",
        line_color=PALETTE["navy"],
        annotation_text=f"Median: {median_los:.1f}d",
        annotation_position="top right",
    )
    fig.add_vline(
        x=mean_los,
        line_dash="dot",
        line_color=PALETTE["amber"],
        annotation_text=f"Mean: {mean_los:.2f}d",
        annotation_position="top right",
        annotation_yshift=-24,
    )
    return _apply_mediscope_layout(
        fig,
        f"Overall Distribution of Inpatient Length of Stay (1 to {max_los_view} Days View)",
        xaxis_title="Length of Stay (Days)",
        yaxis_title="Number of Discharges",
        height=440,
    )


def plot_los_by_severity(sev_df: pd.DataFrame) -> go.Figure:
    """Grouped bar comparing mean and median LOS across severity levels."""
    sev_present = [s for s in SEVERITY_ORDER if s in set(sev_df["apr_severity_description"].astype(str))]
    ordered = sev_df.set_index("apr_severity_description").reindex(sev_present).dropna(how="all").reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Mean LOS",
            x=ordered["apr_severity_description"],
            y=ordered["avg_los"],
            marker_color=PALETTE["navy"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Median LOS",
            x=ordered["apr_severity_description"],
            y=ordered["median_los"],
            marker_color=PALETTE["teal"],
        )
    )
    fig.update_layout(barmode="group")
    return _apply_mediscope_layout(
        fig,
        "Mean vs. Median Length of Stay Across Illness Severity Categories",
        xaxis_title="APR Severity of Illness",
        yaxis_title="Length of Stay (Days)",
        height=420,
    )


def plot_model_comparison(comp_df: pd.DataFrame) -> go.Figure:
    """Compare Train vs Test MAE and RMSE across candidate regression models."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Test MAE (Days)",
            x=comp_df["model"],
            y=comp_df["test_mae"],
            marker_color=PALETTE["teal"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Test RMSE (Days)",
            x=comp_df["model"],
            y=comp_df["test_rmse"],
            marker_color=PALETTE["navy"],
        )
    )
    fig.update_layout(barmode="group")
    return _apply_mediscope_layout(
        fig,
        "Model Comparison on Unseen Test Split",
        xaxis_title="Candidate Model",
        yaxis_title="Error (Hospital Days)",
        height=420,
    )


def plot_actual_vs_predicted(eval_df: pd.DataFrame) -> go.Figure:
    """Actual vs Predicted Length of Stay scatter plot with ideal y=x diagonal."""
    fig = px.scatter(
        eval_df,
        x="actual_los",
        y="predicted_los",
        opacity=0.35,
        color_discrete_sequence=[PALETTE["teal"]],
        hover_data=["apr_severity_code", "age_group", "residual"],
    )
    max_val = float(max(eval_df["actual_los"].max(), eval_df["predicted_los"].max(), 30))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Ideal Prediction (y = x)",
            line=dict(color=PALETTE["navy"], dash="dash", width=2),
        )
    )
    return _apply_mediscope_layout(
        fig,
        "Actual vs. Predicted Length of Stay (Unseen Test Sample)",
        xaxis_title="Actual Length of Stay (Days)",
        yaxis_title="Predicted Length of Stay (Days)",
        height=430,
    )


def plot_residuals_distribution(eval_df: pd.DataFrame) -> go.Figure:
    """Histogram of model prediction residuals (Actual - Predicted)."""
    clipped = eval_df[(eval_df["residual"] >= -30) & (eval_df["residual"] <= 40)]
    fig = px.histogram(
        clipped,
        x="residual",
        nbins=60,
        color_discrete_sequence=[PALETTE["navy"]],
    )
    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["teal"])
    return _apply_mediscope_layout(
        fig,
        "Residual Distribution (Actual − Predicted Days)",
        xaxis_title="Residual (Days)",
        yaxis_title="Count",
        height=400,
    )


def plot_residuals_vs_predicted(eval_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of residuals against predicted length of stay."""
    fig = px.scatter(
        eval_df,
        x="predicted_los",
        y="residual",
        opacity=0.35,
        color_discrete_sequence=[PALETTE["grey"]],
    )
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["teal"])
    return _apply_mediscope_layout(
        fig,
        "Residuals vs. Predicted Length of Stay",
        xaxis_title="Predicted Length of Stay (Days)",
        yaxis_title="Residual (Actual − Predicted)",
        height=400,
    )


def plot_feature_importance(perm_rows: list[dict]) -> go.Figure:
    """Horizontal bar chart of Permutation Importance (MAE increase when shuffled)."""
    imp_df = pd.DataFrame(perm_rows).sort_values("importance_mean", ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=imp_df["importance_mean"],
                y=imp_df["feature"],
                orientation="h",
                error_x=dict(type="data", array=imp_df["importance_std"]),
                marker_color=PALETTE["teal"],
                hovertemplate="Feature: <b>%{y}</b><br>MAE Increase: <b>%{x:.4f} days</b><extra></extra>",
            )
        ]
    )
    return _apply_mediscope_layout(
        fig,
        "Permutation Feature Importance (Impact on Test MAE)",
        xaxis_title="Mean Increase in MAE When Permuted (Days)",
        yaxis_title="Predictor Feature",
        height=410,
    )
