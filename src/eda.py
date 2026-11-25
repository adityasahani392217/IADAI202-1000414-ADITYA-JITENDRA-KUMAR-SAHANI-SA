"""Exploratory Data Analysis, Research Question Statistics, and Cloud Aggregate Builder."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    AGGREGATES_DIR,
    CLOUD_AGE_GENDER_AGG_PATH,
    CLOUD_DIAGNOSIS_AGG_PATH,
    CLOUD_FACILITY_COUNTY_AGG_PATH,
    CLOUD_LOS_HIST_AGG_PATH,
    CLOUD_PAYMENT_AGG_PATH,
    CLOUD_SAMPLE_PARQUET_PATH,
    CLOUD_SEVERITY_AGG_PATH,
    CLOUD_SUMMARY_JSON_PATH,
    PROCESSED_PARQUET_PATH,
    REPORTS_DIR,
)


def compute_core_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute core inpatient discharge KPIs from a cleaned DataFrame."""
    if df.empty:
        return {
            "total_discharges": 0,
            "unique_facilities": 0,
            "unique_counties": 0,
            "avg_los": 0.0,
            "median_los": 0.0,
            "min_los": 0.0,
            "max_los": 0.0,
            "los_std": 0.0,
            "los_iqr": 0.0,
            "avg_charges": 0.0,
            "median_charges": 0.0,
            "total_charges_sum": 0.0,
            "avg_costs": 0.0,
            "median_costs": 0.0,
            "total_costs_sum": 0.0,
            "avg_cost_per_day": 0.0,
            "avg_charge_to_cost": 0.0,
            "ed_pct": 0.0,
            "surgical_pct": 0.0,
        }

    los = df["length_of_stay"].astype("float64")
    chg = df["total_charges"].astype("float64")
    cst = df["total_costs"].astype("float64")

    q1_los = float(los.quantile(0.25))
    q3_los = float(los.quantile(0.75))

    ed_pct = (
        float((df["emergency_department_indicator"].astype(str) == "Y").mean() * 100.0)
        if "emergency_department_indicator" in df.columns
        else 0.0
    )
    surg_pct = (
        float((df["apr_medical_surgical_description"].astype(str) == "Surgical").mean() * 100.0)
        if "apr_medical_surgical_description" in df.columns
        else 0.0
    )

    return {
        "total_discharges": int(len(df)),
        "unique_facilities": int(df["facility_name"].nunique()) if "facility_name" in df.columns else 0,
        "unique_counties": int(df["hospital_county"].nunique()) if "hospital_county" in df.columns else 0,
        "avg_los": round(float(los.mean()), 2),
        "median_los": round(float(los.median()), 2),
        "min_los": round(float(los.min()), 2),
        "max_los": round(float(los.max()), 2),
        "los_std": round(float(los.std()), 2),
        "los_iqr": round(q3_los - q1_los, 2),
        "avg_charges": round(float(chg.mean()), 2),
        "median_charges": round(float(chg.median()), 2),
        "total_charges_sum": round(float(chg.sum()), 2),
        "avg_costs": round(float(cst.mean()), 2),
        "median_costs": round(float(cst.median()), 2),
        "total_costs_sum": round(float(cst.sum()), 2),
        "avg_cost_per_day": round(float((cst / los).mean()), 2),
        "avg_charge_to_cost": round(float((chg / cst).mean()), 2),
        "ed_pct": round(ed_pct, 2),
        "surgical_pct": round(surg_pct, 2),
    }


def summarize_diagnosis(df: pd.DataFrame, min_discharges: int = 1) -> pd.DataFrame:
    """RQ1: Aggregate Length of Stay, charges, and costs by CCSR Diagnosis."""
    grouped = (
        df.groupby(["ccsr_diagnosis_code", "ccsr_diagnosis_description"], observed=True)
        .agg(
            discharges=("length_of_stay", "count"),
            avg_los=("length_of_stay", "mean"),
            median_los=("length_of_stay", "median"),
            std_los=("length_of_stay", "std"),
            avg_charges=("total_charges", "mean"),
            median_charges=("total_charges", "median"),
            avg_costs=("total_costs", "mean"),
            median_costs=("total_costs", "median"),
        )
        .reset_index()
    )
    if min_discharges > 1:
        grouped = grouped[grouped["discharges"] >= min_discharges]
    return grouped.sort_values("avg_los", ascending=False).reset_index(drop=True)


def summarize_age_gender(df: pd.DataFrame) -> pd.DataFrame:
    """RQ2: Pivot/groupby of hospital charges, costs, and LOS by Age Group and Gender."""
    return (
        df.groupby(["age_group", "gender"], observed=True)
        .agg(
            discharges=("length_of_stay", "count"),
            avg_charges=("total_charges", "mean"),
            median_charges=("total_charges", "median"),
            avg_costs=("total_costs", "mean"),
            median_costs=("total_costs", "median"),
            avg_los=("length_of_stay", "mean"),
            median_los=("length_of_stay", "median"),
        )
        .reset_index()
    )


def summarize_severity(df: pd.DataFrame) -> pd.DataFrame:
    """RQ3: Analyse illness severity vs charges, costs, and length of stay."""
    return (
        df.groupby(["apr_severity_code", "apr_severity_description"], observed=True)
        .agg(
            discharges=("length_of_stay", "count"),
            avg_los=("length_of_stay", "mean"),
            median_los=("length_of_stay", "median"),
            avg_charges=("total_charges", "mean"),
            median_charges=("total_charges", "median"),
            q25_charges=("total_charges", lambda s: float(np.quantile(s, 0.25))),
            q75_charges=("total_charges", lambda s: float(np.quantile(s, 0.75))),
            avg_costs=("total_costs", "mean"),
            median_costs=("total_costs", "median"),
        )
        .reset_index()
        .sort_values("apr_severity_code")
        .reset_index(drop=True)
    )


def summarize_facility_county(df: pd.DataFrame) -> pd.DataFrame:
    """RQ4: Aggregate hospital utilisation and stay duration by facility and county."""
    return (
        df.groupby(
            ["hospital_county", "permanent_facility_id", "facility_name"],
            observed=True,
        )
        .agg(
            discharges=("length_of_stay", "count"),
            total_patient_days=("length_of_stay", "sum"),
            avg_los=("length_of_stay", "mean"),
            median_los=("length_of_stay", "median"),
            avg_charges=("total_charges", "mean"),
            avg_costs=("total_costs", "mean"),
            ed_share_pct=("is_ed_admission", lambda s: float(s.mean() * 100.0)),
        )
        .reset_index()
        .sort_values("discharges", ascending=False)
        .reset_index(drop=True)
    )


def summarize_payment(df: pd.DataFrame) -> pd.DataFrame:
    """RQ5: Analyse primary payment category distribution, charges, and stay duration."""
    total_n = max(len(df), 1)
    grouped = (
        df.groupby("payment_typology_1", observed=True)
        .agg(
            discharges=("length_of_stay", "count"),
            avg_charges=("total_charges", "mean"),
            median_charges=("total_charges", "median"),
            avg_costs=("total_costs", "mean"),
            median_costs=("total_costs", "median"),
            avg_los=("length_of_stay", "mean"),
            median_los=("length_of_stay", "median"),
            avg_payer_count=("payer_count", "mean"),
        )
        .reset_index()
        .sort_values("discharges", ascending=False)
        .reset_index(drop=True)
    )
    grouped["share_pct"] = (grouped["discharges"] / total_n * 100.0).round(2)
    return grouped


def compute_research_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute rigorous parametric and non-parametric statistics for RQ1-RQ5 and correlations."""
    # Correlations (Spearman rank correlation is appropriate for heavily skewed healthcare metrics)
    corr_cols = ["length_of_stay", "total_charges", "total_costs", "severity_numeric", "age_ordinal"]
    avail = [c for c in corr_cols if c in df.columns]
    # Use a deterministic 250k sample for exact Spearman p-values if dataset > 250k to keep execution fast
    sample_df = df[avail].sample(n=min(len(df), 250_000), random_state=42)
    spearman_mat = sample_df.corr(method="spearman").round(4).to_dict()
    pearson_mat = df[avail].corr(method="pearson").round(4).to_dict()

    # Kruskal-Wallis H-tests for non-parametric group differences (RQ3 severity, RQ5 payment)
    sev_groups = [
        g["total_charges"].to_numpy()
        for _, g in df[df["apr_severity_description"] != "Not Applicable"].groupby(
            "apr_severity_description", observed=True
        )
    ]
    kw_sev_charges = stats.kruskal(*sev_groups) if len(sev_groups) >= 2 else (np.nan, np.nan)

    sev_los_groups = [
        g["length_of_stay"].to_numpy()
        for _, g in df[df["apr_severity_description"] != "Not Applicable"].groupby(
            "apr_severity_description", observed=True
        )
    ]
    kw_sev_los = stats.kruskal(*sev_los_groups) if len(sev_los_groups) >= 2 else (np.nan, np.nan)

    pay_groups = [
        g["total_charges"].to_numpy()
        for _, g in df.groupby("payment_typology_1", observed=True)
        if len(g) >= 30
    ]
    kw_pay_charges = stats.kruskal(*pay_groups) if len(pay_groups) >= 2 else (np.nan, np.nan)

    # Skewness
    los_skew = round(float(stats.skew(df["length_of_stay"].to_numpy())), 3)
    chg_skew = round(float(stats.skew(df["total_charges"].to_numpy())), 3)
    cst_skew = round(float(stats.skew(df["total_costs"].to_numpy())), 3)

    # Top distributions
    age_dist = df["age_group"].value_counts().to_dict()
    gender_dist = df["gender"].value_counts().to_dict()
    adm_dist = df["type_of_admission"].value_counts().to_dict()
    med_surg_dist = df["apr_medical_surgical_description"].value_counts().to_dict()
    ed_dist = df["emergency_department_indicator"].value_counts().to_dict()

    # Pivot table: County-level utilisation (top 15 counties by discharges)
    county_summary = (
        df.groupby("hospital_county", observed=True)
        .agg(
            discharges=("length_of_stay", "count"),
            facilities=("facility_name", "nunique"),
            avg_los=("length_of_stay", "mean"),
            median_los=("length_of_stay", "median"),
            avg_charges=("total_charges", "mean"),
            total_patient_days=("length_of_stay", "sum"),
        )
        .reset_index()
        .sort_values("discharges", ascending=False)
    )

    # Newborn vs non-newborn summary
    nb_stats = (
        df.groupby("is_newborn", observed=True)
        .agg(
            discharges=("length_of_stay", "count"),
            avg_los=("length_of_stay", "mean"),
            median_los=("length_of_stay", "median"),
            avg_charges=("total_charges", "mean"),
            avg_costs=("total_costs", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "kpis": compute_core_kpis(df),
        "skewness": {"length_of_stay": los_skew, "total_charges": chg_skew, "total_costs": cst_skew},
        "spearman_corr": spearman_mat,
        "pearson_corr": pearson_mat,
        "kruskal_wallis": {
            "severity_vs_charges": {"H": round(float(kw_sev_charges[0]), 2), "p": float(kw_sev_charges[1])},
            "severity_vs_los": {"H": round(float(kw_sev_los[0]), 2), "p": float(kw_sev_los[1])},
            "payment_vs_charges": {"H": round(float(kw_pay_charges[0]), 2), "p": float(kw_pay_charges[1])},
        },
        "distributions": {
            "age_group": {str(k): int(v) for k, v in age_dist.items()},
            "gender": {str(k): int(v) for k, v in gender_dist.items()},
            "type_of_admission": {str(k): int(v) for k, v in adm_dist.items()},
            "med_surg": {str(k): int(v) for k, v in med_surg_dist.items()},
            "emergency_department": {str(k): int(v) for k, v in ed_dist.items()},
        },
        "top_counties": county_summary.head(15).round(2).to_dict(orient="records"),
        "newborn_vs_general": nb_stats,
    }


def build_cloud_artifacts_and_eda(sample_size: int = 80_000) -> dict[str, Any]:
    """Generate full-dataset aggregates, stratified cloud sample, and EDA JSON."""
    t0 = time.time()
    if not PROCESSED_PARQUET_PATH.exists():
        raise FileNotFoundError("Run `python -m src.preprocessing` before running EDA.")

    df = pd.read_parquet(PROCESSED_PARQUET_PATH)
    AGGREGATES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    diag_df = summarize_diagnosis(df, min_discharges=1)
    age_gender_df = summarize_age_gender(df)
    sev_df = summarize_severity(df)
    fac_county_df = summarize_facility_county(df)
    pay_df = summarize_payment(df)

    # Exact LOS histogram counts (1..120)
    los_hist_df = (
        df.groupby(["length_of_stay", "apr_severity_description"], observed=True)
        .size()
        .reset_index(name="discharges")
    )

    diag_df.to_parquet(CLOUD_DIAGNOSIS_AGG_PATH, index=False)
    age_gender_df.to_parquet(CLOUD_AGE_GENDER_AGG_PATH, index=False)
    sev_df.to_parquet(CLOUD_SEVERITY_AGG_PATH, index=False)
    fac_county_df.to_parquet(CLOUD_FACILITY_COUNTY_AGG_PATH, index=False)
    pay_df.to_parquet(CLOUD_PAYMENT_AGG_PATH, index=False)
    los_hist_df.to_parquet(CLOUD_LOS_HIST_AGG_PATH, index=False)

    # Stratified sample across Hospital Service Area + Severity for interactive cloud fallback
    strata = (
        df["hospital_service_area"].astype(str)
        + "_"
        + df["apr_severity_description"].astype(str)
    )
    frac = min(1.0, sample_size / len(df))
    sample_df = (
        df.assign(_strata=strata)
        .groupby("_strata", group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=42))
        .drop(columns=["_strata"], errors="ignore")
    )
    # Trim to target size if rounding slightly exceeded
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(n=sample_size, random_state=42)
    sample_df.to_parquet(CLOUD_SAMPLE_PARQUET_PATH, index=False, compression="zstd")

    summary = compute_research_statistics(df)
    summary["top_15_longest_stay_diagnoses_min500"] = (
        diag_df[diag_df["discharges"] >= 500].head(15).round(2).to_dict(orient="records")
    )
    summary["top_15_volume_diagnoses"] = (
        diag_df.sort_values("discharges", ascending=False).head(15).round(2).to_dict(orient="records")
    )
    summary["severity_summary"] = sev_df.round(2).to_dict(orient="records")
    summary["age_gender_summary"] = age_gender_df.round(2).to_dict(orient="records")
    summary["payment_summary"] = pay_df.round(2).to_dict(orient="records")
    summary["top_15_facilities_by_volume"] = fac_county_df.head(15).round(2).to_dict(orient="records")
    summary["top_15_facilities_by_los_min500"] = (
        fac_county_df[fac_county_df["discharges"] >= 500]
        .sort_values("avg_los", ascending=False)
        .head(15)
        .round(2)
        .to_dict(orient="records")
    )
    summary["sample_rows_exported"] = int(len(sample_df))
    summary["sample_parquet_mb"] = round(CLOUD_SAMPLE_PARQUET_PATH.stat().st_size / (1024 * 1024), 2)
    summary["elapsed_seconds"] = round(time.time() - t0, 2)

    with open(CLOUD_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[eda] Built full-dataset EDA summary & cloud aggregates in {summary['elapsed_seconds']}s "
        f"(sample={summary['sample_rows_exported']:,} rows, {summary['sample_parquet_mb']} MB)."
    )
    return summary


if __name__ == "__main__":
    build_cloud_artifacts_and_eda()
