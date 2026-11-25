"""Integration tests verifying Streamlit Cloud fallback mode and aggregate tables."""

from __future__ import annotations

import pytest

from src.config import CLOUD_SAMPLE_PARQUET_PATH
from src.data_loader import load_aggregate_table, load_analysis_dataset
from src.eda import compute_core_kpis


@pytest.mark.skipif(
    not CLOUD_SAMPLE_PARQUET_PATH.exists(),
    reason="Cloud sample parquet not yet built.",
)
def test_cloud_sample_loads_and_computes_kpis() -> None:
    df, mode = load_analysis_dataset(prefer_full=False)
    assert mode == "cloud_sample"
    assert len(df) == 80_000
    kpis = compute_core_kpis(df)
    assert kpis["total_discharges"] == 80_000
    assert kpis["avg_los"] > 0
    assert kpis["avg_charges"] > kpis["avg_costs"]


def test_preaggregated_tables_present() -> None:
    for fname in (
        "agg_diagnosis.parquet",
        "agg_severity.parquet",
        "agg_payment.parquet",
        "agg_facility_county.parquet",
        "agg_age_gender.parquet",
        "agg_los_histogram.parquet",
    ):
        tbl = load_aggregate_table(fname)
        assert tbl is not None and not tbl.empty, f"Missing or empty aggregate: {fname}"
