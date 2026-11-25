"""Unit tests for data cleaning and preprocessing logic."""

from __future__ import annotations

import pandas as pd

from src.preprocessing import (
    clean_dataframe,
    parse_length_of_stay,
    parse_monetary_series,
    standardize_columns,
)


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Hospital Service Area": "New York City",
                "Hospital County": "Bronx",
                "Operating Certificate Number": "7000006",
                "Permanent Facility Id": "001169",
                "Facility Name": "Test Hospital A",
                "Age Group": "50 to 69",
                "Zip Code - 3 digits": "104",
                "Gender": "F",
                "Race": "White",
                "Ethnicity": "Not Span/Hispanic",
                "Length of Stay": "120 +",
                "Type of Admission": "Emergency",
                "Patient Disposition": "Home or Self Care",
                "Discharge Year": "2021",
                "CCSR Diagnosis Code": "INF012",
                "CCSR Diagnosis Description": "COVID-19",
                "CCSR Procedure Code": None,
                "CCSR Procedure Description": None,
                "APR DRG Code": "137",
                "APR DRG Description": "MAJOR RESPIRATORY",
                "APR MDC Code": "04",
                "APR MDC Description": "RESPIRATORY",
                "APR Severity of Illness Code": "4",
                "APR Severity of Illness Description": "Extreme",
                "APR Risk of Mortality": "Extreme",
                "APR Medical Surgical Description": "Medical",
                "Payment Typology 1": "Medicare",
                "Payment Typology 2": None,
                "Payment Typology 3": None,
                "Birth Weight": None,
                "Emergency Department Indicator": "Y",
                "Total Charges": "320,922.43",
                "Total Costs": "$60,241.34",
            },
            # Exact duplicate of row 0
            {
                "Hospital Service Area": "New York City",
                "Hospital County": "Bronx",
                "Operating Certificate Number": "7000006",
                "Permanent Facility Id": "001169",
                "Facility Name": "Test Hospital A",
                "Age Group": "50 to 69",
                "Zip Code - 3 digits": "104",
                "Gender": "F",
                "Race": "White",
                "Ethnicity": "Not Span/Hispanic",
                "Length of Stay": "120 +",
                "Type of Admission": "Emergency",
                "Patient Disposition": "Home or Self Care",
                "Discharge Year": "2021",
                "CCSR Diagnosis Code": "INF012",
                "CCSR Diagnosis Description": "COVID-19",
                "CCSR Procedure Code": None,
                "CCSR Procedure Description": None,
                "APR DRG Code": "137",
                "APR DRG Description": "MAJOR RESPIRATORY",
                "APR MDC Code": "04",
                "APR MDC Description": "RESPIRATORY",
                "APR Severity of Illness Code": "4",
                "APR Severity of Illness Description": "Extreme",
                "APR Risk of Mortality": "Extreme",
                "APR Medical Surgical Description": "Medical",
                "Payment Typology 1": "Medicare",
                "Payment Typology 2": None,
                "Payment Typology 3": None,
                "Birth Weight": None,
                "Emergency Department Indicator": "Y",
                "Total Charges": "320,922.43",
                "Total Costs": "$60,241.34",
            },
            # Redacted county + newborn + invalid cost row
            {
                "Hospital Service Area": None,
                "Hospital County": None,
                "Operating Certificate Number": None,
                "Permanent Facility Id": None,
                "Facility Name": "Redacted for Confidentiality",
                "Age Group": "0 to 17",
                "Zip Code - 3 digits": None,
                "Gender": "M",
                "Race": "Other Race",
                "Ethnicity": "Unknown",
                "Length of Stay": "3",
                "Type of Admission": "Newborn",
                "Patient Disposition": "Home or Self Care",
                "Discharge Year": "2021",
                "CCSR Diagnosis Code": "PNL001",
                "CCSR Diagnosis Description": "LIVEBORN",
                "CCSR Procedure Code": None,
                "CCSR Procedure Description": None,
                "APR DRG Code": "640",
                "APR DRG Description": "NEONATE",
                "APR MDC Code": "15",
                "APR MDC Description": "NEWBORNS",
                "APR Severity of Illness Code": "1",
                "APR Severity of Illness Description": "Minor",
                "APR Risk of Mortality": "Minor",
                "APR Medical Surgical Description": "Medical",
                "Payment Typology 1": "Medicaid",
                "Payment Typology 2": "Self-Pay",
                "Payment Typology 3": None,
                "Birth Weight": "3200",
                "Emergency Department Indicator": "N",
                "Total Charges": "12,500.00",
                "Total Costs": "4,000.00",
            },
        ]
    )


def test_parse_monetary_series() -> None:
    s = pd.Series(["320,922.43", "$1,200.50", "bad", None])
    out = parse_monetary_series(s)
    assert out.iloc[0] == 320922.43
    assert out.iloc[1] == 1200.50
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])


def test_parse_length_of_stay() -> None:
    s = pd.Series(["5", "120 +", "0", "-3", "invalid"])
    los, top = parse_length_of_stay(s)
    assert los.iloc[0] == 5.0 and not top.iloc[0]
    assert los.iloc[1] == 120.0 and top.iloc[1]
    assert pd.isna(los.iloc[2])
    assert pd.isna(los.iloc[3])
    assert pd.isna(los.iloc[4])


def test_clean_dataframe_dedup_and_imputation() -> None:
    raw = _sample_raw_df()
    cleaned, audit = clean_dataframe(raw, deduplicate=True)
    assert audit["initial_rows"] == 3
    assert audit["exact_duplicates_removed"] == 1
    assert audit["final_rows"] == 2
    assert cleaned["hospital_county"].iloc[1] == "Redacted / Unknown"
    assert cleaned["ccsr_procedure_code"].iloc[0] == "NONE"
    assert cleaned["payment_typology_2"].iloc[0] == "None"
    assert cleaned["birth_weight_grams"].iloc[1] == 3200.0
    assert bool(cleaned["is_los_topcoded"].iloc[0]) is True


def test_standardize_empty_dataframe() -> None:
    empty = pd.DataFrame(columns=["Hospital County", "Total Charges"])
    std = standardize_columns(empty)
    assert list(std.columns) == ["hospital_county", "total_charges"]
