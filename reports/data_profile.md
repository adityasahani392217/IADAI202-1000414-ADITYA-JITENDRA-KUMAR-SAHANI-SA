# Dataset Profiling Report — MediScope AI

## 1. Source Verification & File Inventory

| Attribute | Observed Value |
|---|---|
| **File Name** | `Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv` |
| **Source System** | New York State Department of Health — Statewide Planning and Research Cooperative System (SPARCS), De-Identified Inpatient Discharges (2021) |
| **File Size** | 832,373,138 bytes (**793.81 MiB**) |
| **Total Raw Records** | **2,101,588** rows |
| **Schema Width** | **33 columns** (exact match to expected SPARCS 2021 schema) |
| **Discharge Years Covered** | **2021 only** (`2,101,588` rows = `100.0%`) |
| **Multi-File Overlap Check** | Only one raw CSV file is supplied in the workspace; no multi-file concatenation required. |

> **Important Profiling Finding:** Although preliminary project notes estimated ~1,048,576 rows (the legacy Excel `.xlsx` row limit), full chunked inspection verified **2,101,588 total inpatient discharge records**.

---

## 2. Schema, Data Types & Cardinality

All 33 columns were ingested initially as explicit strings (`dtype=str`) in 200,000-row chunks to prevent silent type coercion errors on mixed-format fields (`"120 +"`, `"UNKN"`, comma-formatted currencies).

| # | Column Name | Standardized Name | Unique Values | Missing Count | Missing % | Observed Format / Notes |
|---:|---|---|---:|---:|---:|---|
| 1 | Hospital Service Area | `hospital_service_area` | 8 (+missing) | 10,642 | 0.5064% | Missing exclusively on redacted facilities |
| 2 | Hospital County | `hospital_county` | 57 (+missing) | 10,642 | 0.5064% | 57 NYS counties + redacted category (58 total) |
| 3 | Operating Certificate Number | `operating_certificate_number` | 168 | 12,091 | 0.5753% | 7-digit NYS operating certificate ID |
| 4 | Permanent Facility Id | `permanent_facility_id` | 205 | 10,642 | 0.5064% | Zero-padded 6-char facility identifier |
| 5 | Facility Name | `facility_name` | 205 | 0 | 0.0000% | Includes `"Redacted for Confidentiality"` (10,642 rows) |
| 6 | Age Group | `age_group` | 5 | 0 | 0.0000% | Ordered: `0 to 17`, `18 to 29`, `30 to 49`, `50 to 69`, `70 or Older` |
| 7 | Zip Code - 3 digits | `zip_code_3` | 50 | 45,062 | 2.1442% | 3-digit prefix or `"OOS"` |
| 8 | Gender | `gender` | 3 | 0 | 0.0000% | `F` (1,145,483), `M` (955,949), `U` (156) |
| 9 | Race | `race` | 4 | 0 | 0.0000% | `White`, `Other Race`, `Black/African American`, `Multi-racial` |
| 10 | Ethnicity | `ethnicity` | 4 | 0 | 0.0000% | `Not Span/Hispanic`, `Spanish/Hispanic`, `Unknown`, `Multi-ethnic` |
| 11 | Length of Stay | `length_of_stay` | 120 | 0 | 0.0000% | Numeric `1`–`119` plus **`1,561` rows top-coded as `"120 +"`** |
| 12 | Type of Admission | `type_of_admission` | 6 | 0 | 0.0000% | `Emergency`, `Elective`, `Newborn`, `Urgent`, `Trauma`, `Not Available` |
| 13 | Patient Disposition | `patient_disposition` | 19 | 0 | 0.0000% | Post-discharge destination (excluded from ML features) |
| 14 | Discharge Year | `discharge_year` | 1 | 0 | 0.0000% | Constant `2021` |
| 15 | CCSR Diagnosis Code | `ccsr_diagnosis_code` | 477 | 1,634 | 0.0778% | AHRQ CCSR principal diagnosis code |
| 16 | CCSR Diagnosis Description | `ccsr_diagnosis_description` | 477 | 1,634 | 0.0778% | 1:1 with diagnosis code |
| 17 | CCSR Procedure Code | `ccsr_procedure_code` | 320 | 576,021 | 27.4088% | Missing when no inpatient procedure was coded |
| 18 | CCSR Procedure Description | `ccsr_procedure_description` | 320 | 576,021 | 27.4088% | 1:1 with procedure code |
| 19 | APR DRG Code | `apr_drg_code` | 334 | 0 | 0.0000% | 3M APR-DRG grouping code |
| 20 | APR DRG Description | `apr_drg_description` | 334 | 0 | 0.0000% | 1:1 with DRG code |
| 21 | APR MDC Code | `apr_mdc_code` | 26 | 0 | 0.0000% | Major Diagnostic Category (`00`–`25`) |
| 22 | APR MDC Description | `apr_mdc_description` | 26 | 0 | 0.0000% | 1:1 with MDC code |
| 23 | APR Severity of Illness Code | `apr_severity_code` | 5 | 0 | 0.0000% | `1`–`4` plus `0` (`2,550` rows) |
| 24 | APR Severity of Illness Description | `apr_severity_description` | 4 (+missing) | 2,550 | 0.1213% | Missing whenever `apr_severity_code == "0"` |
| 25 | APR Risk of Mortality | `apr_risk_of_mortality` | 4 (+missing) | 2,550 | 0.1213% | Missing whenever `apr_severity_code == "0"` |
| 26 | APR Medical Surgical Description | `apr_medical_surgical_description` | 3 | 0 | 0.0000% | `Medical`, `Surgical`, `Not Applicable` (`2,550` rows) |
| 27 | Payment Typology 1 | `payment_typology_1` | 9 | 0 | 0.0000% | Primary payer (100% populated) |
| 28 | Payment Typology 2 | `payment_typology_2` | 9 (+missing) | 1,072,245 | 51.0207% | Secondary payer (absent for single-payer stays) |
| 29 | Payment Typology 3 | `payment_typology_3` | 9 (+missing) | 1,768,881 | 84.1688% | Tertiary payer |
| 30 | Birth Weight | `birth_weight` | Numeric + `"UNKN"` | 1,894,731 | 90.1571% | Populated only for `206,857` neonatal stays (`119` `"UNKN"`) |
| 31 | Emergency Department Indicator | `emergency_department_indicator` | 2 | 0 | 0.0000% | `Y` (1,316,237) / `N` (785,351) |
| 32 | Total Charges | `total_charges` | Continuous | 0 | 0.0000% | Comma-formatted string (`"320,922.43"`); 0 values <= $0 |
| 33 | Total Costs | `total_costs` | Continuous | 0 | 0.0000% | Comma-formatted string (`"60,241.34"`); 0 values <= $0 |

---

## 3. Data Quality Findings & Cleaning Decisions

1. **Duplicate Records (`7,104` rows / `0.338%`):**
   - Full-dataset hash and row verification detected `7,104` exact duplicate rows across all 33 columns. These were removed (`keep="first"`), leaving **`2,094,484` unique discharge records** (`reports/preprocessing_audit.json`).
2. **Top-Coded Length of Stay (`1,561` raw / `1,559` deduplicated rows):**
   - `Length of Stay` is integer `1`–`119` for `99.93%` of rows, while `0.07%` are top-coded by NYS SPARCS as `"120 +"` for patient privacy. Rather than dropping complex long-stay patients, `"120 +"` is mapped to `120.0` alongside a boolean indicator `is_los_topcoded = True`.
3. **Monetary Formatting & Range Validation:**
   - `Total Charges` and `Total Costs` contain thousands commas (e.g., `"320,922.43"`) requiring string stripping before `float64` parsing. Zero non-numeric values or `<= 0` amounts were encountered.
   - Raw range (before deduplication): Charges `$0.34`–`$17,935,752.00` (median `$38,421.52`, mean `$73,305.42`); Costs `$0.15`–`$12,311,280.35` (median `$11,595.29`, mean `$21,990.13`).
4. **Confidentiality Redactions (`10,642` rows):**
   - Every row missing `Hospital County`, `Hospital Service Area`, and `Permanent Facility Id` corresponds exactly to `Facility Name == "Redacted for Confidentiality"`. These are labeled `"Redacted / Unknown"` rather than dropped so statewide volume totals remain unbiased.
5. **Unassigned APR Severity (`2,550` rows):**
   - All `2,550` rows missing `APR Severity of Illness Description` and `APR Risk of Mortality` have `APR Severity of Illness Code == "0"` and `APR Medical Surgical Description == "Not Applicable"` (primarily `PRE MDC` `2,078` rows and `PREGNANCY` `466` rows). They are explicitly imputed as `"Not Applicable"`.
6. **Excessive Structural Missingness (`Birth Weight`, `Payment Typology 2/3`, `CCSR Procedure`):**
   - `Birth Weight` (`90.16%` missing) is structurally applicable only to neonatal discharges (`202,985` deduplicated newborn stays) and is isolated into `birth_weight_grams` + `is_newborn` rather than general adult analyses.
   - `Payment Typology 2` (`51.02%` missing) and `Payment Typology 3` (`84.17%` missing) are imputed as `"None"` to preserve payer-count engineering (`payer_count`).
   - `CCSR Procedure Code/Description` (`27.41%` missing) is imputed as `"NONE"` / `"No Procedure Recorded"`.
7. **Outliers:**
   - Tukey's IQR rule (`LOS > Q3 + 1.5 * IQR = 12.0 days`) flags **`207,955` discharges (`9.93%`)** as statistical upper-tail stays. Because prolonged inpatient stays represent genuine high-acuity oncology, psychiatric, neonatal ICU, and critical care demand, **outliers are retained and flagged (`los_iqr_outlier`) rather than deleted**.
