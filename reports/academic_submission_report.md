# MediScope AI — Hospital Inpatient Discharge Analytics & Length-of-Stay Prediction

## Academic Summative Assessment Submission Report

| Field | Details |
|---|---|
| **Student Full Name** | Aditya Jitendra Kumar Sahani |
| **Candidate Registration Number** | 1000414 |
| **CRS / Programme Name** | Artificial Intelligence |
| **Course Name** | Mathematics for AI-II (`IADAI202`) |
| **School Name** | `[Insert School Name]` |
| **Assignment Scenario** | Scenario 2 — Hospital Inpatient Discharges Dashboard (*MediScope Health Analytics*) |
| **GitHub Repository Link** | [https://github.com/adityasahani392217/IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA](https://github.com/adityasahani392217/IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA) |
| **Deployed Streamlit App Link** | `[Insert Live Streamlit Community Cloud URL]` |

---

## 1. Introduction

Hospital operations management depends on accurate visibility into patient case-mix, clinical severity, stay duration, payer reimbursement mix, and regional bed utilisation. **MediScope AI** is an integrated healthcare analytics and machine learning platform built for **MediScope Health Analytics** using **2,101,588 New York State SPARCS de-identified inpatient discharge records (2021)**. The system unifies a reproducible chunked preprocessing pipeline, non-parametric statistical analysis across five research questions, a leak-free **CatBoost Regressor** length-of-stay model, and an interactive **8-page Streamlit + Plotly dashboard**.

---

## 2. Problem Statement

Healthcare administrators face severe right-skewness in inpatient stay duration and hospital charges: a relatively small cohort of high-acuity, complex diagnoses drives a disproportionate share of bed-days and financial expenditure. Without empirical segmentation by diagnosis, illness severity, age/gender, hospital peer group, and payer typology—and without admission-stage operational stay forecasting—hospitals struggle to allocate staffing, inpatient beds, and post-acute discharge transitions effectively.

---

## 3. Research Questions

1. **RQ1:** Which diagnosis categories are associated with the longest average inpatient hospital stays?
2. **RQ2:** How do patient age groups and gender relate to hospital charges?
3. **RQ3:** How does the severity of patient illness influence hospital charges, hospital costs, and length of stay?
4. **RQ4:** Which hospitals and counties demonstrate the highest average inpatient stay durations and hospital utilisation?
5. **RQ5:** How do different payment categories relate to hospital charges and inpatient discharge patterns?

Detailed statistical tables, external literature comparisons, and limitations for every question are documented in [`reports/research_questions.md`](research_questions.md).

---

## 4. Dataset Description

- **Source:** New York State Department of Health — Statewide Planning and Research Cooperative System (**SPARCS**), De-Identified Hospital Inpatient Discharges (2021).
- **Raw File Size:** `793.81 MiB` (`832,373,138 bytes`), `2,101,588` rows × `33` columns.
- **Data Quality Highlights:**
  - `7,104` exact duplicate rows (`0.34%`) detected and removed -> **`2,094,484` unique discharges**.
  - `1,561` raw (`1,559` deduplicated) records top-coded as `"120 +"` for Length of Stay.
  - `10,642` confidentiality-redacted records (`Facility Name == "Redacted for Confidentiality"`) missing county/facility IDs.
  - `Birth Weight` missing in `90.16%` of rows (populated exclusively for `202,985` deduplicated newborn stays).

---

## 5. Data Preprocessing & Feature Engineering

Implemented modularly in [`src/data_loader.py`](../src/data_loader.py), [`src/preprocessing.py`](../src/preprocessing.py), and [`src/feature_engineering.py`](../src/feature_engineering.py):
- **Chunked Ingestion (`250,000` rows/chunk)** with explicit string typing prior to validation.
- **Monetary Conversion:** Comma-formatted `Total Charges` and `Total Costs` parsed into `float64` (zero invalid or `<= 0` records).
- **Top-Coded LOS Handling:** `"120 +"` mapped to `120.0` alongside boolean flag `is_los_topcoded`.
- **Missingness Imputation:** Domain-preserving labels (`"Redacted / Unknown"`, `"Not Applicable"`, `"None"` for absent secondary/tertiary payers, `"No Procedure Recorded"`).
- **Engineered Features:** `age_ordinal`, `severity_numeric`, `is_ed_admission`, `is_newborn`, `payer_count`, zero-division-guarded `cost_per_day`, `charge_per_day`, `charge_to_cost_ratio`, and Tukey IQR outlier flag `los_iqr_outlier` (`207,955` stays > `12.0` days retained and flagged rather than arbitrarily deleted).

---

## 6. Exploratory Data Analysis

Key metrics computed across the full `2,094,484`-record cleaned dataset ([`reports/eda_report.md`](eda_report.md)):
- **Facilities & Geography:** `205` facilities across `58` county buckets.
- **Length of Stay:** Mean `5.75 days` (SD `8.42d`), Median `3.0 days` (IQR `4.0d`), Skewness `5.833`.
- **Financials:** Mean Charges `$73,458.85` (Median `$38,526.63`, Total `$153.86B`); Mean Costs `$22,033.83` (Median `$11,622.84`, Total `$46.15B`).
- **Care Mix:** `62.71%` Emergency Department admissions; `23.53%` Surgical vs. `76.35%` Medical.
- **Statistical Tests & Correlations:** Strong Spearman correlation between LOS and Total Costs ($\rho = 0.6746$) and between Severity and LOS ($\rho = 0.4804$). Kruskal-Wallis tests across severity tiers ($H = 365,804.69$ for charges; $H = 498,275.97$ for LOS) and payment categories ($H = 127,989.62$) confirm $p < 10^{-300}$.

---

## 7. Machine Learning Methodology

- **Target Variable:** `length_of_stay` (`1`–`120` days, supervised regression).
- **Leakage Controls:** Excluded `Total Charges`, `Total Costs`, `Patient Disposition`, `CCSR Procedure Code`, `APR DRG Code`, and protected demographic variables (`Race`, `Ethnicity`).
- **Validation Protocol:** Stratified 80:20 train-test split (`1,675,587` train / `418,897` unseen test, `random_state=42`). All categorical encoders were fitted inside `sklearn.pipeline.Pipeline` strictly on training data.

---

## 8. Model Comparison & 9. Evaluation Results

| Model | Train MAE (d) | Test MAE (d) | Train RMSE (d) | Test RMSE (d) | Test R² |
|---|---:|---:|---:|---:|---:|
| Dummy Regressor (Mean Baseline) | 4.5710 | 4.5710 | 8.4130 | 8.4567 | 0.0000 |
| Ridge Regression (`alpha=5.0`) | 3.5711 | 3.5700 | 7.2678 | 7.3021 | 0.2544 |
| Random Forest Regressor | 3.2998 | 3.3005 | 6.8663 | 6.9064 | 0.3330 |
| **CatBoost Regressor (Selected)** | **3.2551** | **3.2544** | **6.7716** | **6.8006** | **0.3533** |

- **Generalisation:** Train and unseen Test errors match within `0.03` days, confirming zero overfitting.
- **Interpretability:** Permutation importance ranks `apr_severity_code` (`+1.024d` MAE when permuted), `apr_medical_surgical_description` (`+0.939d`), `apr_mdc_code` (`+0.595d`), and `ccsr_diagnosis_code` (`+0.561d`) as top predictive signals.
- **Fairness Audit:** Post-hoc error analysis on `418,897` test records found `-0.452d` mean bias (underprediction) on `Black/African American` stays vs. `+0.186d` on `White` stays despite excluding race from features—underscoring the influence of unmeasured social determinants of health.

---

## 10. Dashboard Screenshots

All eight interactive pages and five compulsory Plotly visualisations are captured in `screenshots/`:
- `screenshots/01_overview.png` — Executive Overview
- `screenshots/02_diagnosis_bar.png` — Compulsory Bar Chart: Average Stay per Diagnosis Code
- `screenshots/03_financial_boxplot.png` — Compulsory Box Plot: Total Charges by Severity
- `screenshots/04_hospital_heatmap.png` — Compulsory Heatmap: Avg Stay by Facility & County
- `screenshots/05_payment_pie.png` — Compulsory Pie Chart: Distribution by Primary Payment Type
- `screenshots/06_los_histogram.png` — Compulsory Histogram: Overall Length of Stay Distribution
- `screenshots/07_ml_performance.png` — ML Performance, Residuals & Fairness
- `screenshots/08_los_predictor.png` — Interactive Length of Stay Predictor

---

## 11. Key Findings

1. **Hematologic oncology (`NEO060`: `22.93d`, `NEO059`: `20.89d`) and psychosis (`MBD001`: `16.38d`, `37,169` stays)** drive the longest sustained bed-day demand among high-volume conditions.
2. **Illness severity escalates resource consumption ~5–6x:** `Extreme` severity (`9.57%` of stays) accounts for `28.41%` of total billed hospital charges (`$218,156` mean vs. `$36,855` for `Minor`).
3. **Age × Gender interaction:** Women aged `18–49` have higher inpatient volume due to maternity care (`median 2.0d`) with lower per-stay charges than men in the same age bracket.
4. **Facility specialisation drives raw LOS extremes:** Specialty LTAC, palliative, psychiatric, and rehabilitation hospitals lead average stay duration (`15–46d`), distinct from acute academic medical centers (`5–7d`).
5. **Public payer dominance:** Medicare (`39.37%`) and Medicaid (`30.71%`) fund `70.08%` of all inpatient discharges.

---

## 12. Recommendations

1. **Separate Capacity Planning by Unit Type:** Model behavioral health (`MBD001`), oncology, and post-acute step-down demand separately from general medical/surgical beds.
2. **Case-Mix Adjust Hospital Benchmarks:** Always stratify hospital utilisation comparisons by APR Severity, ED share, and facility care type.
3. **Target Post-Acute Discharge Bottlenecks:** Invest in skilled-nursing transition coordination for `Major`/`Extreme` Medicare stays (`6.89d` mean LOS) and underserved patient cohorts identified in the fairness audit.

---

## 13. Limitations

- **Single-Year Scope (2021):** Reflects COVID-19 pandemic utilisation patterns; multi-year temporal validation requires additional SPARCS years.
- **Top-Coding & Redactions:** `1,559` stays are right-censored at `120 +` days and `10,642` sensitive records have facility/county redacted.
- **Operational Scope Only:** Predictions support capacity planning (`R² = 0.3533`) and are not clinical guidance.

---

## 14. Conclusion

**MediScope AI** delivers a rigorous, tested, deployment-ready solution integrating chunked big-data preprocessing (`2.10M` records), comprehensive statistical EDA, ethical machine learning, and an intuitive healthcare analytics interface aligned with all assessment criteria.

---

## 15. References

1. Agency for Healthcare Research and Quality (AHRQ). (2021). *Clinical Classifications Software Refined (CCSR) for ICD-10-CM Diagnoses*.
2. Averill, R. F., et al. (2003). *All Patient Refined Diagnosis Related Groups (APR-DRGs): Methodology Overview*. 3M Health Information Systems.
3. Dieleman, J. L., et al. (2020). US Health Care Spending by Payer and Health Condition, 1996–2016. *JAMA*, 323(9), 863–884.
4. McDermott, K. W., Elixhauser, A., & Sun, R. (2017). *Trends in Hospital Inpatient Stays in the United States, 2005–2014*. HCUP Statistical Brief #225, AHRQ.
5. New York State Department of Health. (2023). *SPARCS De-Identified Inpatient Discharges (2021)*. Health Data NY.
6. Wennberg, J. E., Fisher, E. S., Goodman, D. C., & Skinner, J. S. (2008). *The Dartmouth Atlas of Health Care*.
7. Zeller, S., Calma, N., & Stone, A. (2014). Psychiatric emergency boarding and length of stay. *Western Journal of Emergency Medicine*, 15(1), 1–6.

---

## 16. GitHub & Deployment Links

- **GitHub Repository:** [https://github.com/adityasahani392217/IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA](https://github.com/adityasahani392217/IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA)
- **Streamlit Community Cloud App:** `[Insert Live Streamlit Community Cloud URL]`
