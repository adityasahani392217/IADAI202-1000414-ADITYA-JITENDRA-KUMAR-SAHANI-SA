# MediScope AI — Hospital Inpatient Discharge Analytics & Length-of-Stay Prediction

**Academic Summative Assessment | Course:** Mathematics for AI-II (Artificial Intelligence Programme)  
**Scenario:** Scenario 2 — Hospital Inpatient Discharges Dashboard (*MediScope Health Analytics*)  
**Repository Naming Convention:** `IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA`
**Student:** Aditya Jitendra Kumar Sahani (`1000414`)

---

## 1. Project Overview

**MediScope AI** is an end-to-end healthcare analytics and machine learning platform designed to help hospital administrators, clinicians, and operations managers understand inpatient discharge patterns, clinical severity drivers, hospital utilisation, payer mixes, financial costs, and predicted **Length of Stay (LOS)** for operational bed-capacity planning.

Designed around a clean clinical interface (**Teal `#0D9488`**, **Navy `#10234B`**, **Light Teal `#E6F5F4`**, **White `#FFFFFF`**, **Grey `#64748B`**), the platform combines a chunked preprocessing pipeline across **2.10 million New York State SPARCS inpatient records**, rigorous non-parametric statistical testing, a zero-leakage **CatBoost Regressor** pipeline, and an interactive **8-page Streamlit application**.

---

## 2. Problem Statement & Objectives

Hospital inpatient capacity and healthcare expenditures are disproportionately concentrated in high-acuity and long-stay admissions. Without structured operational analytics, health systems struggle to anticipate bed bottlenecks, benchmark hospital utilisation fairly across peer groups, and forecast inpatient bed-days upon admission.

**Objectives:**
1. Profile and validate the complete raw inpatient discharge dataset without silent assumptions.
2. Design a modular, memory-safe preprocessing and feature engineering pipeline preserving raw data integrity.
3. Answer five domain-grounded research questions (`RQ1`–`RQ5`) combining dataset analytics with external healthcare literature.
4. Build and benchmark leak-free regression models (`Dummy`, `Ridge`, `Random Forest`, `CatBoost`) to predict Length of Stay.
5. Deliver a responsive, interactive 8-page Plotly + Streamlit dashboard deployable on **Streamlit Community Cloud**.

---

## 3. Dataset Description & Licensing

- **Source File:** `Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv` (`793.81 MiB`)
- **Publisher:** New York State Department of Health — Statewide Planning and Research Cooperative System (**SPARCS**), Health Data NY Open Data Portal.
- **License / Terms:** Publicly released de-identified administrative health data governed by New York State Department of Health SPARCS De-Identified Data Terms of Use (no re-identification attempts permitted).
- **Verified Volume:** **2,101,588 raw records** across **33 columns** (`Discharge Year = 2021`). After removing `7,104` exact duplicate rows (`0.34%`), the cleaned analytical dataset comprises **`2,094,484` unique inpatient discharges** across **205 facilities** and **58 counties** (including 1 confidentiality-redacted bucket).

> **GitHub & Cloud Storage Note:** Because the raw CSV (`793.81 MiB`) exceeds GitHub's 100 MB file limit, raw CSVs and the full local Parquet (`84.66 MB`) are excluded via `.gitignore`. Compact pre-aggregated full-dataset tables and an 80,000-row stratified sample (`3.86 MB`) reside in `data/aggregates/` so the deployed Streamlit Community Cloud app remains fast and accurate within free-tier memory constraints.

---

## 4. Research Questions

1. **RQ1:** Which diagnosis categories are associated with the longest average inpatient hospital stays?
2. **RQ2:** How do patient age groups and gender relate to hospital charges?
3. **RQ3:** How does the severity of patient illness influence hospital charges, hospital costs, and length of stay?
4. **RQ4:** Which hospitals and counties demonstrate the highest average inpatient stay durations and hospital utilisation?
5. **RQ5:** How do different payment categories relate to hospital charges and inpatient discharge patterns?

Full statistical findings, external references, and administrative implications are documented in [`reports/research_questions.md`](reports/research_questions.md).

---

## 5. Technologies Used

- **Language:** Python 3.13
- **Data Processing:** Pandas, NumPy, PyArrow, SciPy
- **Machine Learning:** Scikit-learn, CatBoost, Joblib
- **Visualisation & UI:** Plotly, Streamlit
- **Testing & QA:** Pytest, `streamlit.testing.v1.AppTest`
- **Deployment:** GitHub, Streamlit Community Cloud

---

## 6. Architecture

```text
MediScope-AI/
├── app.py                          # Main Streamlit launchpad & global KPI summary
├── .streamlit/config.toml          # LogiSight Teal/Navy theme tokens
├── src/
│   ├── config.py                   # Paths, palette tokens, schema maps, ML feature definitions
│   ├── data_loader.py              # Chunked CSV reader & dual-mode (local/cloud) Parquet loader
│   ├── preprocessing.py            # Deduplication, currency & '120 +' LOS parsing, missingness handling
│   ├── feature_engineering.py      # Ratios, zero-division guards, IQR outlier flags, newborn flags
│   ├── eda.py                      # Full-dataset KPIs, Spearman/Kruskal tests, cloud aggregate builder
│   ├── train_model.py              # Stratified 80:20 training, comparison & artifact export
│   ├── evaluation.py               # MAE/RMSE/R², permutation importance & fairness audit
│   ├── prediction.py               # Input validation & inference wrapper
│   ├── visualizations.py           # Compulsory & diagnostic Plotly charts
│   ├── dashboard_state.py          # Cached loaders, global sidebar filters & CSV export
│   └── utils.py                    # CSS styling, KPI cards & formatting helpers
├── pages/                          # 8 interactive dashboard modules
├── models/                         # Saved CatBoost pipeline (1.24 MB) & evaluation metadata
├── data/                           # Local raw/processed instructions + cloud aggregates
├── reports/                        # Data profile, EDA, RQ1-RQ5, ML evaluation & academic report
├── notebooks/exploratory_analysis.ipynb
├── tests/                          # Pytest unit & integration tests
└── screenshots/                    # Verified application screenshots
```

---

## 7. Data Preprocessing & Feature Engineering Methodology

- **Chunked Ingestion (`250,000` rows/chunk):** Prevents memory spikes on the 793.81 MiB CSV.
- **Exact Deduplication:** Removed `7,104` exact duplicates across all 33 columns (`2,094,484` retained).
- **Length of Stay Validation:** Parsed `1`–`119` integer stays and mapped `1,559` top-coded `"120 +"` records to `120.0` alongside a boolean `is_los_topcoded` flag.
- **Monetary Parsing:** Stripped comma/dollar formatting on `Total Charges` and `Total Costs` into `float64` (zero missing or `<= 0` records).
- **Missingness & Confidentiality:** Labeled `10,642` privacy-redacted county/facility IDs as `"Redacted / Unknown"`; mapped `2,550` `APR Severity Code == 0` missing descriptions to `"Not Applicable"`; isolated `Birth Weight` (`90.16%` missing) into `birth_weight_grams` + `is_newborn` rather than polluting adult analyses.
- **Engineered Features:** `age_ordinal`, `severity_numeric`, `is_ed_admission`, `is_newborn`, `payer_count`, `cost_per_day`, `charge_per_day`, `charge_to_cost_ratio` (with zero-division guards), and `los_iqr_outlier` (retained rather than arbitrarily deleted).

---

## 8. Exploratory Data Analysis Highlights

- **Statewide Volume:** `2,094,484` discharges across `205` facilities and `58` counties (`62.71%` Emergency Department admissions; `23.53%` Surgical).
- **Skewness:** Length of Stay (`mean = 5.75d`, `median = 3.0d`, `skew = 5.83`), Total Charges (`mean = $73,459`, `median = $38,527`, `skew = 17.84`), and Total Costs (`mean = $22,034`, `median = $11,623`, `skew = 27.66`).
- **Severity Impact:** Kruskal-Wallis tests confirm highly significant separation across APR severity tiers ($H = 365,804.69$ for charges; $H = 498,275.97$ for LOS, $p < 10^{-300}$), with mean LOS rising from `2.87d` (`Minor`) to `14.56d` (`Extreme`).

Full results are in [`reports/eda_report.md`](reports/eda_report.md).

---

## 9. Machine Learning Methodology & Evaluation

- **Target:** `length_of_stay` (regression, 1–120 days).
- **Leakage Controls:** Strictly excluded `Total Charges`, `Total Costs`, `Patient Disposition`, `CCSR Procedure Code`, `APR DRG Code`, and protected demographic attributes (`Race`, `Ethnicity`).
- **Validation:** Stratified 80:20 split (`1,675,587` train / `418,897` unseen test) with all encoders fitted solely inside `Pipeline` on `X_train`.

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| Dummy Regressor (Mean Baseline) | 4.5710d | 4.5710d | 8.4130d | 8.4567d | 0.0000 |
| Ridge Regression | 3.5711d | 3.5700d | 7.2678d | 7.3021d | 0.2544 |
| Random Forest Regressor | 3.2998d | 3.3005d | 6.8663d | 6.9064d | 0.3330 |
| **CatBoost Regressor (Selected)** | **3.2551d** | **3.2544d** | **6.7716d** | **6.8006d** | **0.3533** |

Full diagnostics, permutation importance, admission-only sensitivity analysis, and subgroup fairness audits are documented in [`reports/model_evaluation.md`](reports/model_evaluation.md).

---

## 10. Dashboard Features (8 Pages + 5 Compulsory Charts)

1. **Overview:** Statewide KPI cards, dynamic executive summary, admission/surgical mix.
2. **Diagnosis Analytics:** **Compulsory Bar Chart** (Avg stay per diagnosis code), top volume/charge diagnoses, severity stack.
3. **Financial Analytics:** **Compulsory Box Plot** (Total charges segmented by severity with log-scale toggle), charges vs. costs, age × gender charges.
4. **Hospital Performance:** **Compulsory Heatmap** (Avg stay duration by facility & county), county utilisation, facility volume vs. LOS scatter.
5. **Payment Analytics:** **Compulsory Pie Chart** (Patient distribution by primary payment type), primary vs. secondary coverage.
6. **Length of Stay:** **Compulsory Histogram** (Overall LOS distribution with mean/median lines), severity stay comparison.
7. **ML Performance:** Model benchmarks, actual vs. predicted scatter, residual plots, permutation importance, fairness audit.
8. **LOS Predictor:** Validated interactive prediction interface using `models/los_model_pipeline.joblib`.

---

## 11. Setup, Training & Running Locally

### Installation
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd IADAI102(Student_ID)-StudentName
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### Supplying the Raw Dataset & Running Offline Pipelines
1. Place `Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv` in `data/raw/` or the project root (or set `MEDISCOPE_DATA_PATH`).
2. Run preprocessing, EDA aggregate generation, and ML training:
   ```bash
   python -m src.preprocessing
   python -m src.eda
   python -m src.train_model
   ```
3. Run automated tests:
   ```bash
   pytest tests/ -v
   ```
4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 12. GitHub & Streamlit Community Cloud Deployment

1. Create a GitHub repository named `IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA`.
2. Verify `.gitignore` excludes raw `*.csv` files and `data/processed/*.parquet` while preserving `data/aggregates/*` and `models/*`.
3. Push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: MediScope AI inpatient analytics platform"
   git branch -M main
   git remote add origin https://github.com/adityasahani392217/IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA.git
   git push -u origin main
   ```
4. Deploy on [Streamlit Community Cloud](https://share.streamlit.io/):
   - Select your GitHub repository, branch `main`, and main file path `app.py`.
   - Click **Deploy**.
5. **Links:**
   - **GitHub Repository URL:** [https://github.com/adityasahani392217/IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA](https://github.com/adityasahani392217/IADAI202-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA)
   - **Live Streamlit App URL:** `[Insert Live Streamlit Community Cloud URL Here]`

---

## 13. Screenshots

Verified screenshots of the running MediScope AI dashboard are saved in [`screenshots/`](screenshots/):
- `01_overview.png` — Home / Executive Overview
- `02_diagnosis_bar.png` — Diagnosis Analytics (Compulsory Bar Chart)
- `03_financial_boxplot.png` — Financial Analytics (Compulsory Severity Box Plot)
- `04_hospital_heatmap.png` — Hospital Performance (Compulsory Facility × County Heatmap)
- `05_payment_pie.png` — Payment Analytics (Compulsory Payer Pie Chart)
- `06_los_histogram.png` — Length of Stay (Compulsory Histogram)
- `07_ml_performance.png` — ML Performance & Fairness Audit
- `08_los_predictor.png` — Interactive LOS Predictor

---

## 14. Limitations, Ethics & Future Improvements

- **Limitations:** Single-year data (`2021` pandemic period); `120 +` right-censoring; gross billed charges vs. net reimbursement; principal diagnosis only without longitudinal clinical labs.
- **Ethical Considerations:** De-identified public records only; `Race` and `Ethnicity` excluded from predictive inputs; residual subgroup error audit highlights structural disparities (`-0.45d` mean bias for Black/African American stays), requiring human-in-the-loop governance and prohibiting punitive/restrictive use.
- **Future Improvements:** Multi-year SPARCS temporal validation (`2019–2023`), quantile regression intervals (`P10–P90` stay bounds), and HL7 FHIR admission API integration.
