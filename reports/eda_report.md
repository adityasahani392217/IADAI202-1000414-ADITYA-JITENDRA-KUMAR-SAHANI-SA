# Exploratory Data Analysis Report — MediScope AI

**Dataset:** New York State SPARCS De-Identified Inpatient Discharges (2021)  
**Cleaned Population:** `2,094,484` unique inpatient discharges (`7,104` exact duplicates removed from `2,101,588` raw rows)  
**Source Script:** [`src/eda.py`](../src/eda.py) | **Summary Artifact:** [`data/aggregates/eda_summary.json`](../data/aggregates/eda_summary.json)

---

## 1. Statewide Summary Metrics

| Metric | Value | Notes |
|---|---:|---|
| **Total Discharges** | **2,094,484** | Cleaned unique 2021 stays |
| **Unique Facilities** | **205** | 204 named hospitals + 1 redacted confidentiality bucket |
| **Unique Counties** | **58** | 57 NYS counties + `"Redacted / Unknown"` |
| **Mean Length of Stay** | **5.75 days** | Std Dev: `8.42 days` |
| **Median Length of Stay** | **3.0 days** | IQR: `4.0 days` (`Q1=2.0`, `Q3=6.0`) |
| **Valid Stay Range** | **1 to 120 days** | `1,559` top-coded `120 +` stays |
| **Mean Total Charges** | **$73,458.85** | Total statewide billed charges: **$153.86B** |
| **Median Total Charges** | **$38,526.63** | Heavily right-skewed (`skew = 17.838`) |
| **Mean Total Costs** | **$22,033.83** | Total statewide reported costs: **$46.15B** |
| **Median Total Costs** | **$11,622.84** | Heavily right-skewed (`skew = 27.662`) |
| **Mean Cost per Inpatient Day** | **$4,729.45** | Guarded division (`total_costs / length_of_stay`) |
| **Mean Charge-to-Cost Ratio** | **3.75x** | Ratio of billed charges to reported hospital costs |
| **Emergency Department Admissions** | **62.71%** | `1,313,520` ED (`Y`) vs. `780,964` non-ED (`N`) |
| **Surgical vs. Medical Mix** | **23.53% Surgical** | `1,599,098` Medical (76.35%), `492,838` Surgical (23.53%), `2,548` N/A (0.12%) |

---

## 2. Demographic & Admission Distributions

### 2.1 Age Group Distribution
- **70 or Older:** `618,250` (`29.52%`)
- **50 to 69:** `583,426` (`27.86%`)
- **30 to 49:** `415,208` (`19.82%`)
- **0 to 17:** `285,784` (`13.64%`)
- **18 to 29:** `191,816` (`9.16%`)

Adults aged **50 and older represent 57.38%** of all inpatient hospitalisations in New York State.

### 2.2 Gender Distribution
- **Female (`F`):** `1,141,454` (`54.50%`) — driven partly by obstetric/childbirth admissions (`221,017` pregnancy MDC discharges).
- **Male (`M`):** `952,874` (`45.49%`).
- **Unknown (`U`):** `156` (`0.01%`).

### 2.3 Admission Type Breakdown
- **Emergency:** `1,382,030` (`65.98%`)
- **Elective:** `351,665` (`16.79%`)
- **Newborn:** `193,946` (`9.26%`)
- **Urgent:** `157,993` (`7.54%`)
- **Trauma:** `7,688` (`0.37%`)
- **Not Available:** `1,162` (`0.06%`)

### 2.4 Newborn vs. General Inpatient Cohort
Separating neonatal/newborn stays (`is_newborn == True`, `202,985` stays) from general inpatient stays (`1,891,499` stays) confirms distinct operational profiles:
- **Newborn Cohort:** Mean LOS **3.68 days** (median 2.0d), mean charges **$38,525**, mean costs **$9,890**.
- **General Cohort:** Mean LOS **5.97 days** (median 3.0d), mean charges **$77,208**, mean costs **$23,337**.

---

## 3. Skewness, Outliers, and Correlation Analysis

### 3.1 Skewness
All primary numeric healthcare outcomes are strongly right-skewed:
- `length_of_stay`: **5.833** (mean `5.75d` vs. median `3.0d`)
- `total_charges`: **17.838** (mean `$73,459` vs. median `$38,527`)
- `total_costs`: **27.662** (mean `$22,034` vs. median `$11,623`)

Accordingly, both **means** (for aggregate bed-day and financial budget planning) and **medians / IQRs** (for typical patient experience) are reported throughout MediScope AI, and non-parametric **Spearman rank correlations** and **Kruskal-Wallis H-tests** are used alongside Pearson linear correlations.

### 3.2 Correlation Matrix (Spearman Rank / Pearson Linear)

| Variable Pair | Spearman $\rho$ | Pearson $r$ | Interpretation |
|---|---:|---:|---|
| `total_charges` vs. `total_costs` | **0.8776** | 0.8617 | Strong monotonic co-movement between billed charges and underlying hospital costs |
| `length_of_stay` vs. `total_costs` | **0.6746** | 0.6753 | Longer stays consume substantially more nursing, bed, and ancillary resources |
| `length_of_stay` vs. `total_charges` | **0.6120** | 0.6761 | Billed charges scale strongly with inpatient duration |
| `severity_numeric` vs. `length_of_stay` | **0.4804** | 0.3636 | Higher clinical severity is moderately correlated with longer hospital stays |
| `severity_numeric` vs. `total_costs` | **0.4161** | 0.2848 | Higher acuity aligns with elevated hospital resource cost |
| `age_ordinal` vs. `severity_numeric` | **0.3540** | 0.3507 | Older inpatient cohorts tend to present with greater illness complexity |

*Note: Observed correlations establish statistical association, not causal proof.*

---

## 4. Key Diagnosis, Severity, Facility, and Payer Findings (Linked to RQ1–RQ5)

1. **RQ1 — Diagnosis Stay Duration (`n >= 500`):** Acute Myeloid Leukemia (`NEO060`, avg **22.93d**, median 20d) and Acute Lymphoblastic Leukemia (`NEO059`, avg **20.89d**) lead mean LOS, while Schizophrenia Spectrum Disorders (`MBD001`, avg **16.38d**, **37,169** stays) impose the largest behavioral-health bed-day burden. High-volume drivers are Liveborn (`PNL001`, `195,144`), Septicemia (`INF002`, `137,753`, avg `9.72d`), and COVID-19 (`INF012`, `82,451`, avg `8.64d`).
2. **RQ2 — Age & Gender Charges:** Females aged `18–29` and `30–49` have higher admission volume than males (`137,911` vs. `53,871`; `264,923` vs. `150,258`) due to maternity care, yet substantially lower average charges (`$39,821` vs. `$67,822` in `18–29`). Charges rise steadily with age, peaking at `$93,301` (males `50–69`) and `$92,671` (males `70+`).
3. **RQ3 — Severity Impact:** Kruskal-Wallis tests confirm massive differences across severity tiers ($H = 365,804.69, p < 0.0001$ for charges; $H = 498,275.97, p < 0.0001$ for LOS). Mean charges escalate from **$36,855** (`Minor`) to **$218,156** (`Extreme`), and mean LOS rises from **2.87 days** (`Minor`) to **14.56 days** (`Extreme`).
4. **RQ4 — Geography & Facilities:** Manhattan generates the highest inpatient volume (`364,013` discharges; `2,135,450` patient-days; avg charges `$117,244`), while Mount Sinai Hospital (`49,522` discharges) and North Shore University Hospital (`49,112`) lead facility volume. Longest average stays (`n >= 500`) occur at long-term acute, palliative, psychiatric, and rehabilitation hospitals (Henry J. Carter Specialty Hospital: `45.82d`; Calvary Hospital: `23.24d`).
5. **RQ5 — Payers:** **Medicare** (`39.37%`) and **Medicaid** (`30.71%`) fund `70.08%` of all inpatient discharges. Medicare stays show the highest primary-payer mean LOS (`6.89d`) and average `1.99` active payment fields per stay.
