# Research Questions Analysis — MediScope AI

This document presents the statistical investigation of five core operational and clinical-analytics research questions using the cleaned **2021 New York State SPARCS De-Identified Inpatient Discharges dataset (`N = 2,094,484` records)**. Throughout each section, **empirical findings computed from the supplied dataset** are explicitly separated from **external healthcare and data science literature**.

---

## RQ1: Which diagnosis categories are associated with the longest average inpatient hospital stays?

**Variables:** `CCSR Diagnosis Code`, `CCSR Diagnosis Description`, `Length of Stay`

### Empirical Findings (Computed From 2021 NYS SPARCS Dataset)
To prevent rare single-patient anomalies from distorting rankings, principal diagnosis categories were evaluated both across all codes and with a minimum threshold of `n >= 500` discharges:

| Rank | CCSR Code | Diagnosis Description | Discharges | Mean LOS (Days) | Median LOS (Days) | Mean Charges |
|---:|---|---|---:|---:|---:|---:|
| 1 | `NEO060` | Leukemia — Acute Myeloid Leukemia (AML) | 1,566 | **22.93** | 20.0 | $399,174 |
| 2 | `NEO059` | Leukemia — Acute Lymphoblastic Leukemia (ALL) | 586 | **20.89** | 18.0 | $438,436 |
| 3 | `CIR022` | Sequela of Hemorrhagic Cerebrovascular Disease | 1,126 | **18.54** | 16.0 | $121,603 |
| 4 | `INJ045` | Traumatic Brain Injury (TBI); Subsequent Encounter | 987 | **16.92** | 14.0 | $111,039 |
| 5 | `MBD001` | Schizophrenia Spectrum and Other Psychotic Disorders | 37,169 | **16.38** | 11.0 | $73,476 |
| 6 | `CIR025` | Sequela of Cerebral Infarction / Cerebrovascular Disease | 5,996 | **14.26** | 13.0 | $96,419 |
| 7 | `SKN003` | Pressure Ulcer of Skin | 1,447 | **14.14** | 8.0 | $92,656 |
| 8 | `CIR004` | Endocarditis and Endocardial Disease | 814 | **13.64** | 10.0 | $203,196 |

By **total inpatient volume**, the leading diagnoses are `PNL001` (Liveborn: `195,144` discharges, mean LOS `3.54d`), `INF002` (Septicemia: `137,753` discharges, mean LOS `9.72d`), and `INF012` (COVID-19: `82,451` discharges, mean LOS `8.64d`). Notably, `MBD001` (Schizophrenia Spectrum Disorders) combines high volume (`37,169` discharges) with prolonged stays (`16.38d` mean), generating over **608,800 inpatient bed-days** statewide.

### Plain-Language Explanation & External Literature Context
- **Dataset Interpretation:** Prolonged stays cluster into three clinical profiles: (1) hematologic malignancies requiring multi-week induction chemotherapy and neutropenic monitoring (`NEO060`, `NEO059`), (2) post-acute neuro-rehabilitation and complex sequelae (`CIR022`, `INJ045`), and (3) severe behavioral health admissions (`MBD001`).
- **External Literature:** The Agency for Healthcare Research and Quality (AHRQ) Healthcare Cost and Utilization Project (HCUP) Statistical Briefs similarly identify septicemia, heart failure, and serious mental illness among the costliest and longest-duration U.S. inpatient conditions (McDermott et al., 2017; AHRQ HCUP CCSR documentation, 2021). Behavioral health boarding and limited community step-down beds are widely documented drivers of extended psychiatric LOS (Zeller et al., 2014).

### Administrative Implications & Limitations
- **Implications:** Capacity planners should separate **unit-level bed forecasting** (hematology/oncology, inpatient psychiatry, neuro-rehabilitation) from acute general medical beds, and expand step-down/outpatient psychiatric transition programs to relieve `MBD001` bed bottlenecks.
- **Limitations:** SPARCS captures only the *principal* diagnosis in `CCSR Diagnosis Code`; secondary comorbidities and social barriers to discharge (e.g., housing instability, SNF placement delays) are not directly observable, and stays >=120 days are top-coded at `120`.

---

## RQ2: How do patient age groups and gender relate to hospital charges?

**Variables:** `Age Group`, `Gender`, `Total Charges`

### Empirical Findings (Computed From 2021 NYS SPARCS Dataset)

| Age Group | Female Discharges | Female Mean Charges | Female Median Charges | Male Discharges | Male Mean Charges | Male Median Charges |
|---|---:|---:|---:|---:|---:|---:|
| `0 to 17` | 138,913 | $45,773 | $15,336 | 146,814 | $50,390 | $15,928 |
| `18 to 29` | 137,911 | $39,821 | $26,775 | 53,871 | $67,822 | $35,251 |
| `30 to 49` | 264,923 | $50,839 | $31,935 | 150,258 | $73,378 | $37,877 |
| `50 to 69` | 260,844 | $86,282 | $49,806 | 322,561 | $93,301 | $49,980 |
| `70 or Older` | 338,863 | $82,522 | $52,174 | 279,370 | $92,671 | $55,076 |

*Note: `Gender == 'U'` represents 156 records (`0.007%`).* Overall Spearman rank correlation between `age_ordinal` and `total_charges` is $\rho = 0.3418$ (moderate positive association).

### Plain-Language Explanation & External Literature Context
- **Dataset Interpretation:** Two clear patterns emerge:
  1. **Charges increase markedly with age:** Median charges more than triple from `$15,336–$15,928` (`0–17`) to `$52,174–$55,076` (`70+`), reflecting higher surgical complexity and multi-morbidity in older adults.
  2. **The Reproductive-Age Gender Divergence (`18–49`):** Women aged `18–49` account for more than double the inpatient admissions of men (`402,834` vs. `204,129`), yet incur **29%–41% lower mean charges** (`$39,821` vs. `$67,822` in `18–29`). This reflects high-volume, short-stay (`median 2.0 days`) maternity/childbirth hospitalizations among women versus higher-acuity trauma, behavioral health, and emergency medical/surgical admissions among young adult men.
- **External Literature:** National inpatient expenditure studies by the Centers for Medicare & Medicaid Services (CMS) and Dieleman et al. (*JAMA*, 2020) show steep healthcare spending escalation after age 50, while HCUP maternal hospitalization briefs confirm uncomplicated childbirth drives high female admission volume with lower per-stay cost relative to non-obstetric acute admissions.

### Administrative Implications & Limitations
- **Implications:** Financial budgeting and bundled-payment forecasting must stratify by both **age and clinical service line** (obstetric vs. medical/surgical); age-gender raw comparisons without obstetric adjustment can misrepresent per-patient cost drivers.
- **Limitations:** `Total Charges` reflect hospital chargemaster prices rather than actual negotiated reimbursement; geographic pricing variations across NY counties also influence raw dollar figures.

---

## RQ3: How does the severity of patient illness influence hospital charges, hospital costs, and length of stay?

**Variables:** `APR Severity of Illness Description`, `APR Severity of Illness Code`, `Total Charges`, `Total Costs`, `Length of Stay`

### Empirical Findings (Computed From 2021 NYS SPARCS Dataset)

| APR Severity Tier | Discharges | Mean LOS | Median LOS | Mean Charges | Median Charges | Q1–Q3 Charges | Mean Costs | Median Costs |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `1 — Minor` | 633,790 | **2.87d** | 2.0d | **$36,855** | $24,642 | $13,030 – $45,385 | **$10,930** | $7,574 |
| `2 — Moderate` | 759,229 | **4.81d** | 3.0d | **$54,681** | $35,835 | $19,066 – $65,605 | **$16,624** | $10,786 |
| `3 — Major` | 498,555 | **7.29d** | 5.0d | **$90,493** | $55,809 | $29,059 – $106,370 | **$27,000** | $16,383 |
| `4 — Extreme` | 200,362 | **14.56d** | 10.0d | **$218,156** | $113,437 | $54,806 – $237,171 | **$65,336** | $33,777 |
| `0 — Not Applicable` | 2,548 | 6.00d | 4.0d | $62,383 | $30,910 | $14,455 – $64,107 | $19,134 | $9,931 |

Non-parametric **Kruskal-Wallis H-tests** across the four clinical severity tiers (`1–4`) confirm statistically significant separation for both **Total Charges** ($H = 365,804.69, p < 10^{-300}$) and **Length of Stay** ($H = 498,275.97, p < 10^{-300}$). Spearman rank correlation between `severity_numeric` and `length_of_stay` is $\rho = 0.4804$, and with `total_costs` is $\rho = 0.4161$.

### Plain-Language Explanation & External Literature Context
- **Dataset Interpretation:** Illness severity is the single strongest clinical stratifier of resource intensity in the dataset. Moving from `Minor` to `Extreme` severity multiplies mean Length of Stay by **5.07x** (`2.87d` to `14.56d`), mean charges by **5.92x** (`$36,855` to `$218,156`), and mean reported costs by **5.98x** (`$10,930` to `$65,336`). Although `Extreme` severity comprises only **9.57%** of discharges (`200,362` cases), it consumes **28.41% of total statewide hospital charges** (`$43.71B` of `$153.86B`).
- **External Literature:** 3M Health Information Systems' All Patient Refined Diagnosis Related Groups (APR-DRG) methodology (Averill et al., 2003) was explicitly designed to capture physiological decompensation and multi-system comorbidity burden, and peer-reviewed health-services studies consistently validate strong monotonic resource gradients across APR Severity of Illness levels (Quinn, 2014).

### Administrative Implications & Limitations
- **Implications:** Case-mix adjustment using APR Severity is indispensable when benchmarking hospital cost performance; hospitals serving safety-net or tertiary referral populations with high `Major`/`Extreme` shares will naturally exhibit longer stays and higher per-case costs.
- **Limitations:** APR Severity is assigned using all coded diagnoses during the episode; therefore, severe hospital-acquired complications during a stay can elevate the final severity tier (reverse/bidirectional temporal association).

---

## RQ4: Which hospitals and counties demonstrate the highest average inpatient stay durations and hospital utilisation?

**Variables:** `Facility Name`, `Hospital County`, `Length of Stay`, `Permanent Facility Id`

### Empirical Findings (Computed From 2021 NYS SPARCS Dataset)
- **Top Counties by Inpatient Volume & Bed-Days:**
  - **Manhattan:** `364,013` discharges across `19` facilities (`2,135,450` patient-days; mean LOS `5.87d`; mean charges `$117,244`).
  - **Kings (Brooklyn):** `189,886` discharges (`14` facilities; `1,146,788` patient-days; mean LOS `6.04d`).
  - **Nassau:** `182,783` discharges (`12` facilities; `980,784` patient-days; mean LOS `5.37d`).
  - **Queens:** `169,940` discharges (`10` facilities; `971,452` patient-days; mean LOS `5.72d`).
  - **Bronx:** `162,394` discharges (`10` facilities; `1,011,988` patient-days; mean LOS `6.23d`).
  - **Highest-LOS Large Counties:** **Westchester** (`6.70d` avg across `114,871` discharges) and **Monroe** (`6.50d` avg across `108,228` discharges).
- **Top Facilities by Total Volume:**
  1. **Mount Sinai Hospital** (Manhattan, ID `001456`): `49,522` discharges (`321,510` patient-days, mean LOS `6.49d`).
  2. **North Shore University Hospital** (Nassau, ID `000541`): `49,112` discharges (`271,741` patient-days, mean LOS `5.53d`).
  3. **NYU Langone Hospitals** (Manhattan, ID `001463`): `42,787` discharges (`214,056` patient-days, mean LOS `5.00d`).
- **Top Facilities by Average Stay Duration (`n >= 500` discharges):**
  1. **Henry J. Carter Specialty Hospital** (Manhattan): **45.82 days** (median `37.0d`, `n = 523`, ED share `0.38%`).
  2. **Calvary Hospital Inc** (Bronx): **23.24 days** (median `11.0d`, `n = 2,660`, palliative/hospice care, ED share `0.0%`).
  3. **NYP Westchester Behavioral Health Center** (Westchester): **17.26 days** (median `12.0d`, `n = 3,769`, psychiatric).
  4. **Helen Hayes Hospital** (Rockland): **16.84 days** (median `14.0d`, `n = 1,527`, physical rehabilitation).

### Plain-Language Explanation & External Literature Context
- **Dataset Interpretation:** Raw hospital stay averages are heavily shaped by **facility mission and care specialization**. Specialty long-term acute care (LTAC), palliative, psychiatric, and inpatient rehabilitation facilities exhibit mean stays of `15–46 days` with near-zero ED admissions, whereas major acute academic medical centers (Mount Sinai, NYU Langone, Strong Memorial) process `35,000–50,000` annual discharges with `5.0–7.0d` mean LOS.
- **External Literature:** The Dartmouth Atlas of Health Care and CMS Hospital Compare emphasize risk-standardization and peer-group stratification (acute tertiary vs. specialty/rehabilitation vs. safety-net) before interpreting regional utilisation differences as operational inefficiency (Wennberg et al., 2008).

### Administrative Implications & Limitations
- **Implications:** Health systems should benchmark acute hospitals only against peer acute facilities of comparable severity and ED intake, and use county-level bed-day totals (`Manhattan: 2.14M days`) for regional post-acute SNF and home-health capacity planning.
- **Limitations:** `10,642` sensitive records have facility and county redacted for privacy (`Redacted / Unknown`); moreover, large urban referral centers attract out-of-county complex patients.

---

## RQ5: How do different payment categories relate to hospital charges and inpatient discharge patterns?

**Variables:** `Payment Typology 1`, `Payment Typology 2`, `Payment Typology 3`, `Total Charges`, `Length of Stay`

### Empirical Findings (Computed From 2021 NYS SPARCS Dataset)
Evaluating `Payment Typology 1` as the exclusive primary payer (preventing double-counting):

| Primary Payment Typology | Discharges | Share (%) | Mean Charges | Median Charges | Mean LOS | Median LOS | Mean Active Payers |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Medicare** | 824,504 | **39.37%** | $88,049 | $52,362 | **6.89d** | 4.0d | 1.99 |
| **Medicaid** | 643,229 | **30.71%** | $62,726 | $29,623 | **5.50d** | 3.0d | 1.50 |
| **Private Health Insurance** | 306,175 | **14.62%** | $70,144 | $36,356 | **4.52d** | 2.0d | 1.33 |
| **Blue Cross/Blue Shield** | 223,137 | **10.65%** | $60,835 | $30,525 | **4.24d** | 2.0d | 1.37 |
| **Managed Care, Unspecified** | 28,201 | 1.35% | $49,295 | $26,784 | 4.65d | 3.0d | 1.18 |
| **Self-Pay** | 27,058 | 1.29% | $50,513 | $26,098 | 4.58d | 2.0d | 1.23 |
| **Miscellaneous/Other** | 22,419 | 1.07% | $97,057 | $54,107 | 5.65d | 3.0d | 1.64 |
| **Federal/State/Local/VA** | 18,527 | 0.88% | $45,420 | $23,529 | 5.24d | 3.0d | 1.31 |
| **Department of Corrections** | 1,234 | 0.06% | $72,134 | $35,780 | 5.85d | 4.0d | 1.20 |

A Kruskal-Wallis H-test confirms highly significant charge variation across payment typologies ($H = 127,989.62, p < 10^{-300}$). Secondary coverage (`Payment Typology 2`) is present on `48.98%` of discharges (most frequently Medicaid at `415k+` secondary instances, reflecting Medicare-Medicaid dual-eligible beneficiaries), and tertiary coverage (`Payment Typology 3`) appears on `15.83%` of stays (`Self-Pay` co-insurance most common).

### Plain-Language Explanation & External Literature Context
- **Dataset Interpretation:** Public payers (**Medicare + Medicaid**) cover **70.08%** of all New York State inpatient discharges. Medicare stays show the longest average stay (`6.89d`) and highest average charges among major payers (`$88,049`) because Medicare predominantly covers older (`65+`) and medically complex patients—not because insurance status *causes* longer stays. Commercial payers (`Private Health Insurance` and `Blue Cross/Blue Shield`) cover younger, lower-LOS cohorts (`median 2.0d`).
- **External Literature:** KFF (Kaiser Family Foundation) and AHRQ HCUP payer-mix reports similarly document that Medicare and Medicaid account for roughly two-thirds to 70% of U.S. inpatient stays and drive the largest share of complex chronic inpatient utilization (Freeman et al., 2018; Weiss & Elixhauser, 2014).

### Administrative Implications & Limitations
- **Implications:** Hospital revenue-cycle teams face substantial public-payer concentration (`70%+`), where fixed DRG prospective reimbursements tighten operating margins during prolonged stays (`6.89d` Medicare average). Reducing avoidable delays in post-acute discharge directly protects both bed capacity and financial sustainability.
- **Limitations:** `Total Charges` are gross billed amounts, not net collected revenue; payer-specific negotiated discounts and Medicaid disproportionate-share adjustments are not included in SPARCS de-identified files.

---

## External References
1. Agency for Healthcare Research and Quality (AHRQ). (2021). *Clinical Classifications Software Refined (CCSR) for ICD-10-CM Diagnoses*. Healthcare Cost and Utilization Project (HCUP).
2. Averill, R. F., Goldfield, N., Hughes, J. S., Bonazelli, J., McCullough, E. C., Steinbeck, B. A., et al. (2003). *All Patient Refined Diagnosis Related Groups (APR-DRGs): Methodology Overview*. 3M Health Information Systems.
3. Dieleman, J. L., Cao, J., Chapin, A., Chen, C., Li, Z., Liu, A., et al. (2020). US Health Care Spending by Payer and Health Condition, 1996–2016. *JAMA*, 323(9), 863–884.
4. Freeman, W. J., Weiss, A. J., & Heslin, K. C. (2018). *Overview of U.S. Hospital Stays in 2016: Variation by Geographic Region*. HCUP Statistical Brief #246, AHRQ.
5. McDermott, K. W., Elixhauser, A., & Sun, R. (2017). *Trends in Hospital Inpatient Stays in the United States, 2005–2014*. HCUP Statistical Brief #225, AHRQ.
6. New York State Department of Health. (2023). *Hospital Inpatient Discharges (SPARCS De-Identified): 2021 Data Dictionary and Confidentiality Guidelines*. Health Data NY.
7. Quinn, K. (2014). After the revolution: DRGs at age 30. *Annals of Internal Medicine*, 160(6), 426–429.
8. Wennberg, J. E., Fisher, E. S., Goodman, D. C., & Skinner, J. S. (2008). *Tracking the Care of Patients with Severe Chronic Illness: The Dartmouth Atlas of Health Care*. The Dartmouth Institute.
9. Zeller, S., Calma, N., & Stone, A. (2014). Effects of a dedicated regional psychiatric emergency service on boarding of psychiatric patients in area emergency departments. *Western Journal of Emergency Medicine*, 15(1), 1–6.
