# Machine Learning Evaluation Report — MediScope AI

**Objective:** Supervised regression predicting inpatient **Length of Stay (days)** to support operational hospital bed-capacity planning (not clinical diagnosis or treatment).  
**Training Script:** [`src/train_model.py`](../src/train_model.py) | **Model Artifact:** [`models/los_model_pipeline.joblib`](../models/los_model_pipeline.joblib) (`1.24 MB`)

---

## 1. Feature Selection, Timing & Leakage Prevention

### 1.1 Strictly Excluded Columns (Zero Target Leakage)
The following variables were deliberately excluded because they become known only during or after discharge, or encode target statistics:
- `Total Charges`, `Total Costs`, `cost_per_day`, `charge_per_day`, `charge_to_cost_ratio` (direct functions of stay length and billing)
- `Patient Disposition` (known only upon discharge/transfer/mortality)
- `CCSR Procedure Code` / `Description` (inpatient procedures accumulate across the hospital stay)
- `APR DRG Code` (final DRG incorporates procedures and inpatient complications)
- `Discharge Year` (constant `2021` across all 2,094,484 records)
- `Race` and `Ethnicity` (excluded from predictive inputs to prevent demographic bias; evaluated separately in Section 5 Fairness Audit)

### 1.2 Selected Predictors (9 Features)
1. `age_group` (5 categories)
2. `gender` (3 categories)
3. `type_of_admission` (6 categories)
4. `emergency_department_indicator` (`Y` / `N`)
5. `ccsr_diagnosis_code` (477 primary diagnosis categories; ordinal/native categorical handling inside pipeline)
6. `apr_mdc_code` (26 Major Diagnostic Categories)
7. `apr_medical_surgical_description` (`Medical`, `Surgical`, `Not Applicable`)
8. `apr_severity_code` (`0`–`4` clinical illness severity staging)
9. `apr_risk_of_mortality` (mortality risk tier)

> **Timing Clarification:** Including `apr_severity_code` and `apr_risk_of_mortality` models **acuity-adjusted operational forecasting** once initial severity staging is assigned. To quantify admission-only predictive power before acuity staging, an additional **Admission-Only Sensitivity Experiment** (7 predictors, excluding APR severity and mortality risk) was executed (`MAE = 3.9028d`, `RMSE = 7.8116d`, `R² = 0.1468`).

---

## 2. Validation Strategy

- **Split Protocol:** Stratified 80:20 train-test split (`random_state=42`, stratified on `apr_severity_code`) across the full cleaned dataset:
  - **Training Set:** `1,675,587` inpatient discharges
  - **Unseen Test Set:** `418,897` inpatient discharges
- **Leakage Control:** All categorical encoders (`OneHotEncoder(min_frequency=100)` for Ridge; `OrdinalEncoder` for tree ensembles) were wrapped inside `sklearn.pipeline.Pipeline` and fitted strictly on `X_train`.
- **Chronological Validation Note:** Because the supplied SPARCS dataset contains exclusively `Discharge Year == 2021` (`100%`), cross-year temporal splitting is not applicable.

---

## 3. Model Comparison Results

| Model | Train MAE (d) | Train RMSE (d) | Train R² | Test MAE (d) | Test RMSE (d) | Test R² | Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Dummy Regressor (Mean Baseline)** | 4.5710 | 8.4130 | 0.0000 | 4.5710 | 8.4567 | 0.0000 | 5.69s |
| **Ridge Regression (`alpha=5.0`)** | 3.5711 | 7.2678 | 0.2537 | 3.5700 | 7.3021 | 0.2544 | 18.04s |
| **Random Forest Regressor** | 3.2998 | 6.8663 | 0.3339 | 3.3005 | 6.9064 | 0.3330 | 13.50s |
| **CatBoost Regressor (Selected)** | **3.2551** | **6.7716** | **0.3521** | **3.2544** | **6.8006** | **0.3533** | 222.60s |

### Selection Rationale
**CatBoost Regressor** (`iterations=350`, `depth=7`, `learning_rate=0.08`, native categorical splitting) was selected because:
1. It achieves the lowest **Test MAE (`3.2544` hospital days)** and **Test RMSE (`6.8006` days)**, and highest **Test R² (`0.3533`)**, reducing MAE by **28.8%** relative to the mean baseline.
2. **Zero Overfitting:** Train (`MAE = 3.2551d`, `R² = 0.3521`) and unseen Test (`MAE = 3.2544d`, `R² = 0.3533`) metrics are virtually identical.
3. **Deployment Efficiency:** The serialized pipeline is only **`1.24 MB`**, enabling instant loading on Streamlit Community Cloud without exceeding memory/storage quotas.

---

## 4. Feature Interpretability (Permutation Importance)

Computed via 5-repeat permutation importance (`scoring="neg_mean_absolute_error"`) on `25,000` unseen test discharges:

| Rank | Feature | Mean MAE Increase When Permuted (Days) | Std Dev |
|---:|---|---:|---:|
| 1 | `apr_severity_code` | **1.0240** | 0.0201 |
| 2 | `apr_medical_surgical_description` | **0.9388** | 0.0154 |
| 3 | `apr_mdc_code` | **0.5947** | 0.0134 |
| 4 | `ccsr_diagnosis_code` | **0.5607** | 0.0125 |
| 5 | `apr_risk_of_mortality` | **0.3926** | 0.0075 |
| 6 | `type_of_admission` | **0.3260** | 0.0099 |
| 7 | `age_group` | **0.1185** | 0.0045 |
| 8 | `emergency_department_indicator` | **0.0235** | 0.0013 |
| 9 | `gender` | **0.0017** | 0.0003 |

*Interpretation Note:* Permutation importance measures predictive contribution within the model and does not imply causal mechanisms.

---

## 5. Fairness Audit & Ethical Considerations

Although `Race` and `Ethnicity` were excluded from model inputs, subgroup error analysis on all `418,897` unseen test records highlights important disparities:

| Protected Attribute | Subgroup | Test Count | Mean Actual LOS | Mean Predicted LOS | Mean Bias (Pred − Actual) | MAE (d) |
|---|---|---:|---:|---:|---:|---:|
| **Race** | `Black/African American` | 77,419 | 6.525 | 6.073 | **−0.452** | 3.693 |
| **Race** | `White` | 227,083 | 5.570 | 5.756 | **+0.186** | 3.151 |
| **Race** | `Other Race` | 109,905 | 5.577 | 5.429 | −0.148 | 3.151 |
| **Race** | `Multi-racial` | 4,490 | 5.930 | 5.666 | −0.264 | 3.419 |
| **Ethnicity** | `Not Span/Hispanic` | 313,299 | 5.785 | 5.813 | +0.029 | 3.267 |
| **Ethnicity** | `Spanish/Hispanic` | 64,822 | 5.321 | 5.235 | −0.086 | 2.980 |
| **Gender** | `F` | 228,403 | 5.289 | 5.258 | −0.031 | 2.888 |
| **Gender** | `M` | 190,455 | 6.308 | 6.291 | −0.017 | 3.694 |

**Key Fairness Insight:** Because unmeasured social determinants of health, discharge placement delays (e.g., skilled nursing availability), and hospital resource differences correlate with neighborhood demographics, omitting race from inputs does not automatically eliminate residual bias (`Black/African American` stays are underpredicted by `0.45d` on average). Therefore, MediScope AI predictions must never be used to restrict care duration or discharge eligibility.

---

## 6. Model Limitations

1. **Explained Variance (`R² = 0.3533`):** Without post-admission clinical telemetry, lab trajectories, surgical scheduling, or social-work barriers, roughly 65% of stay variance reflects in-hospital events and external discharge bottlenecks.
2. **Right-Censoring (`120 +`):** Stays >=120 days are top-coded at `120`, compressing extreme long-stay predictions.
3. **Single-Year Scope:** Trained solely on 2021 pandemic-era discharges; recalibration is required before deployment on multi-year post-2021 cohorts.
