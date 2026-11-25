# Testing & Quality Assurance Report — MediScope AI

All verification tests were executed directly against the implemented source code, processed `2,094,484`-row Parquet dataset, trained CatBoost pipeline, and running Streamlit server.

---

## 1. Automated Pytest Suite

Executed via `pytest tests/ -v`:

| Test File | Test Cases | Scope Verified |
|---|---|---|
| [`tests/test_preprocessing.py`](../tests/test_preprocessing.py) | 4 tests | Currency string parsing, `"120 +"` LOS top-code handling, invalid non-positive LOS filtering, exact duplicate removal, confidentiality redaction imputation, empty DataFrame handling |
| [`tests/test_features.py`](../tests/test_features.py) | 2 tests | Age ordinal, severity numeric, ED & newborn flags, payer count, zero-division protection (`cost_per_day`, `charge_per_day`, `charge_to_cost_ratio`) |
| [`tests/test_prediction.py`](../tests/test_prediction.py) | 3 tests | Missing-field rejection, invalid-category validation, end-to-end prediction using saved `models/los_model_pipeline.joblib` |
| [`tests/test_cloud_mode.py`](../tests/test_cloud_mode.py) | 2 tests | Cloud fallback (`80,000`-row stratified sample), KPI calculation, presence and non-emptiness of all 6 pre-aggregated Parquet tables |

---

## 2. Pipeline & Artifact Verification

1. **Data Loading & Preprocessing (`python -m src.preprocessing`):**
   - Processed `2,101,588` raw rows across 9 chunks (`250,000` rows/chunk) in `48.69s`.
   - Removed `7,104` exact duplicates (`0` invalid numeric drops); wrote `data/processed/discharges_cleaned.parquet` (`84.66 MB`).
2. **EDA & Cloud Aggregate Generation (`python -m src.eda`):**
   - Generated all 6 full-dataset aggregate Parquet files, `eda_summary.json`, and `80,000`-row stratified `discharges_sample.parquet` (`3.86 MB`) in `5.51s`.
3. **Model Training & Evaluation (`python -m src.train_model`):**
   - Trained Dummy, Ridge, Random Forest, and CatBoost on `1,675,587` train rows and evaluated on `418,897` unseen test rows.
   - Selected CatBoost Regressor (`1.24 MB` artifact; `MAE = 3.2544d`, `RMSE = 6.8006d`, `R² = 0.3533`).

---

## 3. Streamlit Multipage & UI Testing (`AppTest` + Headless Chromium)

1. **`streamlit.testing.v1.AppTest` Across All 9 Scripts:**
   - Verified `app.py` and `pages/1_Overview.py` through `pages/8_LOS_Predictor.py` render with `0` exceptions and `<1.1s` cached page execution (except initial Cold-Load and full Box Plot render).
   - Tested global sidebar filter selection (`Hospital County = Manhattan`, `Age Group = 0 to 17`) with `0` exceptions.
   - Tested interactive **Predict Length of Stay** form submission on Page 8 (`0` exceptions, validated prediction returned).
2. **Live Browser Rendering & Visual Inspection (`Playwright + Chromium`):**
   - Rendered all 8 dashboard pages on `http://localhost:8509`, visually inspected every screenshot in `screenshots/`, detected and resolved a markdown `</div>` header indentation glitch and clipped 2-column chart titles, then re-verified clean rendering.

---

## 4. Deployment & GitHub Size Check

- Verified `.gitignore` excludes raw `*.csv` (`793.81 MiB`) and `data/processed/*.parquet` (`84.66 MB`).
- Verified cloud-committed artifacts (`data/aggregates/*` + `models/*`) total under `6 MB`, well within GitHub (`100 MB` file limit) and Streamlit Community Cloud free-tier limits.
