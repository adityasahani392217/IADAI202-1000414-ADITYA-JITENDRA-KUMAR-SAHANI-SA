# Data Directory

## Dataset Source
This project uses the **New York State Department of Health — Hospital Inpatient Discharges (SPARCS De-Identified): 2021** dataset:
- File: `Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv`
- Publisher: New York State Department of Health (Health Data NY / SPARCS)

## How to Supply the Raw Dataset Locally
Due to GitHub's 100 MB file-size limit, the raw CSV (~832 MB) is excluded from version control via `.gitignore`.

To run the full local pipeline:
1. Download the 2021 SPARCS De-Identified Hospital Inpatient Discharges CSV from Health Data NY (or place your supplied CSV file in the project root or `data/raw/`).
2. Optionally set the `MEDISCOPE_DATA_PATH` environment variable to point to the CSV path.
3. Run the artifact preparation and model training scripts:
   ```bash
   python -m src.eda
   python -m src.train_model
   ```

## Cloud / Pre-Aggregated Mode
`data/aggregates/` stores compact, pre-aggregated analytical tables and a stratified sample generated from the full dataset so that the Streamlit Community Cloud deployment remains fast and functional within free-tier memory/storage limits.
