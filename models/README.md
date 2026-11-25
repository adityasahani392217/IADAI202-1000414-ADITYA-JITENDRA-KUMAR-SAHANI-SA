# Models Directory

This directory stores the trained machine learning artifacts for **MediScope AI — Length of Stay Predictor**.

## Generated Artifacts
- `los_model_pipeline.joblib`: Serialised scikit-learn / CatBoost preprocessing and regression pipeline trained on the prepared training split.
- `model_metadata.json`: Exact feature schema, allowed categorical values, training/test sample sizes, validation split details, and evaluation metrics (`MAE`, `RMSE`, `R²`).

## Training Command
Run from the project root:
```bash
python -m src.train_model
```
