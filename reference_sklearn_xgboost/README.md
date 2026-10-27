# Reference: library-based version (scikit-learn and XGBoost)

The earlier version of the project, kept for comparison. It uses `xgboost` and `scikit-learn` where the main project uses the from-scratch code in `f1tyre/numpy_ml.py`. Everything else (features, cleaning, validation, baselines) is identical.

It reads the shared `data/` folder at the project root: `python reference_sklearn_xgboost/evaluate.py` (run from anywhere). Requires `scikit-learn` and `xgboost`. Results are written to `reference_sklearn_xgboost/results/`.

| 2025 holdout | XGBoost (reference) | From scratch (main project) |
|---|---|---|
| Pre-lap MAE (s) / R2 | 0.4660 / 0.6569 | 0.4646 / 0.6581 |
| Live MAE (s) / R2 | 0.2967 / 0.8162 | 0.2976 / 0.8152 |
| Mixed-year CV pre-lap MAE / R2 | 0.4898 / 0.6461 | 0.4924 / 0.6436 |
| Mixed-year CV live MAE / R2 | 0.3140 / 0.8088 | 0.3156 / 0.8071 |

Only the final library-based pipeline is kept here. The exploratory scripts from earlier iterations (stint-level model, tuning and diagnostic scripts) were removed during the cleanup.
