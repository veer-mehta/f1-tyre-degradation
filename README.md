# F1 Tyre Degradation

How much slower does an F1 car get with every lap on a tyre, and how fast is a lap going to be? Built from scratch with NumPy and pandas on FastF1 race data (2022 to 2025). No scikit-learn, XGBoost or PyTorch.

![Predictor](report/app_predictor.png)

## What it does

- **Lap-time model.** Gradient boosting (pinball loss, written in `f1tyre/numpy_ml.py`) predicts a lap relative to the race's field-median lap, in three tiers: pre-lap, pre-stint and live.
- **Degradation model.** Within-stint least squares compares every stint only with itself, so driver, car and strategy cancel out. It gives the compound and circuit curves in the app.
- **Circuit characteristics.** Thirteen measures (speed, braking, cornering load, elevation and others) taken from each circuit's fastest-lap telemetry. Cross-validation kept one: elevation change per km.
- **App.** Predictor (curves with 95% bands and a fuel-burn view), Race explorer (real stints against the model), Circuits (ranking) and Model (scores).

## Results

Unseen 2025 races, trained on 2022 to 2024 (24 races, 18,738 laps):

| Model | MAE (s) | R2 |
|---|---|---|
| Median baseline | 0.875 | 0.00 |
| Formula (fuel and form, fitted) | 0.588 | 0.46 |
| Ridge regression | 0.503 | 0.60 |
| **Gradient boosting, pre-lap** | **0.468** | **0.66** |
| **Gradient boosting, pre-stint** | **0.437** | **0.70** |
| Persistence (last 3 laps) | 0.368 | 0.72 |
| **Gradient boosting, live** | **0.294** | **0.82** |

- **Pre-lap:** everything known at the start of the lap except the driver's own previous laps.
- **Pre-stint:** adds the driver's earlier stints in the same race. On laps in stint 2 or later, MAE falls from 0.500 to 0.457 s.
- **Live:** adds recent laps of the current stint, 20% better than persistence.
- Mixed-year 5-fold cross-validation over 90 races (the main estimate): MAE 0.494 / 0.461 / 0.310 for pre-lap / pre-stint / live.
- 90% prediction intervals (conformal) cover 93% of 2025 laps in every tier.

Degradation on unseen 2025 stints (seconds lost per lap of tyre age):

| Compound | Actual | Predicted |
|---|---|---|
| Soft | 0.067 | 0.064 |
| Medium | 0.050 | 0.049 |
| Hard | 0.038 | 0.038 |

Soft wears about 1.7 times faster than hard, in every season. The curve shape is a parabola because it reproduced the observed tyre-age curve best on held-out stints (error 0.136 s against 0.145 to 0.147 s for flexible shapes). Full tables, bootstrap intervals and ablations: `results/summary.md` and `report/report.pdf`.

## Run it

```
pip install -r requirements.txt
python -m unittest discover -s tests -t .   # tests, including a no-future-leakage check
streamlit run app.py                        # the app
python evaluate.py                          # lap-level scores (about 10 minutes)
python evaluate_degradation.py              # degradation model and circuit analysis
python predict_demo.py "Bahrain Grand Prix" 35
```

The data is in `data/`, so no F1 account is needed. To download it again, install `fastf1`, copy `.env.example` to `.env`, add an F1 account and run `python fetch_data.py` then `python fetch_circuits.py`.

## Layout

```
app.py, evaluate*.py, predict_demo.py, fetch_*.py
f1tyre/      config, data, features, circuits, degradation, numpy_ml, models, metrics
tests/       unit tests
data/        race laps (92 CSVs) and circuits.csv
results/     metrics, figures, summary
report/      report.pdf and editable report.docx, with build scripts
reference_sklearn_xgboost/   earlier scikit-learn and XGBoost version, for comparison
```

## Honest limits

- A single lap carries about 0.33 s of random noise, so lap-level R2 tops out near 0.9. Most pre-lap error is each driver's race-specific offset.
- The target is relative to the field-median lap, which comes from the race itself.
- Circuit effects are weak: with 25 circuits, only elevation change held up, and predicting a circuit's severity from all thirteen fails on unseen circuits.
- The curves describe typical behaviour, not a strategy recommendation. Single stints vary a lot (modelled and actual stint slopes correlate at 0.40), and because the fit weighs every lap equally and long stints are the gentle ones, the line sits about 14% below the average stint slope (soft 0.068 against 0.083 s per lap). The Model tab plots all 3,353 stints. Weighting stints equally removes that gap but fits unseen 2025 stints worse, so it was not adopted.
- Some design choices were made while looking at 2025, so that holdout is slightly optimistic. Cross-validation is the better estimate.

Lap data comes from FastF1 (Formula 1 timing data). Keep the repository private.
