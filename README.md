# F1 Tyre Degradation

How much slower does an F1 car get with every lap on a tyre, and how fast is a lap going to be? Built from scratch with NumPy and pandas on FastF1 race data (2022 to 2025). No scikit-learn, XGBoost or PyTorch.

![Predictor](report/app_predictor.png)

## What it does

- **Lap-time model.** Gradient boosting (pinball loss, written in `f1tyre/numpy_ml.py`) predicts a lap relative to the race's field-median lap, in three tiers: pre-lap, pre-stint and live.
- **Degradation model.** Within-stint least squares compares every stint only with itself, so driver, car and strategy cancel out. It gives the compound and circuit curves in the app.
- **Circuit characteristics.** Thirteen measures (speed, braking, cornering load, elevation and others) taken from each circuit's fastest-lap telemetry. Cross-validation kept one: elevation change per km.
- **App.** Predictor (tyre curves with a band for the average or for single stints and a fuel-burn view, and a lap-time predictor that runs the pre-lap and live gradient boosting models on laps you type, paste or upload), Race explorer (real stints against the model), Circuits (ranking) and Model (scores and circuit characteristics).

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

Soft wears about 1.7 times faster than hard, in every season. The curve is a parabola because it reproduced the observed tyre-age curve best on held-out stints (error 0.136 s against 0.145 to 0.147 s for flexible shapes). It is the average lap: over all 3,353 stints it sits 13% below the average stint slope, because long stints dominate the fit and are the gentle ones, and single stints differ a lot.

**Circuit characteristic.** Elevation change per km correlates 0.59 with measured circuit severity (p = 0.02 after allowing for picking the best of 13 characteristics, and 0.55 to 0.66 when any one circuit is dropped). No physical mechanism is proven.

Full tables, bootstrap intervals and ablations: `results/summary.md` and `report/report.pdf`.

## Run it

```
pip install -r requirements.txt
streamlit run app.py          # the app, loads saved models from artifacts/ in about two seconds
python evaluate.py            # lap-level scores (about 10 minutes)
python evaluate_degradation.py   # degradation model and circuit analysis (about 2 minutes)
python build_app.py           # refit and save everything the app loads (about one minute)
```

Run `evaluate_degradation.py` before `build_app.py`. The data is in `data/`, so no F1 account is needed. To download it again, install `fastf1`, copy `.env.example` to `.env`, add an F1 account and run `python fetch_data.py`, then `python fetch_data.py circuits` for the circuit characteristics.

## Layout

```
app.py                      Streamlit app
build_app.py                trains and saves the models and data the app loads
evaluate.py                 lap-level evaluation
evaluate_degradation.py     degradation model and circuit analysis
fetch_data.py               downloads laps and circuit characteristics from FastF1
f1tyre/                     config, features, models, numpy_ml, degradation, serving
artifacts/                  saved models and data for the app
data/                       race laps (92 CSVs) and circuits.csv
results/                    metrics, figures, summary
report/                     report.pdf and editable report.docx, with build scripts
```

Lap data comes from FastF1 (Formula 1 timing data). Keep the repository private.
