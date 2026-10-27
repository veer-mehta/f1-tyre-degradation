# F1 Tyre Degradation

How much slower does an F1 car get with every lap on a tyre, and how fast is a lap going to be? Built from scratch with NumPy and pandas on FastF1 race data (2022 to 2025). No scikit-learn, XGBoost or PyTorch.

![Predictor](report/app_predictor.png)

## What it does

- **Lap-time model.** Gradient boosting (pinball loss, written in `f1tyre/gradient_boosting.py`) predicts a lap relative to the race's field-median lap, in three tiers: pre-lap, pre-stint and live.
- **Degradation model.** Within-stint least squares compares every stint only with itself, so driver, car and strategy cancel out. It gives the compound and circuit curves in the app.
- **Circuit characteristics.** Thirteen measures (speed, braking, cornering load, elevation and others) taken from each circuit's fastest-lap telemetry. Cross-validation kept one: elevation change per km.
- **App.** Predictor (tyre curves with a band for the average or for single stints and a fuel-burn view, and a lap-time predictor that runs the pre-lap and live gradient boosting models on laps you type, paste or upload), Race explorer (real stints against the model), Circuits (ranking) and Model (scores, a button that re-runs the saved models on the 2025 test laps to verify them, and circuit characteristics).

## What the model uses

**From FastF1** (race laps, 2022 to 2025): driver and team, lap number, lap time, tyre compound, tyre life, new or used tyre, stint number, track status (flags and safety cars), pit-in and pit-out laps, deleted and inaccurate laps, position, the time gaps to the cars ahead and behind (worked out from lap end times and positions), and the weather at each lap (air and track temperature, humidity, wind speed, rain). Circuit elevation comes from the fastest-lap telemetry (position X, Y, Z).

**Target:** lap time minus the median lap time of that race, on clean dry green-flag laps only (no lap 1, pit laps, safety car, yellow flags or rain, and nothing slower than 104% of the driver's best).

**Features for gradient boosting**, built from those fields (every feature uses only information from before the lap):

| Group | Features | Tiers |
|---|---|---|
| Tyre | tyre life, its square and log, new-tyre flag, tyre life of 2 laps or less, compound (soft, medium, hard), tyre life by compound, stint number | all |
| Fuel | lap number, race fraction (lap divided by race length) | all |
| Weather | track temperature, air temperature, humidity, wind speed, track minus air temperature | all |
| Form | driver and team average pace over their previous 3, 5 and 10 races | all |
| Traffic | gaps to the cars ahead and behind, position, and a dirty-air flag (under 1.5 s behind another car), all from the previous lap | all |
| Field context | how the whole field's pace changed recently, laps since the last safety car or yellow flag | all |
| Circuit | elevation change per km | all |
| Earlier stints | driver's mean pace in earlier stints of the race, pace at the end of the previous stint, previous stint length | pre-stint, live |
| Recent laps | previous lap time, mean of the last 3 laps, laps since the previous clean lap | live |

Pre-lap uses the first seven groups, pre-stint adds earlier stints, and live adds recent laps. Pace is always relative to the race median lap.

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

Full tables, bootstrap intervals and ablations: `results/metrics.json` and `report/report.pdf`.

## Run it

```
pip install -r requirements.txt
streamlit run app.py          # the app, loads saved models from cached/ in about two seconds
python evaluate.py            # lap-level scores (about 10 minutes)
python evaluate_degradation.py   # degradation model and circuit analysis (about 2 minutes)
python -m f1tyre.serving      # refit and save everything the app loads (about one minute)
```

Run `evaluate_degradation.py` before `python -m f1tyre.serving`. The data is in `data/`, so no F1 account is needed. To download it again, install `fastf1`, copy `.env.example` to `.env`, add an F1 account and run `python fetch_data.py`, then `python fetch_data.py circuits` for the circuit characteristics.

## Layout

```
app.py                      Streamlit app
evaluate.py                 lap-level evaluation
evaluate_degradation.py     degradation model and circuit analysis
fetch_data.py               downloads laps and circuit characteristics from FastF1
f1tyre/                     config, features, gradient_boosting, models, degradation, serving
cached/                     saved models and data the app loads
data/                       race laps (92 CSVs) and circuits.csv
results/                    metrics, figures, summary
report/                     report.pdf and editable report.docx, with the screenshots they use
```

Lap data comes from FastF1 (Formula 1 timing data). Keep the repository private.
