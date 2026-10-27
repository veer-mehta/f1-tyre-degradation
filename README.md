# F1 Tyre Degradation

Predicts how many seconds slower or faster an F1 lap will be than the race's field-median lap from the tyre, fuel load, weather, recent form, traffic and field context, and turns that into tyre-age degradation curves. Trained on FastF1 race data for 2022 to 2025.

The machine learning is implemented from scratch with NumPy (`f1tyre/numpy_ml.py`): metrics, least squares, ridge regression and gradient-boosted trees with a pinball (quantile) loss. No scikit-learn, XGBoost or PyTorch is used. pandas handles the data and matplotlib and Streamlit the output.

## Results

Unseen 2025 races (trained on 2022 to 2024; 24 races, 18,738 laps):

| Model | MAE (s) | R2 | Tyre-age curve error (s) |
|---|---|---|---|
| Median baseline | 0.875 | 0.00 | 0.203 |
| Formula (fuel and form, fitted) | 0.588 | 0.46 | 0.083 |
| Ridge regression | 0.503 | 0.60 | 0.086 |
| **Boosted trees, pre-lap** | **0.465** | **0.66** | 0.075 |
| Persistence (last 3 laps) | 0.368 | 0.72 | 0.058 |
| **Boosted trees, live** | **0.298** | **0.82** | 0.036 |

Mixed-year 5-fold cross-validation over all 90 races (the main performance estimate): pre-lap MAE 0.492 (R2 0.64), live MAE 0.316 (R2 0.81). Full tables, leave-one-year-out, bootstrap intervals and ablations are in `results/summary.md` and `report/report.pdf`.

- **Pre-lap** uses everything known at the start of the lap except the driver's own previous lap times.
- **Live** adds the driver's previous clean laps in the same stint, like a pit-wall forecast. It is mostly persistence: it improves on the plain last-3-lap average by 19% in MAE.

## Project structure

```
app.py                 Streamlit app (dark, F1 themed)
evaluate.py            runs every evaluation, writes results/
predict_demo.py        what-if tyre-age curves from the command line
fetch_data.py          downloads race data from FastF1 into data/
f1tyre/
  config.py            paths, constants, feature tiers, hyperparameters
  data.py              loading and cleaning laps
  features.py          feature table, target, what-if rows
  numpy_ml.py          from-scratch ML library
  models.py            model fitting, fitted formula baselines, conformal intervals
  metrics.py           lap-level, tyre-age-curve and stint-slope metrics
tests/                 unit tests (including a no-future-leakage test)
data/                  lap data, one CSV per race (2022-2025)
results/               metrics.json, summary.md, figures
report/                report.pdf and editable report.docx (with their build scripts), app screenshots
reference_sklearn_xgboost/   earlier scikit-learn and XGBoost version, for comparison
```

## Setup and usage

```
pip install -r requirements.txt
python -m unittest discover -s tests -t .      # unit tests
python evaluate.py                             # all scores and figures (about 10 minutes)
streamlit run app.py                           # interactive app
python predict_demo.py "Bahrain Grand Prix" 35 25
python report/make_report.py                   # rebuild report.pdf (needs node and playwright)
node report/make_report_docx.js                # rebuild the editable report.docx (needs node and the docx package)
```

The data is already in `data/`. To download it again, install `fastf1`, copy `.env.example` to `.env`, fill in an F1 account (FastF1 3.9 requires one) and run `python fetch_data.py` (or `python fetch_data.py 2025`). `.env` is git-ignored.

## Method

- **Cleaning:** dry, green-flag laps only: no lap 1, pit laps, safety car or yellow flags, rain, deleted laps, or laps slower than 104% of the driver's best. 70,830 laps from 90 races remain.
- **Target:** lap time minus the race's field-median lap time (removes circuit and yearly pace differences).
- **Features:** tyre compound and age, lap number and race fraction (fuel proxy), weather, driver and team form from earlier races, traffic (gaps and position at the end of the previous lap), and field context (field pace trend, laps since the last neutralised lap). The fuel effect (about 2.95 s over a race) is fitted from the data.
- **Model:** histogram-based gradient-boosted trees; each round fits a tree to the negative gradient of the pinball loss and sets each leaf to the matching quantile of its residuals. At the median this is the robust absolute-error loss; the 5% and 95% quantiles, calibrated with split conformal prediction, give 90% prediction intervals (92 to 93% coverage on 2025).
- **Validation:** train 2022 to 2024 and test on 2025; race-grouped, year-stratified cross-validation; leave-one-year-out; race-level bootstrap; a label-shuffle test (R2 0.01); feature-group ablation.

## Limitations

- A single lap carries about 0.33 s of random noise (traffic, mistakes), so lap-level R2 has a ceiling near 0.9. Most pre-lap error is each driver's race-specific offset, which no pre-lap information reveals.
- The target is relative to the race's field-median lap, which comes from the session itself.
- Compound differences in the average curves are small next to circuit, fuel and traffic effects; the app shows typical behaviour, not a strategy recommendation.
- Some design choices (the lap filter, the feature set) were made while looking at 2025 results, so the 2025 holdout is slightly optimistic; cross-validation is the better estimate.
- The pure-NumPy tree ensemble is slower than XGBoost (about 6 s for the pre-lap fit, 30 s for the live fit); on the same split it matched XGBoost within 0.002 s MAE (see `reference_sklearn_xgboost/`).

## Data

Lap timing, tyre and weather data come from the FastF1 library (Formula 1 timing data). The CSVs in `data/` are included so the project runs without an F1 account; keep the repository private.
