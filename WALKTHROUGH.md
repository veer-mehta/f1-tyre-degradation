# Project walkthrough: F1 tyre degradation, from scratch

This is a learning guide, not documentation. It follows the order in which you should read the project, explains every concept the code uses, and ends with the questions a professor is likely to ask, with answers.

Contents:

0. The project in two minutes
1. Background: F1 and the vocabulary
2. README and the results: how to read the numbers
3. `f1tyre/numpy_ml.py`: the machine learning written from scratch
4. `f1tyre/data.py`, `f1tyre/features.py`, `tests/`: from raw laps to features, and how leakage is prevented
5. `f1tyre/models.py`: fuel effect, baselines, prediction intervals
6. `f1tyre/degradation.py`: the within-stint degradation model
7. `f1tyre/circuits.py` and `fetch_circuits.py`: circuit characteristics from telemetry
8. `evaluate.py` and `evaluate_degradation.py`: every test we ran and why
9. `app.py` and the report
10. Viva questions and answers
11. Glossary

---

## 0. The project in two minutes

**The question.** A Formula 1 tyre gets slower as it wears. How many seconds does a driver lose per lap of tyre age, and does it differ by compound (soft, medium, hard) and by circuit? A second question follows: how fast will the next lap be?

**The data.** Lap-by-lap timing for 92 races from 2022 to 2025 (FastF1): lap time, tyre compound, tyre age, stint number, weather, track status flags, gaps to the cars around. After cleaning, 70,830 laps from 90 races remain.

**Two models answer two different questions.**

1. **A lap-time model** (gradient boosting, written by us in NumPy). It predicts `delta`, the lap time minus the median lap time of the whole field in that race. It exists in three tiers by how much the driver's own recent history it may use: pre-lap, pre-stint, live.
2. **A degradation model** (least squares with stint fixed effects, also written by us). It isolates the tyre effect by comparing each stint only with itself. It produces the curves in the app.

**Why two models?** The lap-time model is good at predicting laps but bad at answering "what does age do to a soft tyre?" because many other things move at the same time (fuel, traffic, driver, car). The degradation model is built to answer exactly that one question cleanly, at the cost of not predicting absolute lap time.

**Headline results (2025, unseen races).** Pre-lap MAE 0.468 s (R2 0.66), pre-stint 0.437 s (0.70), live 0.294 s (0.82), against 0.875 s for "always predict the median". Soft tyres lose 0.067 s per lap, medium 0.050, hard 0.038, and the model recovers 0.064, 0.049, 0.038 on stints it never saw.

**What makes the project credible.** Everything is implemented from scratch. Every number comes from races the model never trained on. We tested for leakage (a unit test and a label-shuffle test), tested stability (cross-validation, leave-one-year-out, bootstrap) and reported negative results honestly (circuit characteristics mostly do not help).

---

## 1. Background: F1 and the vocabulary

You need these words before reading any code.

- **Compound.** Softer rubber grips more and is faster when fresh but wears out sooner. Three dry compounds: soft, medium, hard.
- **Tyre life / tyre age.** Laps completed on the current set (`TyreLife`).
- **Stint.** The run between two pit stops. A race with two stops has three stints. Each stint is on one set of tyres.
- **Degradation.** The lap-time loss caused by tyre wear, in seconds per lap of age. Typical values here: 0.04 to 0.07 s per lap.
- **Fuel effect.** Cars start heavy and burn fuel, so they get faster through a race. We estimate about 2.95 s over a full race. This effect points the opposite way to tyre wear, which is why raw lap times can fall while the tyre degrades (the "Lap time with fuel burn" view in the app shows exactly this).
- **Field median lap.** The median lap time of all clean laps in that race. We subtract it so that circuit length and year-to-year car pace disappear. `delta = LapTime - field_median`.
- **Safety car, virtual safety car, yellow flags.** Slow, neutralised laps. They do not reflect tyre or driver performance, so we remove them.
- **Dirty air.** Driving close behind another car disturbs the airflow and costs time. We use the gap to the car ahead as a feature.
- **Telemetry.** High-frequency car data (speed, throttle, brake, position X/Y/Z) used to describe circuits.

---

## 2. README and the results: how to read the numbers

Read `README.md` first, then `results/summary.md`. Here is what every number means.

### The metrics

- **MAE (mean absolute error).** Average of `|actual - predicted|` in seconds. "On a typical lap we are 0.47 s off." It is robust to a few wild laps and is the headline metric.
- **Median AE.** The median of the same quantity. Half of all laps are predicted within this error. Always smaller than MAE because errors have a long tail.
- **RMSE.** Square root of the mean squared error. Punishes big misses more. We use it as a diagnostic only.
- **R2 (coefficient of determination).** `1 - SS_res / SS_tot`. It compares your model against the baseline "always predict the mean". 0 means no better than the mean, 1 means perfect, negative means worse than the mean. It depends on how spread out the test data is, so only compare R2 across models on the same test set.
- **Tyre-age curve error.** Average the actual lap-time loss at each tyre age, do the same for the predictions, and take the mean absolute difference between the two curves. It asks "does the model reproduce the shape of degradation?", which is the thing we actually care about. A model can have a decent MAE and still get this wrong.
- **Slope correlation.** For each long stint (15 laps or more), fit a straight line to actual and to predicted fuel-corrected lap times versus age. The slope is the per-lap degradation of that stint. We then correlate actual slopes with predicted slopes across stints. High means the model ranks stints by wear correctly.
- **Coverage (of intervals).** The share of laps whose true value falls inside the predicted 90% interval. Target 90%, we get 93%.

### Why these particular baselines?

- **Median baseline.** Predict the training median for every lap. By definition R2 is about 0. It anchors the scale.
- **Persistence (last 3 laps).** Predict that the next lap equals the average of the driver's last three laps. This is the honest comparison for the "live" model, because the live model gets the same information. Beating persistence by 20% shows the model adds something beyond "do what you just did".
- **Formula A and B.** Linear formulas of the type used by online tyre calculators: degradation proportional to age with compound, track temperature, air temperature and circuit abrasiveness factors. B adds fuel and form. These are fitted to our data with least squares, so they are the strongest version of the formula approach, not strawmen.
- **Ridge regression.** A linear model on the same features as the pre-lap tier. It shows how much the non-linear boosting adds over a linear model: 0.503 s to 0.468 s.

### The three tiers

| Tier | Information allowed | Real-world use |
|---|---|---|
| Pre-lap | Everything known at the start of the lap, except this driver's own previous lap times | Planning before the race: forecasting a lap from conditions, tyre, traffic and form |
| Pre-stint | Pre-lap plus how the driver went in earlier stints of the same race | Strategy at a pit stop |
| Live | Pre-stint plus the last laps of the current stint | Pit wall forecasting during a stint |

The live tier is largely persistence with corrections (R2 0.82 against persistence at 0.72). The pre-lap tier is the cleanest test of the physics and context features. Be ready to say which is which.

### Why is R2 not higher for pre-lap?

A single lap has about 0.33 s of unavoidable randomness (traffic, small mistakes, lift-and-coast). That puts a ceiling near R2 0.9 on any lap-level model. Also, much of the pre-lap error is each driver's race-specific pace offset (one driver is simply having a good day), which nothing known before the lap can reveal.

---

## 3. `f1tyre/numpy_ml.py`: the machine learning from scratch

215 lines. This is the heart of the "implemented from scratch" requirement. Read it top to bottom.

### 3.1 Metrics and least squares

`mae`, `median_ae`, `rmse`, `r2` are the formulas above.

**`ols_fit`** is ordinary least squares: find coefficients `b` minimising `||y - Xb||^2`. We append a column of ones for the intercept and solve with `np.linalg.lstsq`, which uses a numerically stable decomposition (SVD) instead of explicitly inverting `X^T X`. We switched to `lstsq` after a singular-matrix error: `solve` fails when columns are collinear, `lstsq` does not.

**`RidgeRegression`** adds an L2 penalty: minimise `||y - Xb||^2 + alpha ||b||^2`. Closed form `b = (Z^T Z + alpha I)^-1 Z^T (y - mean(y))`. Details in the code:

- Features are standardised (zero mean, unit variance) first, otherwise the penalty would punish features measured in large units unfairly.
- Missing values become 0 after standardising, i.e. "the average value".
- The intercept is the mean of y and is not penalised.

Why ridge? It shrinks coefficients toward zero, which stabilises estimates when features are correlated (tyre age, age squared, lap number are all strongly correlated here). The unit test `test_ridge_shrinks_coefficients` checks the shrinkage.

### 3.2 Gradient boosting: the idea

Boosting builds one strong model from many weak ones, added one at a time, each correcting the mistakes of the sum so far.

Pseudo-code of what `GradientBoosting.fit` does:

```
prediction = constant (the alpha-quantile of y)
repeat n_trees times:
    take a random subsample of rows and a random subset of columns
    compute the residuals  r = y - prediction
    compute the negative gradient of the loss at the current prediction
    grow a small regression tree that predicts that gradient
    set each leaf's value to the alpha-quantile of the residuals in that leaf
    prediction += learning_rate * leaf_value
```

**Why "gradient" boosting?** Fitting a tree to the residuals is gradient descent in function space. For squared error the negative gradient is exactly the residual. For other losses it is something else, and the tree is fitted to that.

### 3.3 The loss: pinball (quantile) loss

For a quantile level `alpha` and residual `r = y - prediction`:

```
loss = alpha * r          if r > 0     (we under-predicted)
loss = (1 - alpha) * (-r) if r <= 0    (we over-predicted)
```

- With `alpha = 0.5` this is half the absolute error, so minimising it gives the **median**. This is what "LAD" (least absolute deviation) means. We use it for the main model because lap times have outliers (traffic, mistakes), and the median is robust to them where the mean is not. The unit test `test_median_boosting_is_robust_to_outliers` checks this.
- With `alpha = 0.05` or `0.95` the loss penalises under- and over-prediction asymmetrically, and the minimiser is the 5th or 95th percentile. We use this for prediction intervals.

**The negative gradient** of the pinball loss with respect to the prediction is `alpha` when `r > 0` and `-(1 - alpha)` when `r <= 0`. That is exactly the line in the code:

```python
grad = np.where(residual > 0, self.alpha, -(1.0 - self.alpha))
```

So at the median every row contributes just +0.5 or -0.5, depending only on the sign of its residual. Outliers cannot dominate, which is the robustness.

### 3.4 Growing a tree on that gradient

`_grow` builds a regression tree iteratively with a stack (no recursion). At each node it asks `_best_split`.

**Histogram binning (`make_bin_edges`, `bin_matrix`).** Instead of testing every distinct value as a split point, each feature is cut into 64 quantile bins up front (the same trick as XGBoost's `hist` mode and LightGBM). Every feature becomes small integers 1..64. Missing values (NaN) get bin 0, so the tree handles missing data naturally and sends all missing values to the left side. This makes split search fast: counting rows per bin with `np.bincount` is a single vectorised pass.

**The split score (gain).** For a candidate split, with `G_L`, `G_R` the sums of the gradient on each side, `n_L`, `n_R` the row counts, and `lambda` a regulariser:

```
gain = G_L^2 / (n_L + lambda) + G_R^2 / (n_R + lambda) - (G_L + G_R)^2 / (n + lambda)
```

This is the reduction in squared error when fitting the gradient with a constant on each side, with an L2 regulariser `lambda = 5` that shrinks small leaves and discourages splits that isolate a few rows. It is the same structure as XGBoost's split gain with the Hessian replaced by the row count (the pinball loss has no useful second derivative, so we treat each row as weight 1).

`np.cumsum` over the bins gives the left-side sums for every threshold at once, so all thresholds of all features are scored in one go. Splits that would leave fewer than `min_samples_leaf` rows on either side are forbidden. The best split is the largest gain, and if the gain is not positive the node becomes a leaf.

**Leaf values: line search.** After the tree structure is fixed, each leaf is set to `learning_rate * quantile(residual in that leaf, alpha)`. For the pinball loss the best constant correction for a group of rows is exactly the alpha-quantile of their residuals. This is Friedman's LAD-TreeBoost idea generalised from the median to any quantile. Using the quantile rather than the average gradient is what makes the median regression truly median-optimal.

**Tree prediction.** `Tree.predict_binned` walks every row down the tree `max_depth` times at once with `np.where`. Leaves point to themselves, so rows that reach a leaf early just stay there.

### 3.5 The regularisation knobs

| Parameter | Value (pre-lap / live) | Role |
|---|---|---|
| `n_trees` | 150 / 400 | number of boosting rounds |
| `max_depth` | 6 | tree depth, limits interaction order |
| `learning_rate` | 0.05 | shrinks each tree's contribution; smaller needs more trees but generalises better |
| `subsample` | 0.5 / 0.8 | fraction of rows per tree (stochastic gradient boosting, reduces variance) |
| `colsample` | 0.6 / 0.8 | fraction of features per tree (decorrelates trees) |
| `min_samples_leaf` | 100 / 20 | minimum rows per leaf; large values stop the model memorising individual laps |
| `reg_lambda` | 5 | L2 regularisation on the split gain |

The pre-lap tier is regularised much harder (50% rows, big leaves) because it has little signal per lap and overfits otherwise. We found this by checking the gap between training and test error: pre-lap train MAE 0.374 against test 0.468 is a moderate gap, live 0.261 against 0.294 is small.

### 3.6 Feature importance

`gain_` accumulates the split gain per feature across all trees, and `feature_importances_` normalises it. It tells you which features the model used most, not which causal factors matter, and it is biased toward features with many distinct values. Be careful describing it.

### 3.7 How good is it compared to XGBoost?

We ran the same split with XGBoost (kept in `reference_sklearn_xgboost/`) and the from-scratch version matched within 0.002 s MAE. The from-scratch version is slower (about 6 s for pre-lap, 30 s for live) because it is NumPy not C++. What it lacks compared with XGBoost: second-order (Hessian) optimisation, sparsity-aware split finding with learned default directions for missing values (ours always sends NaN left), parallelism and exact greedy mode.

**Which algorithm exactly did we implement?** Friedman's gradient boosting machine with regression trees as base learners, using the pinball loss, a LAD-TreeBoost-style quantile line search per leaf, histogram-based split finding, row and column subsampling and L2-regularised gain. It is "XGBoost-like", not XGBoost.

The tests in `tests/test_numpy_ml.py`: known metric values, OLS and ridge coefficient recovery, learning a non-linear function with missing values, robustness to outliers, quantile coverage, determinism with a seed, input validation.

---

## 4. `data.py`, `features.py`, `tests/`: from raw laps to features

### 4.1 Cleaning (`clean_laps`)

We keep only laps that measure tyre + car + driver, not incidents:

- lap time present, compound in soft/medium/hard, tyre age known
- not lap 1 (standing start), not an in-lap or out-lap (pit stops)
- `TrackStatus == "1"`, i.e. green flag, no yellow, safety car or VSC
- `IsAccurate` true (timing is reliable), not deleted by stewards
- dry (no rain)
- lap time at most 104% of that driver's best lap in that race (removes laps with mistakes or lifts)

The 104% cut is a judgement call made while looking at data, mentioned in the limitations. The result is 70,830 laps from 90 races: the 2022 Japanese and 2024 Sao Paulo Grands Prix were mostly wet, so they have no clean dry laps left.

### 4.2 The target

`delta = LapTime - median(LapTime of all clean laps in that race)`. Subtracting the race median removes circuit length and the pace level of that season's cars, so the model learns "how much faster or slower than the field", comparable across tracks. Caveat: the median comes from the race itself, so strictly the target is not known before the race.

### 4.3 Features, by group (`config.py`)

**Tyre.** `tyre_life`, its square and log (non-linear wear), `age_le2` and `fresh_tyre` (new-tyre effect), one-hot compound, age-by-compound interactions, stint number.

**Fuel progress.** `lap_number`, `race_fraction` (lap / total laps). A proxy for fuel load: the car is lighter later in the race.

**Weather.** Track temperature, air temperature, humidity, wind speed, `temp_gap = track - air`.

**Form.** How fast the driver and team have been in earlier races (see below).

**Traffic.** Gap ahead and behind at the end of the previous lap (and ahead two laps back), position, and a dirty-air flag (gap ahead under 1.5 s). Previous-lap values are used because the current lap's gaps are not known when the lap starts.

**Field context.** `field_vs_cum`: the field's average delta on the previous lap minus the cumulative average up to two laps before. It captures whether the whole field is currently getting faster or slower (track rubbering in, temperature change). `laps_since_neutral`: laps since the last safety car or yellow flag (capped at 40), because tyres and track conditions behave differently just after a restart.

**Circuit.** `climb_m_per_km`, the elevation change per km (the one descriptor selected, section 7).

**Pre-stint tier (earlier stints of the same race).** `race_mean_delta_before` (the driver's average delta over earlier stints), `prev_stint_mean_delta`, `prev_stint_last3_delta` (pace at the end of the last stint) and `prev_stint_laps`. This tells the model whether this driver is having a quick or slow race so far.

**Live tier extras.** `prev_delta`, `prev3_delta` (mean of the last 3 laps in this stint) and `prev_lap_gap` (how many laps ago the previous clean lap was).

### 4.4 Form features and how they avoid leakage

`add_form` computes each driver's (or team's) median lap-time ratio (`LapTime / race median`) per race, orders races chronologically (`order = Year*100 + Round`), then uses `shift(1)` followed by a rolling mean over the previous 5, 3 and 10 races. The `shift(1)` is the crucial part: the current race is excluded, so form uses strictly earlier races. At the very start of the data, where no history exists, the driver form falls back to the team's, and the team's falls back to 1.0, the pace of an exactly median car (`NEUTRAL_PACE_RATIO`). An earlier version used a global median here and leaked future information; we fixed it and made the test stricter.

### 4.5 Data leakage: the most important thing to understand

**Leakage** means the model sees information at training or prediction time that would not be available when the prediction would really be made. It produces great scores that fail in reality. This project defends against it in several ways:

1. **Time-aware features.** Form and stint history only use earlier races or earlier stints (`shift(1)`, cumulative sums minus the current row).
2. **Previous-lap-only traffic and field features.** Nothing from the lap being predicted.
3. **Grouping by race in every split.** Laps from the same race never appear in both train and test, because laps within a race are strongly correlated (same weather, same safety cars).
4. **The leakage unit test** (`test_no_future_information_leaks_into_earlier_races`): it alters a later race and checks that no feature of an earlier race changes. We checked the test itself by deliberately breaking the code, and it failed as it should (a "mutation check").
5. **A second unit test** (`test_stint_history_uses_only_earlier_stints`) for the pre-stint features.
6. **The label-shuffle test** (section 8): train on randomly shuffled targets. If R2 on real test data is about 0 the pipeline contains no hidden leak. We get R2 0.00.
7. **M-free field features.** An earlier field-pace feature carried the race median `M` (itself computed from the whole race, including the lap being predicted), so we kept only versions that do not contain it.

### 4.6 What the other tests cover

`test_clean_laps_exclude_pit_laps_and_lap_one`, `test_target_is_relative_to_field_median`, `test_tier_features_are_present_and_finite`, `test_every_event_has_circuit_features`.

---

## 5. `f1tyre/models.py`: fuel effect, baselines, intervals

### 5.1 Estimating the fuel effect (`estimate_fuel_effect`)

We want the seconds gained over a race as fuel burns. Within each (race, driver) we remove the mean (a fixed effect), then regress lap time on `race_fraction` and on the tyre age terms per compound (age and age squared). Because tyre age is in the model, its effect is separated from the fuel trend. The negative of the `race_fraction` coefficient is the total fuel gain: about 2.95 s per race.

Honest caveat: this coefficient also absorbs anything else that changes steadily through a race (track rubbering in makes the track faster). It is "fuel plus track evolution", and the physical fuel effect alone is usually quoted at roughly 0.03 s per kg per lap with about 100 kg burned in a race, so a total of around 3 s is the right order of magnitude. If asked, say that.

### 5.2 Formula baselines (`fit_formulas`)

Formula A: `delta ~ age*compound + age*track_temp + age*air_temp + age*abrasiveness`. Abrasiveness is a per-circuit constant: the mean degradation slope at that circuit in the training data, computed leaving out the race itself for training rows so that it does not know its own answer. Formula B adds `race_fraction`, driver form and team form. Both are fitted with `ols_fit`. They mimic an online tyre-degradation calculator with constants fitted to our data.

### 5.3 Prediction intervals: conformal prediction (`conformal_interval`)

**Problem.** A point prediction does not say how sure we are. Raw quantile models (5% and 95%) give an interval but are not guaranteed to cover 90% of reality (ours covered 84% to 85%).

**Method used (conformalised quantile regression).**

1. Split the training races into a fitting set (80%) and a calibration set (20%), by race.
2. Fit two boosting models on the fitting set, one for the 5% quantile (`lo`) and one for the 95% quantile (`hi`).
3. On the calibration set compute a conformity score `max(lo - y, y - hi)`. It is negative when the true value is inside the interval and positive by how much it is outside otherwise.
4. Take the `ceil((n+1)(1-alpha))/n` quantile `q` of those scores (the finite-sample correction).
5. The final interval is `[lo - q, hi + q]`.

**Why it works.** If calibration and test data are exchangeable, the interval covers the truth with probability at least 90%, no matter how bad the quantile models are. Here coverage is 93%: slightly conservative, because test years differ a bit from the calibration races (calibration is drawn from the training years, so exchangeability is only approximate).

**Width.** Mean width is 2.2 s pre-lap, 2.1 s pre-stint and 1.5 s live: more information gives tighter intervals.

---

## 6. `f1tyre/degradation.py`: the within-stint degradation model

This is the most conceptually important file. The whole app's curves come from it.

### 6.1 Why a separate model is necessary

Ask the lap-time model "what happens to a soft tyre at age 30?" and you hit two problems.

1. **Confounding.** In the data, drivers on old soft tyres are in specific situations (late stints, particular strategies, particular cars). The model attributes those effects to age.
2. **Extrapolation.** Soft tyres are rarely run for 30 laps. The model has no data there and its prediction is meaningless. This flattened the compound differences in an early version of our app and was the reason the curves looked the same for all compounds.

### 6.2 The idea: stint fixed effects

A stint has one driver, one car, one tyre set, one strategy. Anything constant over the stint (driver pace, car pace, tyre compound's base pace, fuel at the start, track temperature on average) can be removed by subtracting the stint mean from every variable. This is the **within transformation**, equivalent to giving every stint its own intercept (a fixed effect). What remains is how lap time moves around the stint mean as tyre age moves around its mean. That is pure degradation.

Implementation: `within_stint(values, panel)` subtracts the stint mean, and `DegradationModel.fit` regresses the within-stint `y` on within-stint regressors. A stint needs at least 8 clean laps (`MIN_STINT_LAPS`).

### 6.3 The response `y`

`y = delta + fuel_total * LapNumber / TotalLaps` (`build_panel`). Fuel burn makes laps faster over the race, so adding that gain back gives a fuel-corrected lap time in which any remaining upward trend is tyre wear.

### 6.4 The regressors (`_terms`)

For each compound `k` in soft, medium, hard:

- `age[k] = age * 1[compound = k]`
- `age2[k] = age^2 / 100 * 1[compound = k]` (the parabola; divided by 100 to keep the scale comparable)

Then shared effects:

- `age x track_temp` (standardised stint-mean temperature): hotter track, faster wear.
- `age x climb_m_per_km` (standardised): the circuit effect.

Everything interacts with age because stint fixed effects already absorb levels. Only the way lap time changes with age can be estimated. This has an important consequence worth remembering: **the model cannot say that the soft tyre is faster than the hard when fresh**, because compound pace offsets cancel in the stint mean. It only says how fast each compound loses time. That is why the app does not recommend strategies.

The compound terms are not penalised. The temperature and circuit terms get a ridge penalty (`alpha = 100`) because they are weaker and more correlated.

### 6.5 Fitting and the curve

`fit` solves the penalised normal equations with `np.linalg.lstsq`. `curve(compound, circuit_row, track_temp, ages)` evaluates the fitted terms at ages 1..30 and subtracts the value at age 1, so every curve starts at zero: "loss versus a fresh tyre".

### 6.6 Bands: the race bootstrap

To show uncertainty we resample whole races with replacement 200 times, refit the model on each resample and store the coefficients (`degradation_bootstrap.csv`). `curve_band` evaluates the curve with every draw and takes the 2.5% and 97.5% quantiles at each age: the shaded 95% band. Resampling races (not laps) matters because laps within a race are not independent.

### 6.7 Why a parabola and not something fancier

We tested piecewise-linear ("hinge") shapes that can create a flat start or a late cliff. We chose between shapes by **curve fidelity** on held-out data: the mean absolute gap between the observed average loss-versus-age curve and the model's. On unseen 2025 stints the parabola scored 0.136 s against 0.145 to 0.147 s for the flexible ones, so the data do not support a cliff. Do not claim "tyres have no cliff". Claim "within this lap range and data, a smooth curve fits as well or better than hinges".

### 6.7b Does the line fit every stint? (a question you will get)

Looking at single stints in the Race explorer, the white model line often looks flatter than the dots. We measured it over all 3,353 stints:

- On average the model's stint slope is about 13% below the average actual stint slope (soft 0.068 against 0.083 s per lap, medium 0.050 against 0.057, hard 0.041 against 0.046), and 53% of stints are steeper than the line. The bias is worst for short stints (18 laps or fewer: 0.069 actual against 0.055 modelled) and absent for long ones (over 25 laps: 0.035 against 0.037).
- Cause 1, weighting. Least squares weighs every lap equally, so a 45-lap stint counts far more than a 10-lap one. Long stints are the gentle ones, because tyres that wear fast are changed early (a selection effect). The fit therefore describes the typical lap, not the typical stint.
- Cause 2, dispersion. Stint slopes vary a lot and the model explains little of that variation (correlation between modelled and actual slope 0.40). Any single stint is steeper or flatter than the typical line.
- What we tried. Weighting every stint equally removes the bias (calibration slope 1.01) but fits unseen 2025 stints worse (curve error 0.29 against 0.18 s) and makes the circuit characteristic stop helping, so we kept the lap-weighted fit. Adding the stint's length as a covariate helps slightly (calibration 1.53 to 1.31) but is not usable for planning, because a stint's length is a consequence of its wear, not a choice that changes it. Season terms did not improve held-out error.
- What to say: the line is the conditional average, the dots show individual variation, and the Model tab shows the full picture, so the limitation is stated, not hidden.

### 6.7c Tyre cliff detection (`f1tyre/cliff.py`)

**What a cliff is.** A tyre does not always wear steadily. Past some age the rubber can suddenly lose grip, so each extra lap costs far more than the one before.

**Why the average curve cannot show it.** (1) Teams usually pit before a cliff arrives. (2) The age at which it starts differs between stints. (3) An average over all stints at each age mixes stints that have and have not reached it. So we look at stints one by one.

**Method.** For every stint with at least 20 clean laps we fit two models to the fuel-corrected lap times: a straight line, and a straight line plus a hinge `max(age - knee, 0)`, which lets the slope increase after the knee age. We try every possible knee (at least 8 laps before it and 4 after) and keep the best. We call it a cliff when (a) the hinge model fits clearly better by BIC (a gain over 10, so the extra parameter is worth it) (b) the extra slope is at least 0.1 s per lap and (c) the knee lies in the second half of the stint (an early knee is a warm-up shape, where the tyre switches on and then wears, not a cliff). BIC (Bayesian information criterion) is a fit score that charges a penalty for each extra parameter, so a gain of 10 means the hinge is clearly not just fitting noise.

**The noise reference.** Random noise (traffic, a mistake) can make a stint look like it speeds up late. The same noise can make a stint look like it slows down late by the same amount, and a tyre cliff cannot produce that. So the share of stints that slow down late, which we call relief, estimates how many false cliffs to expect. Result: 5.9% of long stints speed up late and 1.5% slow down late, so about 4% have a real cliff, typically 0.25 s per lap steeper afterwards, with a median onset age of 24 laps (soft 18.5, medium 21, hard 27, ordered as you would expect). The soft sample is small (10 cliffs). Our first version accepted a knee anywhere and found 11%, but many of those early knees were warm-up shapes, so we restricted cliffs to the second half of the stint.

**Caveats to state.** The reference controls for symmetric noise only. A slow traffic phase at the end of a stint could still look like a cliff, and tyre-saving or a car problem can mimic one. We cannot see tyre temperature, so we call it a late speed-up in lap-time loss, which is what a cliff does to lap times.

**Where it appears.** `evaluate_degradation.py` writes `results/cliff_stints.csv`, `cliff.png` and the summary in `degradation.json`. The Race explorer draws a red dotted line at the detected onset, and the Model tab shows the table and histogram. Tests are in `tests/test_cliff.py` (a synthetic knee is found, straight stints are not flagged, a late recovery is not called a cliff).

### 6.7d Why elevation change, and why only one characteristic?

**What we found.** Across the 25 circuits, elevation change per km has a rank correlation of 0.59 with measured degradation severity. Because we picked the best of 13 candidates, 0.59 must be compared with the best correlation chance alone would produce among 13 unrelated columns. We did that with a permutation test (shuffle which circuit has which severity, keep the best of 13, repeat 5,000 times): p = 0.02. Dropping any single circuit gives 0.55 to 0.66, so one odd circuit does not drive it. It is statistically real but modest.

**Is there a physical reason?** None is proven. Hypotheses: hilly circuits (Spa, Red Bull Ring, Interlagos, Suzuka, Barcelona, Hungary) are older, flowing layouts where crests and compressions keep changing the vertical load on the tyres, and they may share surface and layout traits that raise wear. Elevation may also be standing in for those traits. Say it is an association, not an established cause.

**Why only one?** With 25 circuits you cannot estimate the separate effect of 13 correlated characteristics: the more we add, the more we fit noise. Cross-validation inside the training seasons chose one (held-out stint error 0.3744 s with it against 0.3766 s without; adding more got worse), and using all 13 reduced the held-out fit. The honest conclusion is that circuit shape is a weak predictor here, because tyre allocation, weather, temperature and race events matter more.

### 6.8 Evaluating degradation (`evaluate`, `curve_table`, `curve_fidelity`)

- **Within-stint R2** is computed on the centred `y`. It is about 0.31, which is much lower than lap-level R2 because the response has had the easy parts removed (driver, car, fuel): what is left is noisy.
- **Slope correlation / slope error** compare per-stint wear slopes (long stints only).
- **`curve_table`** averages actual and predicted loss versus a reference age (3, because age-1 laps are rarely clean) per compound and age, with at least `min_n` stints at each age. This produces the "reality check" chart.

### 6.9 Circuit severity and descriptor selection

`circuit_severity`: for each circuit, the observed pooled within-stint slope divided by the slope we would expect given the mix of compounds used there. A value of 1 is average, above 1 harsher.

`rank_descriptors` orders the 13 circuit measures by absolute Spearman rank correlation with severity (Spearman: correlation of ranks, insensitive to outliers and non-linearity).

`select_descriptors` chooses how many descriptors to keep (0, 1, 2, 3 or 5) by 5-fold race-grouped cross-validation **inside the training years only**, re-ranking inside each fold so that selection never sees the validation races (this is a **nested** procedure: the choice is part of what is being cross-validated). The result: 1 descriptor (CV within-stint MAE 0.3744 against 0.3766 with none), and it is elevation change per km.

`select_shape` does the same for the curve shape, scored by curve fidelity.

---

## 7. `f1tyre/circuits.py` and `fetch_circuits.py`: circuit characteristics

For each of the 25 circuits `fetch_circuits.py` loads the fastest clean race lap's telemetry (speed, throttle, brake, X/Y/Z position, time vs distance) and `profile_from_telemetry` converts it into 13 numbers.

Processing steps:

1. Sort by distance and resample onto a 5 m grid with linear interpolation.
2. Smooth position (`moving_average`) to remove sensor noise.
3. **Heading** = `arctan2(dy, dx)` of the path, unwrapped so it does not jump at +-pi. **Curvature** = rate of change of heading per metre.
4. **Lateral acceleration** = `v^2 * |curvature|` (physics: centripetal acceleration), converted to g by dividing by 9.81. `mean_lat_g` is its time-weighted mean (slow sections weigh more because the car spends more time there), `high_lat_pct` the share of lap above 2.5 g.
5. **Corners** are local minima of smoothed speed with a minimum prominence of 25 km/h, classified slow (under 120 km/h), medium (120 to 200) and fast (over 200).
6. **Braking** share is time with the brake on, and heavy braking zones are separated brake applications that lose at least 100 km/h.
7. **Throttle**: share of time at 98% or more.
8. **Elevation change per km** is the sum of absolute changes in the smoothed height `Z` divided by lap length (total climbing plus descending per km, not net).

The tests build synthetic circles with analytic answers (a circle of known radius and speed has a known lateral g) to confirm the maths.

**Result.** Cross-validation kept only elevation change (Spearman 0.59 with measured severity). All 13 together do not help, and predicting a circuit's severity from all 13 fails on circuits left out (R2 -0.17). The honest explanation: 25 circuits is a tiny sample for 13 variables, and severity also depends on tyre allocation, weather and incidents. Severity is not even stable between years (year-to-year correlations 0.73, 0.41, 0.15). This is a genuine negative result and it is fine to say so.

---

## 8. `evaluate.py` and `evaluate_degradation.py`: every test and why

### 8.1 Validation designs, and what each one answers

| Test | What is held out | Question it answers |
|---|---|---|
| Train 2022 to 2024, test 2025 (holdout) | The whole last season | Does the model work on the future? |
| Mixed-year 5-fold CV by race | About 20% of every year's races in each fold | How good is it on average, using all 90 races? (our main estimate) |
| Leave-one-year-out | One entire season at a time | Do regulation changes between years break it? (each year is similar) |
| Race bootstrap | Resamples the 24 test races 1,000 times | How uncertain is the test score? Pre-lap MAE has a 95% interval of 0.43 to 0.52 s, and beats Formula B by 0.12 s (0.08 to 0.16) in every resample |
| Label shuffle | Targets are randomly permuted | Is there hidden leakage? R2 0.00, MAE 0.87, as for the median baseline |
| Feature-group ablation | Drop one feature group, retrain | Which information matters? Dropping tyre, fuel progress or traffic costs about 0.04 to 0.05 s each |
| Train vs test gap | Compare training and test error | Is it overfitting? Moderate gap pre-lap, small for live |

**Why race-grouped splits?** Laps from one race share weather, safety cars and track state. If some laps of a race were in training and others in test, the model could memorise the race. Splitting by race measures generalisation to new races, which is the real use.

**Which is better, CV or the 2025 holdout?** Cross-validation uses all races and is more stable. The 2025 holdout is the most realistic ("predict the next season") but is a single sample, and some design choices (the 104% filter, feature set) were made while looking at 2025, so it is slightly optimistic. We call CV the main estimate and the holdout a sanity check.

### 8.2 Walking through `evaluate.py`

- `assign_folds`: assign every race to one of 5 folds, shuffled within each year so each fold has about 20% of every year.
- `predict_all`: fits every model on the training laps and predicts the test laps. This function is reused by the holdout and every CV fold.
- `holdout`, `mixed_year_cv` (out-of-fold predictions for all laps then scored), `leave_one_year_out`, `later_stints` (scores restricted to laps in stint 2 or later, where pre-stint features exist: MAE 0.500 to 0.457), `sanity_checks`, `race_bootstrap`, `conformal_interval` for each tier, then `write_summary` and `plot_results`.

### 8.3 Walking through `evaluate_degradation.py`

1. Build panels for train, test and all data.
2. Compute circuit severity, its correlation with each descriptor, year-to-year stability, and a leave-one-circuit-out ridge model predicting severity from all 13.
3. Choose the curve shape by CV curve fidelity.
4. Evaluate single descriptors, then choose the number of descriptors by nested CV.
5. Compare variants on 2025 stints: base (compound, curvature, temperature), plus selected descriptor, plus all 13, plus circuit identity dummies (each circuit's own history), and the lap-level gradient boosting with and without the circuit feature.
6. Produce the curve check (observed vs modelled), bootstrap the final model fitted on all races, and save the figures.

Reading the result table: within-stint R2 0.31 and slope correlation 0.35 with temperature only, 0.37 with elevation, 0.39 with all 13 (but R2 drops to 0.30, i.e. it fits the slopes more but the laps less), and 0.34 with circuit identity (R2 0.19, overfitting). The circuit effect is small. In the lap-level model, adding elevation raises slope correlation from 0.55 to 0.57 but leaves MAE unchanged (0.465 s without, 0.468 s with): helps tyre behaviour, not lap time.

---

## 9. `app.py` and the report

### 9.1 The app (Streamlit)

`load_model` (cached, so it runs once) builds the lap table, fits the fuel effect and the final degradation model on all 2022 to 2025 data using the descriptor and shape settings saved in `results/degradation.json`, and loads the bootstrap draws. Tabs:

- **Predictor.** Choose circuit and track temperature. Calls `model.curve` for each compound and `curve_band` for the shaded 95% interval. The "Lap time with fuel burn" view subtracts `fuel_total / race_laps * (age - 1)`, so you see the net effect of wear plus fuel, which can go down. The tiles give the loss after 10, 20 and 30 laps. The circuit profile shows telemetry-derived measures, with the one used by the model marked red.
- **Race explorer.** Pick a season, race and driver. For each stint of 8 or more clean laps, plots actual fuel-corrected laps (minus the stint mean) against the model's within-stint prediction, with a table of actual and model seconds per lap.
- **Circuits.** Ranks the 25 circuits by predicted loss after N laps for a chosen compound and temperature. Differences are modest because only elevation is a circuit input (plus temperature at a fixed value), so treat it as tendency.
- **Model.** Headline tiles and result tables from `metrics.json` and `degradation.json`, the reality-check chart and the circuit correlation chart.

Design choices worth knowing: Altair charts for the dark F1 theme; all numbers come from the saved results files and the fitted model, none are hard-coded except the "1.7x" tile (soft over hard wear).

### 9.2 The report

`report/make_report.py` builds the PDF from HTML (rendered with Playwright), `make_report_docx.js` builds the editable DOCX. Both read `results/*.json` so numbers cannot drift from the evaluation.

---

## 10. Viva questions and answers

### The problem and data

**Q: What problem are you solving and why does it matter?**
A: Tyre degradation (seconds lost per lap of tyre age) drives pit-stop strategy. We estimate it from race data and also forecast lap times at three levels of information.

**Q: Where does the data come from and how big is it?**
A: FastF1, the public Formula 1 timing API. 92 races from 2022 to 2025, 70,830 clean laps from 90 races after cleaning.

**Q: Why only 2022 to 2025?**
A: The 2022 regulations changed the cars and tyres (18-inch wheels, ground effect). Earlier seasons would be a different distribution and add noise more than information, so we decided not to add them.

**Q: What did you remove and why?**
A: Lap 1, pit laps, safety car, VSC and yellow-flag laps, rain, deleted or inaccurate laps and laps over 104% of the driver's best. They reflect incidents, not tyre wear or pace.

**Q: What is the target and why that target?**
A: `delta`, lap time minus the race's median lap. It removes circuit length and year-to-year pace, so one model can work across tracks.

### Machine learning from scratch

**Q: Which algorithms did you implement?**
A: OLS (via `lstsq`), ridge regression (closed form), and histogram-based gradient boosting with regression trees and the pinball loss, plus all metrics. Conformal intervals and the degradation model are built on those.

**Q: Explain gradient boosting in simple words.**
A: Start with a constant prediction. Repeatedly measure how wrong the current prediction is, train a small tree to predict the correction direction, and add a damped version of it. Many small corrections add up to a strong model.

**Q: Why gradient and not just residual?**
A: For squared error the negative gradient equals the residual. For the pinball loss it is just plus or minus a constant by sign, so we fit trees to that and then set leaf values to the residual quantile.

**Q: Why the pinball loss instead of squared error?**
A: Median regression is robust to outlier laps (traffic, mistakes), and other quantiles give prediction intervals. Squared error would chase outliers and predict the mean.

**Q: What does the quantile at a leaf do?**
A: For the pinball loss the best constant update for a set of rows is the alpha-quantile of their residuals. Setting leaves to that quantile is an exact line search.

**Q: What is histogram binning and why use it?**
A: Pre-cut each feature into 64 quantile bins. Splits are then tested between bins, so finding the best split costs one counting pass over the data instead of sorting every feature at every node. Same trick as LightGBM and XGBoost `hist`.

**Q: How do you handle missing values?**
A: NaN goes to bin 0 and always follows the left branch. Ridge replaces NaN with the feature mean (0 after standardising).

**Q: What is the split gain formula?**
A: `G_L^2/(n_L+lambda) + G_R^2/(n_R+lambda) - G^2/(n+lambda)`: reduction in squared error with an L2 regulariser, computed from gradient sums.

**Q: How do you stop overfitting?**
A: Shrinkage (learning rate 0.05), row and column subsampling, minimum leaf size, L2 regularisation on gain, limited depth, and checking the train-test gap. The pre-lap model uses much stronger settings.

**Q: Is it as good as XGBoost?**
A: On the same split it matched within 0.002 s MAE. It is slower, uses only first-order information, and has no learned missing-value direction.

**Q: Why not just use scikit-learn?**
A: The assignment required implementing from scratch with NumPy and pandas. We kept an sklearn/XGBoost version in `reference_sklearn_xgboost/` for comparison.

**Q: What is the difference between bagging and boosting?**
A: Bagging trains models independently on resamples and averages them (reduces variance, random forest). Boosting trains sequentially, each on the errors of the sum so far (reduces bias). We use row and column subsampling inside boosting for extra variance reduction.

**Q: What is ridge regression and why use it?**
A: Least squares with an L2 penalty on coefficients. It shrinks them and stabilises estimates when features are correlated.

### Features and leakage

**Q: What is data leakage and how did you prevent it?**
A: Using information at training or prediction time that would not be available in reality. We used only earlier races for form (`shift(1)`), previous-lap values for traffic and field context, race-grouped splits, a unit test that changes a later race and checks earlier features stay identical, and a label-shuffle test that gives R2 0.00.

**Q: Why did you use the form features and are they leaky?**
A: They capture that some drivers and cars are faster. They average the driver's or team's pace ratio over earlier races only, so they are not leaky. The first races of 2022 fall back to the team's form, and then to a neutral 1.0.

**Q: What is the difference between the tiers?**
A: Pre-lap uses nothing about the driver's own recent laps. Pre-stint adds earlier stints of the same race. Live adds the last laps of the current stint. More information, lower error, as the results show.

**Q: Your live model is almost persistence. Is that cheating?**
A: It is a legitimate use case (pit wall forecast during a stint) and we compare it against persistence directly: it improves MAE by 20% over it. We also state clearly that the pre-lap tier is the harder test of the features.

### Evaluation

**Q: Why MAE and not RMSE or R2?**
A: MAE is in seconds, robust to outliers and matches the loss we train on. We also report median AE, R2, tyre-curve error and slope correlation because MAE alone does not tell us whether tyre behaviour is right.

**Q: Why is R2 only 0.66 for pre-lap?**
A: Single laps carry about 0.33 s of noise, so R2 has a ceiling near 0.9, and much of the remaining error is each driver's race-specific offset that is unknowable before the lap.

**Q: Why cross-validate by race?**
A: Laps within a race are correlated. Splitting by race measures performance on unseen races.

**Q: Why mixed-year CV and leave-one-year-out?**
A: Mixed-year uses all data and gives a stable average. Leave-one-year-out checks that changes between seasons do not break the model: errors are similar in every year.

**Q: What is the bootstrap for?**
A: To measure the uncertainty of the test score by resampling races. It shows the gain over the formula baseline holds in every resample.

**Q: What does the label-shuffle test show?**
A: A model trained on random targets scores R2 about 0 on real data, so our pipeline has no leak giving free accuracy.

**Q: What is the 90% interval and what is conformal prediction?**
A: A range expected to contain the true lap delta 90% of the time. Conformal prediction calibrates the raw quantile intervals using held-out races so that coverage is guaranteed in a finite-sample sense. We observe 93%.

**Q: Why is coverage 93% and not 90%?**
A: The guarantee is "at least 90%" under exchangeability, and our calibration races (training years) are not exactly exchangeable with the 2025 races, which makes it slightly conservative. Slight over-coverage is the safe direction.

### Degradation model

**Q: Why a separate degradation model?**
A: The lap-level model answers a different question and confounds tyre age with driver, car, fuel and strategy, and extrapolates badly (soft at age 30). The stint fixed effects model compares each stint only with itself.

**Q: What are fixed effects?**
A: A separate intercept per group (here per stint). Subtracting the group mean from all variables removes everything constant within the group. Only the within-group variation drives the estimates.

**Q: What can the degradation model not tell you?**
A: Absolute pace differences between compounds (they cancel in the stint mean) and the best strategy. It describes typical wear only.

**Q: How do you correct for fuel?**
A: We estimate the total fuel gain (about 2.95 s per race) from a within-driver regression and add `fuel * lap/total_laps` back to the lap delta before fitting the tyre model. It also absorbs track evolution.

**Q: Why quadratic in age?**
A: It was the best on held-out curve fidelity (0.136 s against 0.145 to 0.147 s for piecewise-linear shapes) and it is smooth and interpretable.

**Q: How are the bands computed?**
A: Race-level bootstrap: resample races 200 times, refit, take the 2.5% and 97.5% curve quantiles.

**Q: Why does the soft curve look steeper than the hard?**
A: It is: soft loses 0.067 s per lap against 0.038 for hard on unseen 2025 stints and the model reproduces both (0.064 and 0.038).

**Q: Why do lap times fall in the "with fuel burn" view?**
A: The car gets lighter, gaining about 0.05 s per lap, which can outweigh the tyre loss, especially for hards.

### Circuit characteristics

**Q: How did you get circuit characteristics?**
A: From the telemetry of the fastest race lap per circuit: speed, throttle, brake, X/Y/Z. We derived 13 measures including cornering g (from path curvature and speed), corner counts and elevation change per km.

**Q: Which ones helped?**
A: Only elevation change per km, chosen by nested cross-validation inside training years. It improves held-out stint slope correlation from 0.35 to 0.37.

**Q: Why not use all 13?**
A: Only 25 circuits, so 13 descriptors overfit. Using all gives slope correlation 0.39 but lowers within-stint R2 (0.30 against 0.31), and predicting circuit severity from all 13 on unseen circuits gives R2 -0.17.

**Q: Isn't a negative result a failure?**
A: No. It is a tested, reported finding: severity depends on tyre allocation, weather and race events, and it is not even stable between years (consecutive-year correlations 0.73, 0.41, 0.15).

### Cliff and circuit questions

**Q: Did you find the tyre cliff?**
A: Yes, at stint level. About 4% of stints of 20 or more laps have a real late speed-up of roughly 0.25 s per lap, starting at a median age of 24 laps. An average curve cannot show it because teams pit before it and onset ages differ.

**Q: How do you know those are cliffs and not noise?**
A: Stints that slow down late by the same amount appear in 1.5% of stints, against 5.9% that speed up, and symmetric noise would produce both equally. The excess is the cliff.

**Q: How is elevation change related to tyre degradation?**
A: Empirically, circuits with more elevation change had higher measured degradation (rank correlation 0.59, p = 0.02 after allowing for choosing the best of 13). We do not claim a proven mechanism; it may stand in for layout and surface traits of hilly circuits.

**Q: Why use only one circuit feature?**
A: With 25 circuits, cross-validation showed that extra characteristics added noise: using all 13 worsened the held-out fit.

### Honest weaknesses (know these before they are asked)

**Q: What are the limitations?**
A: (1) Lap-level noise caps R2 near 0.9. (2) The target uses the race median, which comes from the race. (3) The fuel effect includes track evolution. (4) Circuit descriptors come from one lap per circuit and ignore layout changes. (5) Only 25 circuits. (6) The 2025 holdout is slightly optimistic since the lap filter and feature set were chosen while looking at it, so cross-validation is the better estimate. (7) Curves show typical behaviour, not strategy advice, because drivers manage tyres.

**Q: What would you do next?**
A: Per-stint tyre allocation and pit-strategy data, tyre temperature or compound-specific wear modelling with more seasons as they are released, a hierarchical (mixed-effects) model with a random circuit effect, and proper time-series cross-validation.

---

## 11. Glossary

- **Bootstrap.** Estimate uncertainty by repeatedly resampling the data with replacement and recomputing the statistic.
- **Conformal prediction.** Method that turns any predictor into intervals with a guaranteed coverage level using a calibration set.
- **Cross-validation.** Repeatedly split the data into training and validation parts, so every observation is validated once.
- **Fixed effect.** A separate constant for each group, removed by subtracting group means.
- **Gradient boosting.** Sequential ensemble in which each new learner fits the negative gradient of the loss at the current prediction.
- **Histogram-based splitting.** Cut features into bins first and search split points among bins.
- **Leakage.** Using information unavailable at prediction time.
- **LAD (least absolute deviation).** Regression minimising absolute error, which estimates the median.
- **Nested cross-validation.** Choosing settings (here, descriptors or shape) inside the training part of each fold so validation data never influence the choice.
- **Persistence.** Predicting that the next value equals a recent one.
- **Pinball (quantile) loss.** Asymmetric absolute loss whose minimiser is a chosen quantile.
- **Regularisation.** Anything that limits model complexity to improve generalisation (shrinkage, penalties, minimum leaf size, subsampling).
- **Ridge regression.** Least squares with an L2 penalty on coefficients.
- **Spearman correlation.** Pearson correlation of the ranks of two variables.
- **Stint.** Laps between pit stops on one set of tyres.
- **Within transformation.** Subtracting group means from all variables, equivalent to group fixed effects.
