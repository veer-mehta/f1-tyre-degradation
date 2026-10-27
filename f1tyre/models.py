"""Model construction, fitted baselines, conformal intervals and what-if curves."""
import numpy as np
import pandas as pd
from f1tyre import numpy_ml as ml
from f1tyre.config import GB_PARAMS


def make_gb(tier, **overrides):
    params = dict(GB_PARAMS["live" if tier == "live" else "pre_lap"])
    params.update(overrides)
    return ml.GradientBoosting(**params)


def fit_gb(train, cols, tier, alpha=0.5):
    return make_gb(tier, alpha=alpha).fit(train[cols].values, train["delta"].values)


def estimate_fuel_effect(train):
    d = train
    group = d["race_id"] + "_" + d["Driver"]
    cols = ["race_fraction", "age_x_soft", "age_x_medium", "age_x_hard"]
    x = d[cols].copy()
    for compound in ["soft", "medium", "hard"]:
        x[f"age2_{compound}"] = d[f"compound_{compound}"] * d["TyreLife"] ** 2
    x = x - x.groupby(group).transform("mean")
    y = d["LapTime"] - d.groupby(group)["LapTime"].transform("mean")
    coef = np.linalg.lstsq(x.values, y.values, rcond=None)[0]
    return -float(coef[0])


def race_degradation_slopes(train, fuel_total_s):
    d = train[["race_id", "Driver", "EventName", "TyreLife", "delta", "LapNumber", "TotalLaps"]].copy()
    d["y"] = d["delta"] + fuel_total_s * d["LapNumber"] / d["TotalLaps"]
    d["y"] = d["y"] - d.groupby(["race_id", "Driver"])["y"].transform("mean")
    d["x"] = d["TyreLife"] - d.groupby(["race_id", "Driver"])["TyreLife"].transform("mean")
    d["xy"] = d["x"] * d["y"]
    d["xx"] = d["x"] ** 2
    g = d.groupby(["EventName", "race_id"])[["xy", "xx"]].sum()
    return (g["xy"] / g["xx"]).rename("slope").reset_index()


def abrasiveness(rows, race_slope, leave_race_out):
    totals = race_slope.groupby("EventName")["slope"].agg(["sum", "count"])
    s = rows["EventName"].map(totals["sum"]).astype(float)
    c = rows["EventName"].map(totals["count"]).astype(float)
    if leave_race_out:
        own = rows["race_id"].map(race_slope.set_index("race_id")["slope"])
        s = s - own.fillna(0.0)
        c = c - own.notna().astype(float)
    return (s / c).where(c > 0).fillna(race_slope["slope"].mean())


def formula_features(frame, abr, with_fuel_and_form):
    x = pd.DataFrame(index=frame.index)
    for compound in ["soft", "medium", "hard"]:
        x[f"age_x_{compound}"] = frame[f"age_x_{compound}"]
    x["age_x_track_temp"] = frame["TyreLife"] * frame["TrackTemp"]
    x["age_x_air_temp"] = frame["TyreLife"] * frame["AirTemp"]
    x["age_x_abrasiveness"] = frame["TyreLife"] * abr
    if with_fuel_and_form:
        x["race_fraction"] = frame["race_fraction"]
        x["driver_form"] = frame["driver_form"]
        x["team_form"] = frame["team_form"]
    return x


def fit_formulas(train, test, fuel_total_s):
    race_slope = race_degradation_slopes(train, fuel_total_s)
    abr_train = abrasiveness(train, race_slope, True)
    abr_test = abrasiveness(test, race_slope, False)
    preds = {}
    for name, full in [("formula_a_calculator_style", False), ("formula_b_with_fuel_and_form", True)]:
        coef = ml.ols_fit(formula_features(train, abr_train, full).values, train["delta"].values)
        preds[name] = ml.ols_predict(coef, formula_features(test, abr_test, full).values)
    return preds


def conformal_interval(train, test, cols, tier, alpha=0.1, seed=42):
    rng = np.random.RandomState(seed)
    races = train["race_id"].unique()
    calib_races = rng.choice(races, size=int(0.2 * len(races)), replace=False)
    is_calib = train["race_id"].isin(calib_races)
    fit, calib = train[~is_calib], train[is_calib]
    lower = fit_gb(fit, cols, tier, alpha=alpha / 2)
    upper = fit_gb(fit, cols, tier, alpha=1 - alpha / 2)
    lo_c, hi_c = lower.predict(calib[cols]), upper.predict(calib[cols])
    scores = np.maximum(lo_c - calib["delta"].values, calib["delta"].values - hi_c)
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    q = float(np.quantile(scores, level))
    lo, hi = lower.predict(test[cols]), upper.predict(test[cols])
    y = test["delta"].values
    raw_cov = float(((y >= lo) & (y <= hi)).mean())
    cal_cov = float(((y >= lo - q) & (y <= hi + q)).mean())
    return {"target_coverage": 1 - alpha, "raw_coverage": round(raw_cov, 4), "calibrated_coverage": round(cal_cov, 4), "calibrated_mean_width_s": round(float((hi - lo + 2 * q).mean()), 4)}
