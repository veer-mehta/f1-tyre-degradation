from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from xgboost import XGBRegressor


TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEARS = [2025]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
SLOW_LAP_LIMIT = 1.04
DIRTY_AIR_GAP_S = 1.5
BASE_FEATURES = ["tyre_life", "tyre_life_sq", "tyre_life_log", "age_le2", "lap_number", "race_fraction", "stint", "track_temp", "air_temp", "humidity", "wind_speed", "temp_gap", "fresh_tyre", "compound_soft", "compound_medium", "compound_hard", "age_x_soft", "age_x_medium", "age_x_hard", "driver_form", "team_form", "driver_form_3", "driver_form_10", "team_form_3", "team_form_10"]
TRAFFIC_FEATURES = ["gap_ahead_prev", "gap_ahead_prev2", "gap_behind_prev", "position_prev", "dirty_air_prev"]
FIELD_FEATURES = ["field_vs_cum", "laps_since_neutral"]
LIVE_FEATURES = ["prev_delta", "prev3_delta", "prev_lap_gap"]
TIERS = {"pre_lap": BASE_FEATURES + TRAFFIC_FEATURES + FIELD_FEATURES, "live": BASE_FEATURES + TRAFFIC_FEATURES + FIELD_FEATURES + LIVE_FEATURES}
XGB_PARAMS = {"live": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20, "reg_lambda": 5.0}, "pre_lap": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.5, "colsample_bytree": 0.6, "min_child_weight": 100, "reg_lambda": 5.0, "reg_alpha": 1.0}}
FEATURE_GROUPS = {"tyre": ["tyre_life", "tyre_life_sq", "tyre_life_log", "age_le2", "fresh_tyre", "compound_soft", "compound_medium", "compound_hard", "age_x_soft", "age_x_medium", "age_x_hard", "stint"], "fuel_progress": ["lap_number", "race_fraction"], "weather": ["track_temp", "air_temp", "humidity", "wind_speed", "temp_gap"], "form": ["driver_form", "team_form", "driver_form_3", "driver_form_10", "team_form_3", "team_form_10"], "traffic": TRAFFIC_FEATURES, "field_context": FIELD_FEATURES}


def load_laps():
    return pd.concat([pd.read_csv(f) for f in sorted((Path(__file__).resolve().parent.parent / "data").glob("laps_*.csv"))], ignore_index=True)


def race_key(df):
    return df["Year"].astype(str) + "_" + df["Round"].astype(str).str.zfill(2)


def clean_laps(raw):
    df = raw.copy()
    df["race_id"] = race_key(df)
    df["TrackStatus"] = df["TrackStatus"].astype(str)
    mask = df["LapTime"].notna() & df["Compound"].isin(COMPOUNDS) & df["TyreLife"].notna()
    mask &= (df["LapNumber"] > 1) & ~df["PitInTime"] & ~df["PitOutTime"]
    mask &= (df["TrackStatus"] == "1") & (df["IsAccurate"] == True) & (df["Deleted"] != True)
    mask &= df["Rainfall"] == False
    df = df[mask].copy()
    best = df.groupby(["race_id", "Driver"])["LapTime"].transform("min")
    return df[df["LapTime"] <= SLOW_LAP_LIMIT * best].reset_index(drop=True)


def add_form(laps, key, window, name):
    table = laps.groupby(["order", key])["ratio"].median().reset_index().sort_values("order")
    table[name] = table.groupby(key)["ratio"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return laps.merge(table[["order", key, name]], on=["order", key], how="left")


def add_field_context(laps, raw):
    lap_mean = laps.groupby(["race_id", "LapNumber"])["delta"].mean()
    parts = []
    for race_id, s in lap_mean.groupby(level=0):
        s = s.droplevel(0)
        s = s.reindex(np.arange(1, int(s.index.max()) + 2))
        vs_cum = s.shift(1) - s.expanding(min_periods=1).mean().shift(2)
        parts.append(pd.DataFrame({"race_id": race_id, "LapNumber": s.index, "field_vs_cum": vs_cum.values}))
    laps = laps.merge(pd.concat(parts), on=["race_id", "LapNumber"], how="left")
    flagged = raw[raw["TrackStatus"].astype(str) != "1"]
    neutral = {race_id: np.sort(g["LapNumber"].unique()) for race_id, g in flagged.groupby(race_key(flagged))}
    race_ids = laps["race_id"].values
    lap_numbers = laps["LapNumber"].values
    since = np.full(len(laps), 40.0)
    for race_id, arr in neutral.items():
        mask = race_ids == race_id
        idx = np.searchsorted(arr, lap_numbers[mask], side="left")
        last = np.where(idx > 0, arr[np.maximum(idx - 1, 0)], -1000.0)
        since[mask] = np.minimum(lap_numbers[mask] - last, 40.0)
    laps["laps_since_neutral"] = since
    return laps


def build_lap_table(raw):
    laps = clean_laps(raw)
    laps["order"] = laps["Year"] * 100 + laps["Round"]
    laps["field_med"] = laps.groupby("race_id")["LapTime"].transform("median")
    laps["ratio"] = laps["LapTime"] / laps["field_med"]
    laps["delta"] = laps["LapTime"] - laps["field_med"]
    laps = add_field_context(laps, raw)
    for key, prefix in [("Driver", "driver_form"), ("Team", "team_form")]:
        for window, suffix in [(5, ""), (3, "_3"), (10, "_10")]:
            laps = add_form(laps, key, window, prefix + suffix)
    for col in ["team_form", "team_form_3", "team_form_10"]:
        laps[col] = laps[col].fillna(laps["ratio"].median())
    for col in ["driver_form", "driver_form_3", "driver_form_10"]:
        laps[col] = laps[col].fillna(laps[col.replace("driver", "team")])
    laps = laps.sort_values(["race_id", "Driver", "Stint", "LapNumber"]).reset_index(drop=True)
    stint = laps.groupby(["race_id", "Driver", "Stint"])
    laps["prev_delta"] = stint["delta"].shift(1)
    laps["prev3_delta"] = stint["delta"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    laps["prev_lap_gap"] = laps["LapNumber"] - stint["LapNumber"].shift(1)
    laps["gap_ahead_prev"] = laps["GapAheadPrev"].fillna(60.0)
    laps["gap_ahead_prev2"] = laps["GapAheadPrev2"].fillna(60.0)
    laps["gap_behind_prev"] = laps["GapBehindPrev"].fillna(60.0)
    laps["position_prev"] = laps["PositionPrev"].fillna(10.0)
    laps["dirty_air_prev"] = (laps["gap_ahead_prev"] < DIRTY_AIR_GAP_S).astype(float)
    laps["tyre_life"] = laps["TyreLife"]
    laps["tyre_life_sq"] = laps["TyreLife"] ** 2
    laps["tyre_life_log"] = np.log1p(laps["TyreLife"])
    laps["age_le2"] = (laps["TyreLife"] <= 2).astype(float)
    laps["lap_number"] = laps["LapNumber"]
    laps["race_fraction"] = laps["LapNumber"] / laps["TotalLaps"]
    laps["stint"] = laps["Stint"]
    laps["track_temp"] = laps["TrackTemp"]
    laps["air_temp"] = laps["AirTemp"]
    laps["humidity"] = laps["Humidity"]
    laps["wind_speed"] = laps["WindSpeed"]
    laps["temp_gap"] = laps["TrackTemp"] - laps["AirTemp"]
    laps["fresh_tyre"] = laps["FreshTyre"].astype(float)
    for compound in COMPOUNDS:
        flag = (laps["Compound"] == compound).astype(float)
        laps[f"compound_{compound.lower()}"] = flag
        laps[f"age_x_{compound.lower()}"] = flag * laps["TyreLife"]
    return laps


def split_years(laps):
    train = laps[laps["Year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test = laps[laps["Year"].isin(TEST_YEARS)].reset_index(drop=True)
    return train, test


def make_xgb(tier, **overrides):
    params = dict(XGB_PARAMS["live" if tier == "live" else "pre_lap"])
    params.update(overrides)
    return XGBRegressor(n_jobs=-1, random_state=42, **params)


def fit_xgb(train, cols, tier):
    return make_xgb(tier, objective="reg:absoluteerror").fit(train[cols], train["delta"])


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


def curve_error(test, pred, max_age=30, min_n=30):
    d = test[["Compound", "TyreLife", "delta"]].assign(pred=pred)
    errors = []
    for compound in COMPOUNDS:
        sub = d[(d["Compound"] == compound) & (d["TyreLife"] <= max_age)]
        g = sub.groupby("TyreLife").agg(actual=("delta", "mean"), pred=("pred", "mean"), n=("delta", "size"))
        g = g[g["n"] >= min_n]
        errors.append((g["actual"] - g["pred"]).abs())
    return float(pd.concat(errors).mean())


def stint_slopes(test, pred, fuel_total_s, min_laps=15):
    d = test[["race_id", "Driver", "Stint", "TyreLife", "delta", "LapNumber", "TotalLaps"]].assign(pred=pred)
    fuel = fuel_total_s * d["LapNumber"] / d["TotalLaps"]
    d["true_fc"] = d["delta"] + fuel
    d["pred_fc"] = d["pred"] + fuel
    rows = []
    for _, g in d.groupby(["race_id", "Driver", "Stint"]):
        if len(g) >= min_laps and g["TyreLife"].nunique() >= min_laps - 2:
            rows.append((np.polyfit(g["TyreLife"], g["true_fc"], 1)[0], np.polyfit(g["TyreLife"], g["pred_fc"], 1)[0]))
    return np.array(rows)


def score(test, pred, fuel_total_s=None):
    y = test["delta"].values
    out = {"mae_s": mean_absolute_error(y, pred), "median_ae_s": median_absolute_error(y, pred), "rmse_s": float(np.sqrt(mean_squared_error(y, pred))), "r2": r2_score(y, pred), "curve_mae_s": curve_error(test, pred)}
    if fuel_total_s is not None:
        s = stint_slopes(test, pred, fuel_total_s)
        out["slope_mae_s_per_lap"] = float(np.abs(s[:, 0] - s[:, 1]).mean())
        out["slope_corr"] = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) if np.std(s[:, 1]) > 0 else 0.0
    return {k: round(float(v), 4) for k, v in out.items()}


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
        model = LinearRegression().fit(formula_features(train, abr_train, full), train["delta"])
        preds[name] = model.predict(formula_features(test, abr_test, full))
    return preds


def conformal_interval(train, test, cols, tier, alpha=0.1, seed=42):
    rng = np.random.RandomState(seed)
    races = train["race_id"].unique()
    calib_races = rng.choice(races, size=int(0.2 * len(races)), replace=False)
    is_calib = train["race_id"].isin(calib_races)
    fit, calib = train[~is_calib], train[is_calib]
    lower = make_xgb(tier, objective="reg:quantileerror", quantile_alpha=alpha / 2).fit(fit[cols], fit["delta"])
    upper = make_xgb(tier, objective="reg:quantileerror", quantile_alpha=1 - alpha / 2).fit(fit[cols], fit["delta"])
    lo_c, hi_c = lower.predict(calib[cols]), upper.predict(calib[cols])
    scores = np.maximum(lo_c - calib["delta"].values, calib["delta"].values - hi_c)
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    q = float(np.quantile(scores, level))
    lo, hi = lower.predict(test[cols]), upper.predict(test[cols])
    y = test["delta"].values
    raw_cov = float(((y >= lo) & (y <= hi)).mean())
    cal_cov = float(((y >= lo - q) & (y <= hi + q)).mean())
    return {"target_coverage": 1 - alpha, "raw_coverage": round(raw_cov, 4), "calibrated_coverage": round(cal_cov, 4), "calibrated_mean_width_s": round(float((hi - lo + 2 * q).mean()), 4)}
