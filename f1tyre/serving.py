import gzip
import json
import pickle
import numpy as np
import pandas as pd
from f1tyre import degradation, features, models
from f1tyre.config import COMPOUNDS, DIRTY_AIR_GAP_S, NEUTRAL_PACE_RATIO, RESULTS_DIR, ROOT, TIERS


ARTIFACT_PATH = ROOT / "artifacts" / "app.pkl.gz"
SERVED_TIERS = ["pre_lap", "live"]
NO_NEUTRAL_LAPS = 40.0


def reference_defaults(laps, circuits):
    latest = laps[laps["Year"] == laps.groupby("EventName")["Year"].transform("max")]
    events = latest.groupby("EventName").agg(race_median_s=("field_med", "mean"), total_laps=("TotalLaps", "max"))
    return {"events": events, "temp_gap": float((laps["TrackTemp"] - laps["AirTemp"]).median()), "humidity": float(laps["Humidity"].median()), "wind_speed": float(laps["WindSpeed"].median()), "mean_climb": float(circuits["climb_m_per_km"].mean()), "total_laps": float(laps["TotalLaps"].median())}


def default_settings(defaults, circuit=None):
    events = defaults["events"]
    known = circuit in events.index
    return {"median_s": float(events.loc[circuit, "race_median_s"]) if known else None, "total_laps": float(events.loc[circuit, "total_laps"]) if known else defaults["total_laps"], "stint": 1, "fresh_tyre": True, "driver_ratio": NEUTRAL_PACE_RATIO, "gap_ahead_s": 5.0, "gap_behind_s": 5.0, "position": 10, "field_vs_cum": 0.0, "air_temp": None, "humidity": defaults["humidity"], "wind_speed": defaults["wind_speed"], "laps_since_neutral": NO_NEUTRAL_LAPS}


def clean_rows(rows):
    out = rows.copy()
    out["Compound"] = out["Compound"].astype(str).str.upper().str.strip()
    for column in ["LapNumber", "LapTime", "TyreLife", "TrackTemp"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[out["Compound"].isin(COMPOUNDS) & out["LapNumber"].notna() & out["TyreLife"].notna() & out["TrackTemp"].notna()]
    return out.sort_values("LapNumber").reset_index(drop=True)


def earlier_laps(rows, median_s):
    prev_delta, prev3_delta, prev_gap = [], [], []
    known = []
    for lap_number, lap_time in zip(rows["LapNumber"], rows["LapTime"]):
        if known:
            deltas = [d for _, d in known]
            prev_delta.append(deltas[-1])
            prev3_delta.append(float(np.mean(deltas[-3:])))
            prev_gap.append(lap_number - known[-1][0])
        else:
            prev_delta.append(np.nan)
            prev3_delta.append(np.nan)
            prev_gap.append(np.nan)
        if not np.isnan(lap_time):
            known.append((lap_number, lap_time - median_s))
    return prev_delta, prev3_delta, prev_gap


def build_features(rows, settings, climb, defaults):
    rows = clean_rows(rows)
    age = rows["TyreLife"].to_numpy(dtype=float)
    track_temp = rows["TrackTemp"].to_numpy(dtype=float)
    air_temp = np.full(len(rows), settings["air_temp"]) if settings["air_temp"] is not None else track_temp - defaults["temp_gap"]
    gap_ahead = float(settings["gap_ahead_s"])
    out = pd.DataFrame(index=rows.index)
    out["tyre_life"] = age
    out["tyre_life_sq"] = age ** 2
    out["tyre_life_log"] = np.log1p(age)
    out["age_le2"] = (age <= 2).astype(float)
    out["lap_number"] = rows["LapNumber"].to_numpy(dtype=float)
    out["race_fraction"] = out["lap_number"] / float(settings["total_laps"])
    out["stint"] = float(settings["stint"])
    out["track_temp"] = track_temp
    out["air_temp"] = air_temp
    out["humidity"] = float(settings["humidity"])
    out["wind_speed"] = float(settings["wind_speed"])
    out["temp_gap"] = track_temp - air_temp
    out["fresh_tyre"] = float(settings["fresh_tyre"])
    for compound in COMPOUNDS:
        flag = (rows["Compound"] == compound).to_numpy(dtype=float)
        out[f"compound_{compound.lower()}"] = flag
        out[f"age_x_{compound.lower()}"] = flag * age
    for name in ["driver_form", "team_form", "driver_form_3", "driver_form_10", "team_form_3", "team_form_10"]:
        out[name] = float(settings["driver_ratio"])
    out["gap_ahead_prev"] = gap_ahead
    out["gap_ahead_prev2"] = gap_ahead
    out["gap_behind_prev"] = float(settings["gap_behind_s"])
    out["position_prev"] = float(settings["position"])
    out["dirty_air_prev"] = float(gap_ahead < DIRTY_AIR_GAP_S)
    out["field_vs_cum"] = float(settings["field_vs_cum"])
    out["laps_since_neutral"] = float(settings["laps_since_neutral"])
    out["climb_m_per_km"] = float(climb)
    for name in TIERS["live"]:
        if name not in out:
            out[name] = np.nan
    out["prev_delta"], out["prev3_delta"], out["prev_lap_gap"] = earlier_laps(rows, settings["median_s"])
    return rows, out[TIERS["live"]]


def predict_laps(served, built, median_s):
    return {tier: served[tier].predict(built[TIERS[tier]].to_numpy()) + median_s for tier in SERVED_TIERS}


def example_rows(laps):
    latest = laps[laps["Year"] == laps["Year"].max()]
    sub = latest[latest["EventName"] == "Bahrain Grand Prix"]
    sub = sub if len(sub) else latest
    driver, stint = sub.groupby(["Driver", "Stint"]).size().idxmax()
    stint_laps = sub[(sub["Driver"] == driver) & (sub["Stint"] == stint)].sort_values("LapNumber")
    overrides = {"stint": int(stint), "driver_ratio": float(stint_laps["driver_form"].iloc[0]), "position": int(round(stint_laps["position_prev"].mean())), "gap_ahead_s": float(stint_laps["gap_ahead_prev"].mean()), "gap_behind_s": float(stint_laps["gap_behind_prev"].mean()), "field_vs_cum": float(stint_laps["field_vs_cum"].mean()), "laps_since_neutral": float(stint_laps["laps_since_neutral"].mean()), "air_temp": float(stint_laps["air_temp"].mean()), "humidity": float(stint_laps["humidity"].mean()), "wind_speed": float(stint_laps["wind_speed"].mean()), "fresh_tyre": bool(stint_laps["fresh_tyre"].iloc[0] > 0.5)}
    return stint_laps[["LapNumber", "LapTime", "Compound", "TyreLife", "TrackTemp"]].round(3).reset_index(drop=True), overrides


def stint_spread(model, circuits, panel):
    predicted = pd.Series(model.predict_within(panel, circuits), index=panel.index)
    gap = degradation.stint_slopes(panel["y"], panel) - degradation.stint_slopes(predicted, panel)
    compound = panel.groupby(degradation.STINT_KEY)["Compound"].first()
    return {c: (float(gap[compound == c].quantile(0.1)), float(gap[compound == c].quantile(0.9))) for c in COMPOUNDS}


def verify_scores(verify):
    x, y = verify["x"], verify["y"]
    preds = {"Median baseline": np.full(len(y), verify["median"]), "Persistence (last 3 laps)": x["prev3_delta"].fillna(verify["median"]).to_numpy()}
    for tier, name in [("pre_lap", "Gradient boosting, pre-lap"), ("live", "Gradient boosting, live")]:
        preds[name] = verify["models"][tier].predict(x[TIERS[tier]].to_numpy())
    return {name: (models.mae(y, p), models.r2(y, p)) for name, p in preds.items()}


def build():
    laps = features.build_lap_table(features.load_laps())
    train, test = features.split_years(laps)
    verify = {"models": {tier: models.fit_gb(train, TIERS[tier], tier) for tier in SERVED_TIERS}, "x": test[TIERS["live"]].reset_index(drop=True), "y": test["delta"].to_numpy(), "median": float(train["delta"].median())}
    circuits = features.load_circuits()
    report = json.loads((RESULTS_DIR / "degradation.json").read_text())
    fuel_total_s = models.estimate_fuel_effect(laps)
    panel = degradation.build_panel(laps, fuel_total_s)
    model = degradation.model_from_report(report).fit(panel, circuits)
    return {"degradation": model, "circuits": circuits, "selected": report["final_model"]["circuit_columns"], "draws": pd.read_csv(RESULTS_DIR / "degradation_bootstrap.csv"), "totals": laps.groupby("EventName")["TotalLaps"].max().to_dict(), "fuel_total_s": fuel_total_s, "panel": panel, "spread": stint_spread(model, circuits, panel), "served": {tier: models.fit_gb(laps, TIERS[tier], tier) for tier in SERVED_TIERS}, "defaults": reference_defaults(laps, circuits), "example": example_rows(laps), "verify": verify}


def refresh():
    artifacts = build()
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    with gzip.open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    return artifacts


def load():
    if ARTIFACT_PATH.exists():
        try:
            with gzip.open(ARTIFACT_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return refresh()
