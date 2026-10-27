"""Lap-time prediction from user-entered laps: saved gradient boosting models, feature building and defaults.

The models predict a lap relative to the race's field-median lap, so a race median lap time is needed to turn
a prediction into seconds. Inputs the user does not give (weather, traffic, driver form) take neutral defaults.
"""
import pickle
import numpy as np
import pandas as pd
from f1tyre import models as fitted_models
from f1tyre.config import COMPOUNDS, DIRTY_AIR_GAP_S, NEUTRAL_PACE_RATIO, ROOT, TIERS


MODEL_DIR = ROOT / "models"
SERVED_TIERS = ["pre_lap", "live"]
NO_GAP_S = 60.0
NO_NEUTRAL_LAPS = 40.0


def model_path(tier):
    return MODEL_DIR / f"gb_{tier}.pkl"


def train_and_save(laps):
    MODEL_DIR.mkdir(exist_ok=True)
    out = {}
    for tier in SERVED_TIERS:
        out[tier] = fitted_models.fit_gb(laps, TIERS[tier], tier)
        with open(model_path(tier), "wb") as f:
            pickle.dump(out[tier], f)
    return out


def load_models(laps=None):
    if all(model_path(t).exists() for t in SERVED_TIERS):
        out = {}
        for tier in SERVED_TIERS:
            with open(model_path(tier), "rb") as f:
                out[tier] = pickle.load(f)
        return out
    return train_and_save(laps)


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


def predict_laps(served, features, median_s):
    return {tier: served[tier].predict(features[TIERS[tier]].to_numpy()) + median_s for tier in SERVED_TIERS}
