"""Feature table: one row per clean lap, with the target and every model feature."""
import numpy as np
import pandas as pd
from f1tyre.config import COMPOUNDS, DIRTY_AIR_GAP_S, NEUTRAL_PACE_RATIO
from f1tyre.data import clean_laps, race_key


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
        laps[col] = laps[col].fillna(NEUTRAL_PACE_RATIO)
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


def what_if_rows(laps, event, compound, track_temp, air_temp, start_lap=10, dirty_air=False, max_age=30):
    total_laps = int(laps.loc[laps["EventName"] == event, "TotalLaps"].max())
    age = np.arange(1, max_age + 1)
    lap_number = start_lap + age
    rows = pd.DataFrame({"tyre_life": age, "lap_number": lap_number, "race_fraction": lap_number / total_laps, "stint": 2, "track_temp": track_temp, "air_temp": air_temp, "humidity": laps["humidity"].median(), "wind_speed": laps["wind_speed"].median(), "temp_gap": track_temp - air_temp, "fresh_tyre": 1.0})
    rows["tyre_life_sq"] = age ** 2
    rows["tyre_life_log"] = np.log1p(age)
    rows["age_le2"] = (age <= 2).astype(float)
    for c in COMPOUNDS:
        rows[f"compound_{c.lower()}"] = float(c == compound)
        rows[f"age_x_{c.lower()}"] = float(c == compound) * age
    for col in ["driver_form", "team_form", "driver_form_3", "driver_form_10", "team_form_3", "team_form_10"]:
        rows[col] = laps[col].median()
    gap = 0.8 if dirty_air else 10.0
    rows["gap_ahead_prev"] = rows["gap_ahead_prev2"] = gap
    rows["gap_behind_prev"] = 10.0
    rows["position_prev"] = 10.0
    rows["dirty_air_prev"] = float(dirty_air)
    rows["field_vs_cum"] = laps["field_vs_cum"].median()
    rows["laps_since_neutral"] = 40.0
    return rows, total_laps
