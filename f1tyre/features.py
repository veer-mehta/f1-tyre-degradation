import numpy as np
import pandas as pd
from f1tyre.config import CIRCUIT_FEATURES, COMPOUNDS, DATA_DIR, DIRTY_AIR_GAP_S, NEUTRAL_PACE_RATIO, PRE_STINT_FEATURES, SLOW_LAP_LIMIT, TEST_YEARS, TRAIN_YEARS


def load_laps():
    return pd.concat([pd.read_csv(f) for f in sorted(DATA_DIR.glob("laps_*.csv"))], ignore_index=True)


def race_key(df):
    return df["Year"].astype(str) + "_" + df["Round"].astype(str).str.zfill(2)


def clean_laps(raw):
    df = raw.copy()
    df["race_id"] = race_key(df)
    df["TrackStatus"] = df["TrackStatus"].astype(str)
    mask = df["LapTime"].notna() & df["Compound"].isin(COMPOUNDS) & df["TyreLife"].notna()
    mask &= (df["LapNumber"] > 1) & ~df["PitInTime"] & ~df["PitOutTime"]
    mask &= (df["TrackStatus"] == "1") & df["IsAccurate"].eq(True) & ~df["Deleted"].eq(True)
    mask &= df["Rainfall"].eq(False)
    df = df[mask].copy()
    best = df.groupby(["race_id", "Driver"])["LapTime"].transform("min")
    return df[df["LapTime"] <= SLOW_LAP_LIMIT * best].reset_index(drop=True)


def split_years(laps):
    train = laps[laps["Year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test = laps[laps["Year"].isin(TEST_YEARS)].reset_index(drop=True)
    return train, test


def load_circuits():
    return pd.read_csv(DATA_DIR / "circuits.csv").set_index("EventName")


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


def add_stint_history(laps):
    key = ["race_id", "Driver"]
    ordered = laps.sort_values(key + ["Stint", "LapNumber"])
    summary = ordered.groupby(key + ["Stint"], as_index=False).agg(n=("delta", "size"), total=("delta", "sum"))
    summary["last3"] = ordered.groupby(key + ["Stint"])["delta"].apply(lambda s: s.tail(3).mean()).to_numpy()
    summary = summary.sort_values(key + ["Stint"]).reset_index(drop=True)
    by_race = summary.groupby(key)
    earlier_n = by_race["n"].cumsum() - summary["n"]
    summary["race_mean_delta_before"] = ((by_race["total"].cumsum() - summary["total"]) / earlier_n.replace(0, np.nan))
    summary["prev_stint_mean_delta"] = (summary["total"] / summary["n"]).groupby([summary[k] for k in key]).shift(1)
    summary["prev_stint_last3_delta"] = by_race["last3"].shift(1)
    summary["prev_stint_laps"] = by_race["n"].shift(1)
    return laps.merge(summary[key + ["Stint"] + PRE_STINT_FEATURES], on=key + ["Stint"], how="left")


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
    laps = laps.merge(load_circuits()[CIRCUIT_FEATURES].reset_index(), on="EventName", how="left")
    laps = add_stint_history(laps)
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
