import os
import shutil
import sys
import tempfile
import warnings
import fastf1
import numpy as np
import pandas as pd
from f1tyre import features
from f1tyre.config import DATA_DIR, ROOT


warnings.filterwarnings("ignore")
DEFAULT_YEARS = [2022, 2023, 2024, 2025]
LAP_COLUMNS = ["Driver", "Team", "LapNumber", "LapTime", "Compound", "TyreLife", "FreshTyre", "Stint", "TrackStatus", "IsAccurate", "PitInTime", "PitOutTime", "Deleted", "Time", "Position"]
WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]


def load_env(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def add_gaps(df):
    df["TimeS"] = df["Time"].dt.total_seconds()
    key = df[["LapNumber", "Position", "TimeS"]].dropna().drop_duplicates(["LapNumber", "Position"])
    ahead = key.rename(columns={"TimeS": "TimeAhead"})
    ahead["Position"] = ahead["Position"] + 1
    behind = key.rename(columns={"TimeS": "TimeBehind"})
    behind["Position"] = behind["Position"] - 1
    df = df.merge(ahead, on=["LapNumber", "Position"], how="left").merge(behind, on=["LapNumber", "Position"], how="left")
    df["GapAhead"] = (df["TimeS"] - df["TimeAhead"]).clip(upper=60)
    df["GapBehind"] = (df["TimeBehind"] - df["TimeS"]).clip(upper=60)
    by_driver = df.sort_values(["Driver", "LapNumber"]).groupby("Driver")
    df["GapAheadPrev"] = by_driver["GapAhead"].shift(1)
    df["GapAheadPrev2"] = by_driver["GapAhead"].shift(2)
    df["GapBehindPrev"] = by_driver["GapBehind"].shift(1)
    df["PositionPrev"] = by_driver["Position"].shift(1)
    return df.drop(columns=["Time", "TimeS", "TimeAhead", "TimeBehind", "GapAhead", "GapBehind"])


def fetch_race(year, round_number):
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=True, messages=False)
    laps = session.laps
    weather = laps.get_weather_data().reset_index(drop=True)
    df = pd.DataFrame(laps[LAP_COLUMNS]).reset_index(drop=True)
    df["LapTime"] = df["LapTime"].dt.total_seconds()
    df["PitInTime"] = df["PitInTime"].notna()
    df["PitOutTime"] = df["PitOutTime"].notna()
    for col in WEATHER_COLUMNS:
        df[col] = weather[col].values
    df = add_gaps(df)
    df["Year"] = year
    df["Round"] = round_number
    df["EventName"] = session.event["EventName"]
    df["TotalLaps"] = df["LapNumber"].max()
    return df


def fetch_if_missing(year, round_number):
    path = DATA_DIR / f"laps_{year}_{round_number:02d}.csv"
    if path.exists():
        return
    try:
        fetch_race(year, round_number).to_csv(path, index=False)
        print(f"ok {path.name}", flush=True)
    except Exception as error:
        print(f"fail {path.name}: {error}", flush=True)


def fetch_year(year):
    cache_dir = ROOT / "cache" / str(year)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for round_number in schedule["RoundNumber"]:
        fetch_if_missing(year, int(round_number))


GRID_STEP_M = 5.0
G = 9.81


def moving_average(values, window):
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def local_minima(speed, half_window, prominence_window, min_prominence):
    n = len(speed)
    view = np.lib.stride_tricks.sliding_window_view(np.pad(speed, half_window, mode="edge"), 2 * half_window + 1)
    is_min = speed <= view.min(axis=1)
    wide = np.lib.stride_tricks.sliding_window_view(np.pad(speed, prominence_window, mode="edge"), 2 * prominence_window + 1)
    prominent = (wide.max(axis=1) - speed) >= min_prominence
    return np.flatnonzero(is_min & prominent & (np.arange(n) > 0) & (np.arange(n) < n - 1))


def braking_zones(brake_flag, speed, min_drop_kmh):
    zones = 0
    start = None
    for i, braking in enumerate(brake_flag):
        if braking and start is None:
            start = i
        if (not braking or i == len(brake_flag) - 1) and start is not None:
            if speed[start] - speed[start : i + 1].min() >= min_drop_kmh:
                zones += 1
            start = None
    return zones


def profile_from_telemetry(tel, lap_time_s):
    tel = tel.sort_values("Distance").drop_duplicates("Distance")
    distance = tel["Distance"].to_numpy(dtype=float)
    length_m = float(distance.max())
    time_s = tel["Time"].dt.total_seconds().to_numpy() if hasattr(tel["Time"], "dt") else tel["Time"].to_numpy(dtype=float)
    dt = np.diff(time_s, append=time_s[-1])
    total_time = dt.sum()
    grid = np.arange(0.0, length_m, GRID_STEP_M)
    speed = np.interp(grid, distance, tel["Speed"].to_numpy(dtype=float))
    brake = np.interp(grid, distance, tel["Brake"].to_numpy(dtype=float)) > 0.5
    x = moving_average(np.interp(grid, distance, tel["X"].to_numpy(dtype=float)) / 10.0, 5)
    y = moving_average(np.interp(grid, distance, tel["Y"].to_numpy(dtype=float)) / 10.0, 5)
    z = moving_average(np.interp(grid, distance, tel["Z"].to_numpy(dtype=float)) / 10.0, 9)
    heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    curvature = moving_average(np.gradient(heading) / GRID_STEP_M, 5)
    v = np.maximum(speed / 3.6, 1.0)
    lateral_g = v ** 2 * np.abs(curvature) / G
    weight = GRID_STEP_M / v
    weight = weight / weight.sum()
    smooth_speed = moving_average(speed, 5)
    corners = local_minima(smooth_speed, half_window=15, prominence_window=60, min_prominence=25.0)
    corner_speed = smooth_speed[corners]
    return {
        "length_km": length_m / 1000.0,
        "avg_speed_kmh": length_m / lap_time_s * 3.6,
        "top_speed_kmh": float(speed.max()),
        "min_speed_kmh": float(speed.min()),
        "full_throttle_pct": 100.0 * float(dt[tel["Throttle"].to_numpy(dtype=float) >= 98].sum() / total_time),
        "brake_pct": 100.0 * float(dt[tel["Brake"].to_numpy(dtype=float) > 0.5].sum() / total_time),
        "heavy_braking_zones": braking_zones(brake, speed, 100.0),
        "mean_lat_g": float((weight * lateral_g).sum()),
        "high_lat_pct": 100.0 * float(weight[lateral_g > 2.5].sum()),
        "slow_corners": int((corner_speed < 120).sum()),
        "medium_corners": int(((corner_speed >= 120) & (corner_speed < 200)).sum()),
        "fast_corners": int((corner_speed >= 200).sum()),
        "climb_m_per_km": float(np.abs(np.diff(z)).sum() / (length_m / 1000.0)),
    }


NEEDED = ["Distance", "Speed", "Throttle", "Brake", "X", "Y", "Z", "Time"]


def usable_lap(session, max_tries=10):
    quick = session.laps.pick_quicklaps().sort_values("LapTime")
    for _, lap in quick.head(max_tries).iterrows():
        tel = session.laps.loc[[lap.name]].iloc[0].get_telemetry()
        if tel[NEEDED].notna().all().all() and tel["Distance"].max() > 2000:
            return lap["LapTime"].total_seconds(), tel
    raise RuntimeError("no lap with complete telemetry")


def profile_event(event, year, round_number):
    cache_dir = tempfile.mkdtemp(prefix="f1cache_")
    try:
        fastf1.Cache.enable_cache(cache_dir)
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        lap_time_s, tel = usable_lap(session)
        row = profile_from_telemetry(tel, lap_time_s)
        row.update({"EventName": event, "ProfileYear": year})
        return row
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def fetch_circuits():
    raw = features.load_laps()
    latest = raw.groupby("EventName")["Year"].max()
    out_path = DATA_DIR / "circuits.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    finished = set(done["EventName"]) if len(done) else set()
    for event, year in latest.items():
        if event in finished:
            continue
        round_number = int(raw[(raw["EventName"] == event) & (raw["Year"] == year)]["Round"].iloc[0])
        try:
            rows.append(profile_event(event, int(year), round_number))
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"ok {event} {year}", flush=True)
        except Exception as error:
            print(f"fail {event} {year}: {error}", flush=True)


def main():
    load_env()
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME and F1_PASSWORD (environment or .env) to download data from FastF1.")
    DATA_DIR.mkdir(exist_ok=True)
    if sys.argv[1:] == ["circuits"]:
        fetch_circuits()
        return
    for year in [int(y) for y in sys.argv[1:]] or DEFAULT_YEARS:
        fetch_year(year)


if __name__ == "__main__":
    main()
