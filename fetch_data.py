"""Download race laps, weather and gap features from FastF1 into data/.

Usage: python fetch_data.py [YEAR ...]   (default: 2022 2023 2024 2025)
FastF1 needs an F1 account: set F1_USERNAME and F1_PASSWORD in the environment or in .env.
"""
import os
import sys
import warnings
import pandas as pd
import fastf1
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


def main():
    load_env()
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME and F1_PASSWORD (environment or .env) to download data from FastF1.")
    years = [int(y) for y in sys.argv[1:]] or DEFAULT_YEARS
    DATA_DIR.mkdir(exist_ok=True)
    for year in years:
        fetch_year(year)


if __name__ == "__main__":
    main()
