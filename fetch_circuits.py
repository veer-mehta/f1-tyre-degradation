"""Compute circuit descriptors from FastF1 telemetry (fastest race lap per circuit) into data/circuits.csv.

Usage: python fetch_circuits.py      (needs F1_USERNAME and F1_PASSWORD, see fetch_data.py)
"""
import os
import shutil
import sys
import tempfile
import warnings
import pandas as pd
import fastf1
from f1tyre import circuits, data
from f1tyre.config import DATA_DIR
from fetch_data import load_env


warnings.filterwarnings("ignore")
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
        row = circuits.profile_from_telemetry(tel, lap_time_s)
        row.update({"EventName": event, "ProfileYear": year})
        return row
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def main():
    load_env()
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME and F1_PASSWORD (environment or .env).")
    raw = data.load_laps()
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


if __name__ == "__main__":
    main()
