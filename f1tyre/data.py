"""Loading and cleaning the raw lap data."""
import pandas as pd
from f1tyre.config import COMPOUNDS, DATA_DIR, SLOW_LAP_LIMIT, TEST_YEARS, TRAIN_YEARS


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
