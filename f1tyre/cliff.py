"""Tyre cliff detection: does a stint's lap-time loss speed up late in the stint?

Every long stint is fitted with a straight line and with a straight line plus one hinge (the slope
changes at a knee age). A cliff is a hinge that improves the fit clearly (BIC) and adds a large extra
slope, with the knee in the second half of the stint (an early knee is a warm-up shape, not a cliff).
Slowing down by the same amount is just as likely to appear by chance as speeding up, so the share of
stints that "slow down" late is reported next to the cliff share as the noise reference.
"""
import numpy as np
import pandas as pd
from f1tyre.degradation import STINT_KEY


MIN_CLIFF_LAPS = 20
MIN_LAPS_BEFORE = 8
MIN_LAPS_AFTER = 4
MIN_BIC_GAIN = 10.0
MIN_EXTRA_SLOPE = 0.10
MIN_KNEE_POSITION = 0.5


def best_break(age, y, min_before=MIN_LAPS_BEFORE, min_after=MIN_LAPS_AFTER):
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(age)
    ages = np.unique(age)
    candidates = ages[min_before - 1 : len(ages) - min_after]
    if len(candidates) == 0:
        return None
    line = np.column_stack([np.ones(n), age])
    sse_line = max(float(((y - line @ np.linalg.lstsq(line, y, rcond=None)[0]) ** 2).sum()), 1e-12)
    best = None
    for knee in candidates:
        design = np.column_stack([line, np.maximum(age - knee, 0.0)])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        sse = max(float(((y - design @ coef) ** 2).sum()), 1e-12)
        if best is None or sse < best[1]:
            best = (float(knee), sse, float(coef[2]))
    knee, sse, extra = best
    bic_gain = n * (np.log(sse_line / n) - np.log(sse / n)) - 2 * np.log(n)
    return {"knee": knee, "extra_slope": extra, "bic_gain": float(bic_gain)}


def find_cliffs(panel, min_laps=MIN_CLIFF_LAPS):
    rows = []
    for key, stint in panel.groupby(STINT_KEY):
        if len(stint) < min_laps:
            continue
        found = best_break(stint["age"].to_numpy(), stint["y"].to_numpy())
        if found is None:
            continue
        first, last = stint["age"].min(), stint["age"].max()
        rows.append({"race_id": key[0], "Driver": key[1], "Stint": key[2], "Compound": stint["Compound"].iloc[0], "Year": stint["Year"].iloc[0], "laps": len(stint), **found, "knee_position": (found["knee"] - first) / (last - first)})
    table = pd.DataFrame(rows, columns=["race_id", "Driver", "Stint", "Compound", "Year", "laps", "knee", "extra_slope", "bic_gain", "knee_position"])
    clear = (table["bic_gain"] > MIN_BIC_GAIN) & (table["knee_position"] >= MIN_KNEE_POSITION)
    table["cliff"] = clear & (table["extra_slope"] >= MIN_EXTRA_SLOPE)
    table["relief"] = clear & (table["extra_slope"] <= -MIN_EXTRA_SLOPE)
    return table


def summarise_cliffs(table):
    out = {}
    groups = [("ALL", table)] + [(c, table[table["Compound"] == c]) for c in ["SOFT", "MEDIUM", "HARD"]]
    for name, group in groups:
        cliffs = group[group["cliff"]]
        out[name] = {"stints": int(len(group)), "cliff_share": round(float(group["cliff"].mean()), 4) if len(group) else 0.0, "relief_share_noise_reference": round(float(group["relief"].mean()), 4) if len(group) else 0.0, "cliff_stints": int(len(cliffs)), "median_onset_age": float(cliffs["knee"].median()) if len(cliffs) else None, "median_extra_slope_s_per_lap": round(float(cliffs["extra_slope"].median()), 3) if len(cliffs) else None}
    return out
