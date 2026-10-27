import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pipeline as p


START_LAP = 10
MAX_AGE = 30


def synthetic_laps(laps, event, compound, track_temp, air_temp):
    total_laps = int(laps.loc[laps["EventName"] == event, "TotalLaps"].max())
    age = np.arange(1, MAX_AGE + 1)
    lap_number = START_LAP + age
    rows = pd.DataFrame({"tyre_life": age, "lap_number": lap_number, "race_fraction": lap_number / total_laps, "stint": 2, "track_temp": track_temp, "air_temp": air_temp, "humidity": laps["humidity"].median(), "wind_speed": laps["wind_speed"].median(), "temp_gap": track_temp - air_temp, "fresh_tyre": 1.0})
    rows["tyre_life_sq"] = age ** 2
    rows["tyre_life_log"] = np.log1p(age)
    rows["age_le2"] = (age <= 2).astype(float)
    for c in p.COMPOUNDS:
        rows[f"compound_{c.lower()}"] = float(c == compound)
        rows[f"age_x_{c.lower()}"] = float(c == compound) * age
    for col in ["driver_form", "team_form", "driver_form_3", "driver_form_10", "team_form_3", "team_form_10"]:
        rows[col] = laps[col].median()
    rows["gap_ahead_prev"] = rows["gap_ahead_prev2"] = rows["gap_behind_prev"] = 10.0
    rows["position_prev"] = 10.0
    rows["dirty_air_prev"] = 0.0
    rows["field_vs_cum"] = laps["field_vs_cum"].median()
    rows["laps_since_neutral"] = 40.0
    return rows, total_laps


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else "Bahrain Grand Prix"
    track_temp = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
    air_temp = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    laps = p.build_lap_table(p.load_laps())
    model = p.fit_xgb(laps, p.TIERS["pre_lap"], "pre_lap")
    fuel_total_s = p.estimate_fuel_effect(laps)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for compound, color in zip(p.COMPOUNDS, ["red", "goldenrod", "dimgray"]):
        rows, total_laps = synthetic_laps(laps, event, compound, track_temp, air_temp)
        pred = model.predict(rows[p.TIERS["pre_lap"]]) + fuel_total_s * rows["lap_number"].values / total_laps
        loss = pred - pred[2:4].mean()
        ax.plot(rows["tyre_life"], loss, color=color, lw=2, label=compound)
        print(f"{compound}: predicted lap time loss after 10 laps {loss[9]:.2f} s, after 20 laps {loss[19]:.2f} s")
    ax.set(xlabel="Tyre age (laps)", ylabel="Fuel-corrected lap time loss vs age 2-4 (s)", title=f"{event}, track {track_temp:.0f} C, air {air_temp:.0f} C")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "results" / "demo_curve.png", dpi=150)


if __name__ == "__main__":
    main()
