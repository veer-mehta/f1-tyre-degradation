"""Tyre-age degradation curves for a circuit, from the within-stint degradation model.

Usage: python predict_demo.py [CIRCUIT] [TRACK_TEMP_C]   e.g. python predict_demo.py "Spanish Grand Prix" 45
"""
import json
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from f1tyre import config, data, degradation, features, models


COLORS = {"SOFT": "red", "MEDIUM": "goldenrod", "HARD": "dimgray"}


def fit_model():
    laps = features.build_lap_table(data.load_laps())
    circuits = data.load_circuits()
    report = json.loads((config.RESULTS_DIR / "degradation.json").read_text())
    panel = degradation.build_panel(laps, models.estimate_fuel_effect(laps))
    return degradation.model_from_report(report).fit(panel, circuits), circuits


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else "Bahrain Grand Prix"
    track_temp = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
    model, circuits = fit_model()
    ages = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for compound in config.COMPOUNDS:
        loss = model.curve(compound, circuits.loc[event], track_temp, ages)
        ax.plot(ages, loss, color=COLORS[compound], lw=2, label=compound.title())
        print(f"{compound.title()}: lap-time loss after 10 laps {loss[9]:.2f} s, after 20 laps {loss[19]:.2f} s")
    ax.set(xlabel="Tyre age (laps)", ylabel="Fuel-corrected lap-time loss vs age 1 (s)", title=f"{event}, track {track_temp:.0f} C")
    ax.legend()
    fig.tight_layout()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.RESULTS_DIR / "demo_curve.png", dpi=150)


if __name__ == "__main__":
    main()
