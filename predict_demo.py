import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from f1tyre import config, data, features, models


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else "Bahrain Grand Prix"
    track_temp = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
    air_temp = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    laps = features.build_lap_table(data.load_laps())
    model = models.fit_gbt(laps, config.TIERS["pre_lap"], "pre_lap")
    curves = models.predict_degradation(model, laps, models.estimate_fuel_effect(laps), event, track_temp, air_temp)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (compound, g), color in zip(curves.groupby("compound", sort=False), ["red", "goldenrod", "dimgray"]):
        ax.plot(g["age"], g["loss"], color=color, lw=2, label=compound)
        print(f"{compound}: predicted lap time loss after 10 laps {g['loss'].iloc[9]:.2f} s, after 20 laps {g['loss'].iloc[19]:.2f} s")
    ax.set(xlabel="Tyre age (laps)", ylabel="Fuel-corrected lap time loss vs age 2-4 (s)", title=f"{event}, track {track_temp:.0f} C, air {air_temp:.0f} C")
    ax.legend()
    fig.tight_layout()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.RESULTS_DIR / "demo_curve.png", dpi=150)


if __name__ == "__main__":
    main()
