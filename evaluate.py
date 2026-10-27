import json
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from f1tyre import config, features, models


RESULTS_DIR = config.RESULTS_DIR
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 42
MODEL_ORDER = ["median_baseline", "persistence_last3_laps", "formula_a_calculator_style", "formula_b_with_fuel_and_form", "ridge_pre_lap_features", "gb_pre_lap", "gb_pre_stint", "gb_live"]


def curve_error(test, pred, max_age=30, min_n=30):
    d = test[["Compound", "TyreLife", "delta"]].assign(pred=pred)
    errors = []
    for compound in config.COMPOUNDS:
        sub = d[(d["Compound"] == compound) & (d["TyreLife"] <= max_age)]
        g = sub.groupby("TyreLife").agg(actual=("delta", "mean"), pred=("pred", "mean"), n=("delta", "size"))
        g = g[g["n"] >= min_n]
        errors.append((g["actual"] - g["pred"]).abs())
    return float(pd.concat(errors).mean())


def stint_slopes(test, pred, fuel_total_s, min_laps=15):
    d = test[["race_id", "Driver", "Stint", "TyreLife", "delta", "LapNumber", "TotalLaps"]].assign(pred=pred)
    fuel = fuel_total_s * d["LapNumber"] / d["TotalLaps"]
    d["true_fc"] = d["delta"] + fuel
    d["pred_fc"] = d["pred"] + fuel
    rows = []
    for _, g in d.groupby(["race_id", "Driver", "Stint"]):
        if len(g) >= min_laps and g["TyreLife"].nunique() >= min_laps - 2:
            rows.append((np.polyfit(g["TyreLife"], g["true_fc"], 1)[0], np.polyfit(g["TyreLife"], g["pred_fc"], 1)[0]))
    return np.array(rows)


def score(test, pred, fuel_total_s=None):
    y = test["delta"].values
    out = {"mae_s": models.mae(y, pred), "median_ae_s": models.median_ae(y, pred), "rmse_s": models.rmse(y, pred), "r2": models.r2(y, pred), "curve_mae_s": curve_error(test, pred)}
    if fuel_total_s is not None:
        s = stint_slopes(test, pred, fuel_total_s)
        out["slope_mae_s_per_lap"] = float(np.abs(s[:, 0] - s[:, 1]).mean())
        out["slope_corr"] = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) if np.std(s[:, 1]) > 0 else 0.0
    return {k: round(float(v), 4) for k, v in out.items()}


def r4(value):
    return round(float(value), 4)


def mae_r2(y, pred):
    return {"mae_s": r4(models.mae(y, pred)), "r2": r4(models.r2(y, pred))}


def assign_folds(laps):
    rng = np.random.RandomState(SEED)
    fold_of_race = {}
    for _, g in laps.groupby("Year"):
        races = sorted(g["race_id"].unique())
        rng.shuffle(races)
        fold_of_race.update({race: i % N_FOLDS for i, race in enumerate(races)})
    return laps["race_id"].map(fold_of_race)


def predict_all(train, test):
    fuel_total_s = models.estimate_fuel_effect(train)
    median = train["delta"].median()
    preds = {"median_baseline": np.full(len(test), median), "persistence_last3_laps": test["prev3_delta"].fillna(median).values}
    preds.update(models.fit_formulas(train, test, fuel_total_s))
    cols = config.TIERS["pre_lap"]
    preds["ridge_pre_lap_features"] = models.RidgeRegression().fit(train[cols].values, train["delta"].values).predict(test[cols].values)
    fitted = {}
    for tier in config.TIERS:
        fitted[tier] = models.fit_gb(train, config.TIERS[tier], tier)
        preds[f"gb_{tier}"] = fitted[tier].predict(test[config.TIERS[tier]])
    return preds, fitted, fuel_total_s


def mixed_year_cv(laps):
    folds = assign_folds(laps)
    oof = {name: np.zeros(len(laps)) for name in MODEL_ORDER}
    fold_mae = {name: [] for name in MODEL_ORDER}
    for fold in range(N_FOLDS):
        is_test = (folds == fold).values
        preds, _, _ = predict_all(laps[~is_test].reset_index(drop=True), laps[is_test].reset_index(drop=True))
        for name in MODEL_ORDER:
            oof[name][is_test] = preds[name]
            fold_mae[name].append(models.mae(laps.loc[is_test, "delta"], preds[name]))
        print("mixed-year fold", fold, flush=True)
    out = {}
    for name in MODEL_ORDER:
        out[name] = score(laps, oof[name])
        out[name]["fold_mae_std"] = r4(np.std(fold_mae[name], ddof=1))
        out[name]["per_year_mae"] = {int(y): r4(models.mae(laps.loc[laps["Year"] == y, "delta"], oof[name][(laps["Year"] == y).values])) for y in sorted(laps["Year"].unique())}
    return out


def leave_one_year_out(laps):
    out = {}
    for year in sorted(laps["Year"].unique()):
        is_test = (laps["Year"] == year).values
        train, test = laps[~is_test].reset_index(drop=True), laps[is_test].reset_index(drop=True)
        out[int(year)] = {f"gb_{tier}": mae_r2(test["delta"], models.fit_gb(train, config.TIERS[tier], tier).predict(test[config.TIERS[tier]])) for tier in config.TIERS}
        print("leave-one-year-out", year, flush=True)
    return out


def later_stints(test, preds):
    mask = (test["stint"] >= 2).to_numpy()
    subset = test[mask].reset_index(drop=True)
    names = ["median_baseline", "persistence_last3_laps", "gb_pre_lap", "gb_pre_stint", "gb_live"]
    return {"share_of_test_laps": r4(mask.mean()), "scores": {name: score(subset, preds[name][mask]) for name in names}}


def label_shuffle(train, test):
    cols = config.TIERS["pre_lap"]
    shuffled = train.assign(delta=np.random.RandomState(SEED).permutation(train["delta"].values))
    return mae_r2(test["delta"], models.fit_gb(shuffled, cols, "pre_lap").predict(test[cols]))


def train_vs_test(train, test, preds, fitted):
    return {tier: {"train_mae_s": r4(models.mae(train["delta"], fitted[tier].predict(train[config.TIERS[tier]]))), "test_mae_s": r4(models.mae(test["delta"], preds[f"gb_{tier}"]))} for tier in config.TIERS}


def ablation(train, test, preds):
    cols = config.TIERS["pre_lap"]
    out = {"all_features": r4(models.mae(test["delta"], preds["gb_pre_lap"]))}
    for group, group_cols in config.FEATURE_GROUPS.items():
        kept = [c for c in cols if c not in group_cols]
        out[f"drop_{group}"] = r4(models.mae(test["delta"], models.fit_gb(train, kept, "pre_lap").predict(test[kept])))
        print("ablation", group, flush=True)
    return out


def race_bootstrap(test, preds):
    rng = np.random.RandomState(SEED)
    races = test["race_id"].unique()
    rows_of = {r: np.where(test["race_id"].values == r)[0] for r in races}
    abs_err = {name: np.abs(test["delta"].values - preds[name]) for name in MODEL_ORDER}
    draws = {name: [] for name in MODEL_ORDER}
    for _ in range(N_BOOTSTRAP):
        rows = np.concatenate([rows_of[r] for r in rng.choice(races, size=len(races), replace=True)])
        for name in MODEL_ORDER:
            draws[name].append(abs_err[name][rows].mean())
    interval = lambda values: [r4(np.percentile(values, 2.5)), r4(np.percentile(values, 97.5))]
    gain = {}
    for name in ["gb_pre_lap", "gb_pre_stint", "gb_live"]:
        diff = np.array(draws["formula_b_with_fuel_and_form"]) - np.array(draws[name])
        gain[name] = {"mean_s": r4(diff.mean()), "ci95": interval(diff), "share_better": round(float((diff > 0).mean()), 3)}
    return {"mae_ci95": {name: interval(draws[name]) for name in MODEL_ORDER}, "gain_over_formula_b": gain}


def plot_predictions(test, preds):
    fig, axes = plt.subplots(1, len(config.TIERS), figsize=(5.4 * len(config.TIERS), 5))
    for ax, tier in zip(axes, config.TIERS):
        ax.scatter(test["delta"], preds[f"gb_{tier}"], s=2, alpha=0.15)
        ax.plot([-3, 6], [-3, 6], color="red", lw=1)
        ax.set(xlim=(-3, 6), ylim=(-3, 6), xlabel="Actual lap time minus field median (s)", ylabel="Predicted (s)", title=f"{tier} model, 2025 test laps")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/pred_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_tyre_curves(test, preds, fuel_total_s):
    fuel = fuel_total_s * test["LapNumber"] / test["TotalLaps"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, compound in zip(axes, config.COMPOUNDS):
        mask = ((test["Compound"] == compound) & (test["TyreLife"] <= 30)).values
        d = pd.DataFrame({"age": test.loc[mask, "TyreLife"].values, "actual": test.loc[mask, "delta"].values + fuel[mask].values, "pre_lap": preds["gb_pre_lap"][mask] + fuel[mask].values, "live": preds["gb_live"][mask] + fuel[mask].values})
        g = d.groupby("age").mean()
        g = g - g.loc[2:4].mean()
        for col, style in [("actual", "-"), ("pre_lap", "--"), ("live", ":")]:
            ax.plot(g.index, g[col], style, lw=2, label=col)
        ax.set(title=compound, xlabel="Tyre age (laps)")
    axes[0].set_ylabel("Fuel-corrected lap time loss vs age 2-4 (s)")
    axes[0].legend()
    fig.suptitle("Tyre-age curves on unseen 2025 races")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/degradation_curves.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    laps = features.build_lap_table(features.load_laps())
    train, test = features.split_years(laps)
    print(f"clean laps {len(laps)}, races {laps['race_id'].nunique()}, train {len(train)}, test {len(test)}", flush=True)
    preds, fitted, fuel_total_s = predict_all(train, test)
    report = {"fuel_effect_total_s_fitted": round(fuel_total_s, 3), "holdout_2025": {name: score(test, preds[name], fuel_total_s) for name in MODEL_ORDER}}
    report["holdout_2025_stint_2_plus"] = later_stints(test, preds)
    report["mixed_year_cv"] = mixed_year_cv(laps)
    report["leave_one_year_out"] = leave_one_year_out(laps)
    report["sanity_checks"] = {"label_shuffle": label_shuffle(train, test), "train_vs_test": train_vs_test(train, test, preds, fitted), "ablation_pre_lap_test_mae_s": ablation(train, test, preds), **race_bootstrap(test, preds)}
    report["prediction_intervals_90"] = {tier: models.conformal_interval(train, test, config.TIERS[tier], tier) for tier in config.TIERS}
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    plot_predictions(test, preds)
    plot_tyre_curves(test, preds, fuel_total_s)
    print("done", flush=True)


if __name__ == "__main__":
    main()
