import json
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import pipeline as p


warnings.filterwarnings("ignore")
RESULTS_DIR = str(Path(__file__).resolve().parent / "results")
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 42
MODEL_ORDER = ["median_baseline", "persistence_last3_laps", "formula_a_calculator_style", "formula_b_with_fuel_and_form", "xgboost_pre_lap", "xgboost_live"]


def assign_folds(laps):
    rng = np.random.RandomState(SEED)
    fold_of_race = {}
    for _, g in laps.groupby("Year"):
        races = sorted(g["race_id"].unique())
        rng.shuffle(races)
        fold_of_race.update({race: i % N_FOLDS for i, race in enumerate(races)})
    return laps["race_id"].map(fold_of_race)


def predict_all(train, test):
    fuel_total_s = p.estimate_fuel_effect(train)
    median = train["delta"].median()
    preds = {"median_baseline": np.full(len(test), median), "persistence_last3_laps": test["prev3_delta"].fillna(median).values}
    preds.update(p.fit_formulas(train, test, fuel_total_s))
    models = {}
    for tier in p.TIERS:
        models[tier] = p.fit_xgb(train, p.TIERS[tier], tier)
        preds[f"xgboost_{tier}"] = models[tier].predict(test[p.TIERS[tier]])
    return preds, models, fuel_total_s


def holdout(train, test):
    preds, models, fuel_total_s = predict_all(train, test)
    scores = {name: p.score(test, preds[name], fuel_total_s) for name in MODEL_ORDER}
    return scores, preds, models, fuel_total_s


def mixed_year_cv(laps):
    folds = assign_folds(laps)
    oof = {name: np.zeros(len(laps)) for name in MODEL_ORDER}
    fold_mae = {name: [] for name in MODEL_ORDER}
    for fold in range(N_FOLDS):
        is_test = (folds == fold).values
        preds, _, _ = predict_all(laps[~is_test].reset_index(drop=True), laps[is_test].reset_index(drop=True))
        for name in MODEL_ORDER:
            oof[name][is_test] = preds[name]
            fold_mae[name].append(mean_absolute_error(laps.loc[is_test, "delta"], preds[name]))
        print("mixed-year fold", fold, flush=True)
    out = {}
    for name in MODEL_ORDER:
        out[name] = p.score(laps, oof[name])
        out[name]["fold_mae_std"] = round(float(np.std(fold_mae[name], ddof=1)), 4)
        out[name]["per_year_mae"] = {int(y): round(float(mean_absolute_error(laps.loc[laps["Year"] == y, "delta"], oof[name][(laps["Year"] == y).values])), 4) for y in sorted(laps["Year"].unique())}
    return out


def leave_one_year_out(laps):
    out = {}
    for year in sorted(laps["Year"].unique()):
        is_test = (laps["Year"] == year).values
        train, test = laps[~is_test].reset_index(drop=True), laps[is_test].reset_index(drop=True)
        out[int(year)] = {}
        for tier in p.TIERS:
            pred = p.fit_xgb(train, p.TIERS[tier], tier).predict(test[p.TIERS[tier]])
            out[int(year)][f"xgboost_{tier}"] = {"mae_s": round(float(mean_absolute_error(test["delta"], pred)), 4), "r2": round(float(r2_score(test["delta"], pred)), 4)}
        print("leave-one-year-out", year, flush=True)
    return out


def sanity_checks(train, test, preds, models):
    cols = p.TIERS["pre_lap"]
    shuffled = train.copy()
    shuffled["delta"] = np.random.RandomState(SEED).permutation(train["delta"].values)
    shuffle_pred = p.fit_xgb(shuffled, cols, "pre_lap").predict(test[cols])
    out = {"label_shuffle": {"mae_s": round(float(mean_absolute_error(test["delta"], shuffle_pred)), 4), "r2": round(float(r2_score(test["delta"], shuffle_pred)), 4)}}
    out["train_vs_test"] = {}
    for tier in p.TIERS:
        train_pred = models[tier].predict(train[p.TIERS[tier]])
        out["train_vs_test"][tier] = {"train_mae_s": round(float(mean_absolute_error(train["delta"], train_pred)), 4), "test_mae_s": round(float(mean_absolute_error(test["delta"], preds[f"xgboost_{tier}"])), 4)}
    out["ablation_pre_lap_test_mae_s"] = {"all_features": round(float(mean_absolute_error(test["delta"], preds["xgboost_pre_lap"])), 4)}
    for group, group_cols in p.FEATURE_GROUPS.items():
        kept = [c for c in cols if c not in group_cols]
        pred = p.fit_xgb(train, kept, "pre_lap").predict(test[kept])
        out["ablation_pre_lap_test_mae_s"][f"drop_{group}"] = round(float(mean_absolute_error(test["delta"], pred)), 4)
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
    ci = {name: [round(float(np.percentile(draws[name], 2.5)), 4), round(float(np.percentile(draws[name], 97.5)), 4)] for name in MODEL_ORDER}
    gain = {}
    for name in ["xgboost_pre_lap", "xgboost_live"]:
        diff = np.array(draws["formula_b_with_fuel_and_form"]) - np.array(draws[name])
        gain[name] = {"mean_s": round(float(diff.mean()), 4), "ci95": [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)], "share_better": round(float((diff > 0).mean()), 3)}
    return {"mae_ci95": ci, "gain_over_formula_b": gain}


def plot_results(test, preds, models, fuel_total_s):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, tier in zip(axes, p.TIERS):
        ax.scatter(test["delta"], preds[f"xgboost_{tier}"], s=2, alpha=0.15)
        ax.plot([-3, 6], [-3, 6], color="red", lw=1)
        ax.set(xlim=(-3, 6), ylim=(-3, 6), xlabel="Actual lap time minus field median (s)", ylabel="Predicted (s)", title=f"{tier} model, 2025 test laps")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/pred_vs_actual.png", dpi=150)
    plt.close(fig)
    fuel = fuel_total_s * test["LapNumber"] / test["TotalLaps"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, compound in zip(axes, p.COMPOUNDS):
        mask = ((test["Compound"] == compound) & (test["TyreLife"] <= 30)).values
        d = pd.DataFrame({"age": test.loc[mask, "TyreLife"].values, "actual": test.loc[mask, "delta"].values + fuel[mask].values, "pre_lap": preds["xgboost_pre_lap"][mask] + fuel[mask].values, "live": preds["xgboost_live"][mask] + fuel[mask].values})
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
    importance = pd.Series(models["live"].feature_importances_, index=p.TIERS["live"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    importance.plot.barh(ax=ax, title="XGBoost feature importance (live model)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)


def write_summary(report):
    lines = ["# Scores (generated by evaluate.py)", "", "## 2025 holdout (train 2022-2024)", "", "| Model | MAE (s) | Median AE (s) | R2 | Curve MAE (s) | Slope corr | Slope MAE (s/lap) |", "|---|---|---|---|---|---|---|"]
    for name in MODEL_ORDER:
        m = report["holdout_2025"][name]
        lines.append(f"| {name} | {m['mae_s']} | {m['median_ae_s']} | {m['r2']} | {m['curve_mae_s']} | {m['slope_corr']} | {m['slope_mae_s_per_lap']} |")
    lines += ["", "## Mixed-year 5-fold CV (all 90 races)", "", "| Model | MAE (s) | R2 | Fold MAE std | MAE 2022 / 2023 / 2024 / 2025 |", "|---|---|---|---|---|"]
    for name in MODEL_ORDER:
        m = report["mixed_year_cv"][name]
        years = " / ".join(str(v) for v in m["per_year_mae"].values())
        lines.append(f"| {name} | {m['mae_s']} | {m['r2']} | {m['fold_mae_std']} | {years} |")
    lines += ["", "## Leave-one-year-out MAE (R2)", "", "| Held-out year | pre-lap | live |", "|---|---|---|"]
    for year, m in report["leave_one_year_out"].items():
        lines.append(f"| {year} | {m['xgboost_pre_lap']['mae_s']} ({m['xgboost_pre_lap']['r2']}) | {m['xgboost_live']['mae_s']} ({m['xgboost_live']['r2']}) |")
    open(f"{RESULTS_DIR}/summary.md", "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    laps = p.build_lap_table(p.load_laps())
    train, test = p.split_years(laps)
    print(f"clean laps {len(laps)}, races {laps['race_id'].nunique()}, train {len(train)}, test {len(test)}", flush=True)
    scores, preds, models, fuel_total_s = holdout(train, test)
    report = {"fuel_effect_total_s_fitted": round(fuel_total_s, 3), "holdout_2025": scores}
    print(json.dumps(scores, indent=1), flush=True)
    report["mixed_year_cv"] = mixed_year_cv(laps)
    report["leave_one_year_out"] = leave_one_year_out(laps)
    report["sanity_checks"] = sanity_checks(train, test, preds, models)
    report["sanity_checks"].update(race_bootstrap(test, preds))
    report["prediction_intervals_90"] = {tier: p.conformal_interval(train, test, p.TIERS[tier], tier) for tier in p.TIERS}
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    write_summary(report)
    plot_results(test, preds, models, fuel_total_s)
    print("done", flush=True)


if __name__ == "__main__":
    main()
