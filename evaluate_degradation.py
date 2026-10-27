import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from f1tyre import config, features, models
from f1tyre.config import DESCRIPTOR_LABELS
from f1tyre import degradation as dg


RESULTS_DIR = config.RESULTS_DIR
SEED = 42
N_BOOTSTRAP = 200
SELECTION_SIZES = [0, 1, 2, 3, 5]
DESCRIPTORS = ["length_km", "avg_speed_kmh", "top_speed_kmh", "min_speed_kmh", "full_throttle_pct", "brake_pct", "heavy_braking_zones", "mean_lat_g", "high_lat_pct", "slow_corners", "medium_corners", "fast_corners", "climb_m_per_km"]
SHAPE_CANDIDATES = {
    "quadratic": dict(curvature=True, knots=[], shape_alpha=10.0),
    "piecewise-linear, knots 5-25, light smoothing": dict(curvature=False, knots=[5, 10, 15, 20, 25], shape_alpha=10.0),
    "piecewise-linear, knots 5-25, strong smoothing": dict(curvature=False, knots=[5, 10, 15, 20, 25], shape_alpha=100.0),
    "piecewise-linear, knots 4-22, light smoothing": dict(curvature=False, knots=[4, 8, 14, 22], shape_alpha=10.0),
    "piecewise-linear, knots 4-22, strong smoothing": dict(curvature=False, knots=[4, 8, 14, 22], shape_alpha=100.0),
}
COMPOUND_COLORS = {"SOFT": "#E10600", "MEDIUM": "#E0A800", "HARD": "#555555"}


def spearman(a, b):
    return float(pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()))


def summarise(result):
    out = {k: round(v, 4) for k, v in result.items() if k != "mean_slope_by_compound"}
    out["mean_slope_by_compound"] = result["mean_slope_by_compound"].to_dict()
    return out


def scores_of(model, panel, circuits=None):
    return summarise(dg.evaluate(panel, model.predict_within(panel, circuits)))


def severity_from_descriptors(severity, circuits):
    table = circuits.loc[severity.index, DESCRIPTORS]
    z = ((table - table.mean()) / table.std().replace(0, 1.0)).to_numpy()
    predictions = np.zeros(len(severity))
    for i in range(len(severity)):
        keep = np.arange(len(severity)) != i
        predictions[i] = models.RidgeRegression(alpha=10.0).fit(z[keep], severity.to_numpy()[keep]).predict(z[i : i + 1])[0]
    return {"leave_one_circuit_out_r2": round(models.r2(severity.to_numpy(), predictions), 4), "leave_one_circuit_out_spearman": round(spearman(predictions, severity), 4)}


def single_descriptor_effects(panel_all, ptrain, ptest, circuits, baseline, shape):
    effects = {}
    for column in DESCRIPTORS:
        fitted_all = dg.DegradationModel([column], alpha=1.0, **shape).fit(panel_all, circuits)
        fitted_train = dg.DegradationModel([column], alpha=1.0, **shape).fit(ptrain, circuits)
        held_out = dg.evaluate(ptest, fitted_train.predict_within(ptest, circuits))
        effects[column] = {"effect_per_sd_s_per_lap": round(float(fitted_all.coefficients()[f"age x {column}"]), 5), "heldout_slope_corr_change": round(held_out["slope_corr"] - baseline["slope_corr"], 4), "heldout_within_r2_change": round(held_out["within_r2"] - baseline["within_r2"], 4)}
    return effects


def bootstrap_coefficients(panel, circuits, columns, shape):
    rng = np.random.RandomState(SEED)
    races = panel["race_id"].unique()
    rows_of = {r: np.flatnonzero(panel["race_id"].to_numpy() == r) for r in races}
    draws = []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([rows_of[r] for r in rng.choice(races, size=len(races), replace=True)])
        draws.append(dg.DegradationModel(columns, alpha=100.0, **shape).fit(panel.iloc[idx].reset_index(drop=True), circuits).coefficients())
    return pd.DataFrame(draws)


def plot_curves(model, circuits, selected, draws, path):
    ranked = circuits[selected[0]].sort_values() if selected else circuits["length_km"].sort_values()
    picks = [ranked.index[0], ranked.index[-1]]
    ages = np.arange(1, 31)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, event in zip(axes, picks):
        for compound in config.COMPOUNDS:
            ax.plot(ages, model.curve(compound, circuits.loc[event], 35.0, ages), lw=2.5, color=COMPOUND_COLORS[compound], label=compound.title())
            low, high = model.curve_band(compound, circuits.loc[event], 35.0, ages, draws)
            ax.fill_between(ages, low, high, color=COMPOUND_COLORS[compound], alpha=0.15)
        suffix = f"\n{DESCRIPTOR_LABELS[selected[0]]}: {circuits.loc[event, selected[0]]:.1f}" if selected else ""
        ax.set(title=f"{event.replace(' Grand Prix', '')}{suffix}", xlabel="Tyre age (laps)")
    axes[0].set_ylabel("Predicted lap-time loss vs age 1 (s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curve_check(table, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, compound in zip(axes, config.COMPOUNDS):
        g = table[table["Compound"] == compound]
        ax.scatter(g["age"], g["actual"], s=np.clip(g["n"], 8, 120), color=COMPOUND_COLORS[compound], alpha=0.8, label="observed average")
        ax.plot(g["age"], g["predicted"], color="black", lw=2, label="model")
        ax.set(title=compound.title(), xlabel="Tyre age (laps)")
    axes[0].set_ylabel("Lap-time loss vs age 3 (s)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Observed vs modelled tyre-age curves, unseen 2025 stints (dot size = number of stints)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_descriptor_correlations(correlations, selected, path):
    order = sorted(correlations, key=lambda c: correlations[c])
    colors = ["#E10600" if c in selected else "#888888" for c in order]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh([DESCRIPTOR_LABELS[c] for c in order], [correlations[c] for c in order], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman correlation with measured degradation severity (25 circuits)")
    ax.set_title("Circuit characteristics vs degradation (selected in red)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    laps = features.build_lap_table(features.load_laps())
    circuits = features.load_circuits()
    train, test = features.split_years(laps)
    fuel_total_s = models.estimate_fuel_effect(train)
    ptrain, ptest = dg.build_panel(train, fuel_total_s), dg.build_panel(test, fuel_total_s)
    panel_all = dg.build_panel(laps, fuel_total_s)
    report = {"n_train_stints": int(ptrain.groupby(dg.STINT_KEY).ngroups), "n_test_stints": int(ptest.groupby(dg.STINT_KEY).ngroups), "n_circuits": int(len(circuits))}

    severity = dg.circuit_severity(panel_all)
    correlations = {c: round(spearman(circuits.loc[severity.index, c], severity), 3) for c in DESCRIPTORS}
    yearly = {year: dg.circuit_severity(panel_all[panel_all["Year"] == year]) for year in sorted(panel_all["Year"].unique())}
    years = sorted(yearly)
    report["severity_year_to_year_correlation"] = {f"{a}-{b}": round(float(pd.concat([yearly[a], yearly[b]], axis=1, join="inner").dropna().corr().iloc[0, 1]), 3) for a, b in zip(years[:-1], years[1:])}
    report["circuit_severity_x_expected"] = severity.round(3).to_dict()
    report["descriptor_spearman_with_severity"] = correlations
    report["severity_predicted_from_all_descriptors"] = severity_from_descriptors(severity, circuits)
    print(report["severity_predicted_from_all_descriptors"], flush=True)
    report["descriptor_selection_p_value"] = dg.rank_permutation_p(circuits.loc[severity.index, DESCRIPTORS], severity)
    table = circuits.loc[severity.index, "climb_m_per_km"]
    leave_one_out = [spearman(table.drop(c), severity.drop(c)) for c in severity.index]
    report["descriptor_selection_p_value"]["elevation_rho_leave_one_circuit_out_range"] = [round(min(leave_one_out), 3), round(max(leave_one_out), 3)]
    print(report["descriptor_selection_p_value"], flush=True)

    shape_name, shape_scores = dg.select_shape(ptrain, SHAPE_CANDIDATES)
    shape = SHAPE_CANDIDATES[shape_name]
    report["shape_selection"] = {"cv_within_mae_by_shape": {k: round(v, 5) for k, v in shape_scores["within_stint_mae"].items()}, "cv_curve_error_by_shape": {k: round(v, 5) for k, v in shape_scores["curve_error"].items()}, "chosen": shape_name, "settings": shape}
    print(report["shape_selection"], flush=True)

    baseline_model = dg.DegradationModel(**shape).fit(ptrain)
    baseline = dg.evaluate(ptest, baseline_model.predict_within(ptest))
    report["single_descriptor_effects"] = single_descriptor_effects(panel_all, ptrain, ptest, circuits, baseline, shape)

    best_k, k_scores = dg.select_descriptors(ptrain, circuits, DESCRIPTORS, SELECTION_SIZES, **shape)
    selected = dg.rank_descriptors(ptrain, circuits, DESCRIPTORS)[:best_k]
    report["selection"] = {"cv_within_mae_by_number_of_descriptors": {str(k): round(v, 5) for k, v in k_scores.items()}, "chosen_number": best_k, "selected_descriptors": selected}
    print(report["selection"], flush=True)

    variants = {"compound, curvature, temperature": summarise(baseline)}
    all_model = dg.DegradationModel(DESCRIPTORS, alpha=10000.0, **shape).fit(ptrain, circuits)
    variants["+ all 13 circuit characteristics"] = scores_of(all_model, ptest, circuits)
    selected_model = dg.DegradationModel(selected, alpha=100.0, **shape).fit(ptrain, circuits)
    variants["+ selected circuit characteristics"] = scores_of(selected_model, ptest, circuits)
    dummies = pd.DataFrame({f"is_{e}": (circuits.index == e).astype(float) for e in circuits.index}, index=circuits.index)
    variants["+ circuit identity (each circuit's own history)"] = scores_of(dg.DegradationModel(list(dummies.columns), alpha=100.0, **shape).fit(ptrain, dummies), ptest, dummies)
    for name, cols in [("gradient boosting, pre-lap without circuit feature", [c for c in config.TIERS["pre_lap"] if c not in config.CIRCUIT_FEATURES]), ("gradient boosting, pre-lap (lap-level model)", config.TIERS["pre_lap"])]:
        gb = models.fit_gb(train, cols, "pre_lap")
        t = test.assign(gb_fc=gb.predict(test[cols]) + fuel_total_s * test["LapNumber"] / test["TotalLaps"])
        merged = ptest.merge(t[dg.STINT_KEY + ["LapNumber", "gb_fc"]], on=dg.STINT_KEY + ["LapNumber"], how="left").reset_index(drop=True)
        variants[name] = summarise(dg.evaluate(merged, dg.within_stint(merged["gb_fc"], merged).to_numpy()))
    report["holdout_2025"] = variants
    for name, result in variants.items():
        print(name, {k: v for k, v in result.items() if k != "mean_slope_by_compound"}, result["mean_slope_by_compound"], flush=True)

    check = dg.curve_table(selected_model, ptest, circuits, min_n=15)
    report["curve_check_2025"] = {"mean_abs_error_s": round(float((check["actual"] - check["predicted"]).abs().mean()), 4), "points": int(len(check)), "by_shape_2025": {name: round(dg.curve_fidelity(dg.DegradationModel(alpha=100.0, **kw).fit(ptrain), ptest), 4) for name, kw in SHAPE_CANDIDATES.items()}}
    plot_curve_check(check, RESULTS_DIR / "curve_check.png")
    final = dg.DegradationModel(selected, alpha=100.0, **shape).fit(panel_all, circuits)
    draws = bootstrap_coefficients(panel_all, circuits, selected, shape)
    draws.to_csv(RESULTS_DIR / "degradation_bootstrap.csv", index=False)
    low, high = draws.quantile(0.025), draws.quantile(0.975)
    coefficients = final.coefficients()
    report["final_model"] = {"circuit_columns": selected, "alpha": 100.0, "shape": shape}
    report["final_model_coefficients"] = {n: {"value": round(float(coefficients[n]), 5), "ci95": [round(float(low[n]), 5), round(float(high[n]), 5)]} for n in coefficients.index}
    predicted = pd.Series(final.predict_within(panel_all, circuits), index=panel_all.index)
    keyed = panel_all.groupby(dg.STINT_KEY)
    slopes = pd.DataFrame({"actual": dg.stint_slopes(panel_all["y"], panel_all), "model": dg.stint_slopes(predicted, panel_all), "Compound": keyed["Compound"].first(), "laps": keyed["y"].size()})

    def slope_means(group):
        return {"stints": int(len(group)), "actual": round(float(group["actual"].mean()), 4), "model": round(float(group["model"].mean()), 4)}

    report["stint_fit_in_sample"] = {"stints": int(len(slopes)), "model_below_average_stint_pct": round(100 * float(1 - slopes["model"].mean() / slopes["actual"].mean()), 1), "share_of_stints_steeper_than_model": round(float((slopes["actual"] > slopes["model"]).mean()), 3), "slope_corr": round(float(slopes["actual"].corr(slopes["model"])), 3), "by_compound": {c: slope_means(slopes[slopes["Compound"] == c]) for c in config.COMPOUNDS}, "stints_18_laps_or_fewer": slope_means(slopes[slopes["laps"] <= 18]), "stints_over_25_laps": slope_means(slopes[slopes["laps"] > 25])}
    print("stint fit", report["stint_fit_in_sample"], flush=True)
    plot_curves(final, circuits, selected, draws, RESULTS_DIR / "degradation_by_circuit.png")
    plot_descriptor_correlations(correlations, selected, RESULTS_DIR / "circuit_characteristics.png")
    with open(RESULTS_DIR / "degradation.json", "w") as f:
        json.dump(report, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
