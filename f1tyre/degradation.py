"""Tyre degradation model: fuel-corrected lap-time change with tyre age, estimated within stints.

Every stint is compared only with itself (stint fixed effects), so driver pace, car, strategy and
compound pace offsets cancel out. Compound, track temperature and circuit characteristics enter
as interactions with tyre age, which makes the coefficients directly interpretable.
"""
import numpy as np
import pandas as pd
from f1tyre.config import COMPOUNDS


STINT_KEY = ["race_id", "Driver", "Stint"]
MIN_STINT_LAPS = 8
CURVATURE_SCALE = 100.0
HINGE_SCALE = 10.0


def build_panel(laps, fuel_total_s, min_laps=MIN_STINT_LAPS):
    panel = laps[STINT_KEY + ["Year", "EventName", "Compound", "TyreLife", "delta", "LapNumber", "TotalLaps", "TrackTemp"]].copy()
    panel["y"] = panel["delta"] + fuel_total_s * panel["LapNumber"] / panel["TotalLaps"]
    panel["age"] = panel["TyreLife"].astype(float)
    panel = panel[panel.groupby(STINT_KEY)["y"].transform("size") >= min_laps]
    panel["stint_temp"] = panel.groupby(STINT_KEY)["TrackTemp"].transform("mean")
    return panel.reset_index(drop=True)


def within_stint(values, panel):
    return values - values.groupby([panel[k] for k in STINT_KEY]).transform("mean")


class DegradationModel:
    def __init__(self, circuit_columns=(), temperature=True, curvature=True, alpha=1000.0, knots=(), shape_alpha=10.0):
        self.circuit_columns = list(circuit_columns)
        self.temperature = temperature
        self.curvature = curvature
        self.alpha = alpha
        self.knots = list(knots)
        self.shape_alpha = shape_alpha

    def _terms(self, age, compound, temp, circuit):
        terms = {}
        for k in COMPOUNDS:
            is_k = (compound == k).astype(float)
            terms[f"age[{k}]"] = age * is_k
            if self.curvature:
                terms[f"age2[{k}]"] = age ** 2 / CURVATURE_SCALE * is_k
            for knot in self.knots:
                terms[f"hinge{knot}[{k}]"] = np.maximum(age - knot, 0.0) / HINGE_SCALE * is_k
        if self.temperature:
            terms["age x track_temp"] = age * temp
        for column in self.circuit_columns:
            terms[f"age x {column}"] = age * circuit[column].to_numpy()
        return pd.DataFrame(terms)

    def _raw_terms(self, panel, circuits):
        temp = ((panel["stint_temp"] - self.temp_mean_) / self.temp_std_).to_numpy()
        circuit = (circuits.loc[panel["EventName"], self.circuit_columns] - self.circuit_mean_) / self.circuit_std_ if self.circuit_columns else pd.DataFrame(index=range(len(panel)))
        return self._terms(panel["age"].to_numpy(), panel["Compound"].to_numpy(), temp, circuit.reset_index(drop=True))

    def _design(self, panel, circuits):
        terms = self._raw_terms(panel, circuits)
        return terms.apply(lambda column: within_stint(column, panel)).to_numpy(), list(terms.columns)

    def predict_level(self, panel, circuits=None):
        return self._raw_terms(panel, circuits).to_numpy() @ self.coef_

    def fit(self, panel, circuits=None):
        self.temp_mean_ = panel["stint_temp"].mean()
        self.temp_std_ = panel["stint_temp"].std() or 1.0
        if self.circuit_columns:
            self.circuit_mean_ = circuits[self.circuit_columns].mean()
            self.circuit_std_ = circuits[self.circuit_columns].std().replace(0, 1.0)
        x, names = self._design(panel, circuits)
        y = within_stint(panel["y"], panel).to_numpy()
        penalty = np.array([0.0 if name.startswith(("age[", "age2[")) else self.shape_alpha if name.startswith("hinge") else self.alpha for name in names])
        self.coef_ = np.linalg.lstsq(x.T @ x + np.diag(penalty), x.T @ y, rcond=None)[0]
        self.names_ = names
        return self

    def predict_within(self, panel, circuits=None):
        x, _ = self._design(panel, circuits)
        return x @ self.coef_

    def coefficients(self):
        return pd.Series(self.coef_, index=self.names_)

    def curve(self, compound, circuit_row, track_temp, ages, coef=None):
        ages = np.asarray(ages, dtype=float)
        n = len(ages)
        circuit = pd.DataFrame({c: np.full(n, (circuit_row[c] - self.circuit_mean_[c]) / self.circuit_std_[c]) for c in self.circuit_columns})
        temp = np.full(n, (track_temp - self.temp_mean_) / self.temp_std_)
        terms = self._terms(ages, np.full(n, compound), temp, circuit)
        values = terms.to_numpy() @ (self.coef_ if coef is None else coef)
        return values - values[0]

    def curve_band(self, compound, circuit_row, track_temp, ages, draws, level=0.95):
        curves = np.array([self.curve(compound, circuit_row, track_temp, ages, coef=row) for row in draws[self.names_].to_numpy()])
        return np.quantile(curves, (1 - level) / 2, axis=0), np.quantile(curves, 1 - (1 - level) / 2, axis=0)


def curve_table(model, panel, circuits=None, reference_age=3, min_n=20, max_age=30):
    frame = panel[STINT_KEY + ["Compound", "age", "y"]].assign(level=model.predict_level(panel, circuits))
    reference = frame[frame["age"] == reference_age].set_index(STINT_KEY)[["y", "level"]].rename(columns={"y": "y_ref", "level": "level_ref"})
    joined = frame.join(reference, on=STINT_KEY).dropna(subset=["y_ref"])
    joined = joined[(joined["age"] > reference_age) & (joined["age"] <= max_age)]
    joined["actual"] = joined["y"] - joined["y_ref"]
    joined["predicted"] = joined["level"] - joined["level_ref"]
    by_age = joined.groupby(["Compound", "age"]).agg(actual=("actual", "mean"), predicted=("predicted", "mean"), n=("actual", "size"))
    return by_age[by_age["n"] >= min_n].reset_index()


def curve_fidelity(model, panel, circuits=None, reference_age=3, min_n=20, max_age=30):
    table = curve_table(model, panel, circuits, reference_age, min_n, max_age)
    return float((table["actual"] - table["predicted"]).abs().mean())


def stint_slopes(values, panel):
    age_c = within_stint(panel["age"], panel)
    frame = pd.DataFrame({"xy": age_c * within_stint(values, panel), "xx": age_c ** 2})
    sums = frame.groupby([panel[k] for k in STINT_KEY]).sum()
    return sums["xy"] / sums["xx"]


def evaluate(panel, predicted_within, min_laps=15):
    y_c = within_stint(panel["y"], panel)
    pred_c = pd.Series(predicted_within, index=panel.index)
    long_enough = panel.groupby(STINT_KEY)["y"].transform("size") >= min_laps
    true_slope = stint_slopes(panel["y"], panel[long_enough])
    pred_slope = stint_slopes(pred_c, panel[long_enough])
    compound = panel[long_enough].groupby(STINT_KEY)["Compound"].first()
    by_compound = pd.DataFrame({"true": true_slope, "pred": pred_slope, "compound": compound}).groupby("compound")[["true", "pred"]].mean()
    return {"within_r2": float(1 - ((y_c - pred_c) ** 2).sum() / (y_c ** 2).sum()), "within_mae_s": float((y_c - pred_c).abs().mean()), "slope_corr": float(np.corrcoef(true_slope, pred_slope)[0, 1]), "slope_mae_s_per_lap": float((true_slope - pred_slope).abs().mean()), "mean_slope_by_compound": by_compound.round(4)}


def race_folds(panel, n_folds=5):
    races = sorted(panel["race_id"].unique())
    fold_of_race = {race: i % n_folds for i, race in enumerate(races)}
    return panel["race_id"].map(fold_of_race).to_numpy()


def select_alpha(panel, circuits, circuit_columns, alphas, n_folds=5):
    folds = race_folds(panel, n_folds)
    scores = {}
    for alpha in alphas:
        errors = []
        for fold in range(n_folds):
            train = panel[folds != fold].reset_index(drop=True)
            valid = panel[folds == fold].reset_index(drop=True)
            model = DegradationModel(circuit_columns, alpha=alpha).fit(train, circuits)
            errors.append(float(np.abs(within_stint(valid["y"], valid) - model.predict_within(valid, circuits)).mean()))
        scores[alpha] = float(np.mean(errors))
    return min(scores, key=scores.get), scores


def circuit_severity(panel):
    age_c = within_stint(panel["age"], panel)
    frame = pd.DataFrame({"EventName": panel["EventName"], "Compound": panel["Compound"], "xy": age_c * within_stint(panel["y"], panel), "xx": age_c ** 2})
    by_compound = frame.groupby("Compound")[["xy", "xx"]].sum()
    beta = by_compound["xy"] / by_compound["xx"]
    by_event_compound = frame.groupby(["EventName", "Compound"])[["xy", "xx"]].sum()
    by_event = by_event_compound.groupby(level=0).sum()
    expected_numerator = (by_event_compound["xx"] * beta.reindex(by_event_compound.index.get_level_values(1)).to_numpy()).groupby(level=0).sum()
    return (by_event["xy"] / by_event["xx"]) / (expected_numerator / by_event["xx"])


def rank_descriptors(panel, circuits, columns):
    severity = circuit_severity(panel)
    table = circuits.loc[severity.index, columns]
    rho = {c: table[c].rank().corr(severity.rank()) for c in columns}
    return sorted(columns, key=lambda c: -abs(rho[c]))


def rank_permutation_p(table, severity, n_permutations=5000, seed=42):
    """p-value for the strongest rank correlation among all columns, allowing for having picked the best of them."""
    ranks = table.rank().to_numpy()
    target = severity.rank().to_numpy()
    rng = np.random.RandomState(seed)

    def strongest(values):
        return max(abs(np.corrcoef(ranks[:, j], values)[0, 1]) for j in range(ranks.shape[1]))

    observed = strongest(target)
    null = np.array([strongest(rng.permutation(target)) for _ in range(n_permutations)])
    return {"strongest_abs_spearman": float(observed), "p_value_best_of_columns": float(np.mean(null >= observed))}


def select_shape(panel, candidates, alpha=100.0, n_folds=5):
    folds = race_folds(panel, n_folds)
    within, fidelity = {}, {}
    for name, kwargs in candidates.items():
        errors, curve_errors = [], []
        for fold in range(n_folds):
            train = panel[folds != fold].reset_index(drop=True)
            valid = panel[folds == fold].reset_index(drop=True)
            model = DegradationModel(alpha=alpha, **kwargs).fit(train)
            errors.append(float(np.abs(within_stint(valid["y"], valid) - model.predict_within(valid)).mean()))
            curve_errors.append(curve_fidelity(model, valid, min_n=10))
        within[name], fidelity[name] = float(np.mean(errors)), float(np.mean(curve_errors))
    return min(fidelity, key=fidelity.get), {"within_stint_mae": within, "curve_error": fidelity}


def select_descriptors(panel, circuits, columns, ks, alpha=100.0, n_folds=5, **model_kwargs):
    folds = race_folds(panel, n_folds)
    errors = {k: [] for k in ks}
    for fold in range(n_folds):
        train = panel[folds != fold].reset_index(drop=True)
        valid = panel[folds == fold].reset_index(drop=True)
        ranked = rank_descriptors(train, circuits, columns)
        for k in ks:
            model = DegradationModel(ranked[:k], alpha=alpha, **model_kwargs).fit(train, circuits)
            errors[k].append(float(np.abs(within_stint(valid["y"], valid) - model.predict_within(valid, circuits)).mean()))
    means = {k: float(np.mean(v)) for k, v in errors.items()}
    return min(means, key=means.get), means


def model_from_report(report):
    spec = report["final_model"]
    return DegradationModel(spec["circuit_columns"], alpha=spec["alpha"], **spec["shape"])
