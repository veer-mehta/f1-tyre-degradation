"""Evaluation metrics, including tyre-age curve and stint-slope scores."""
import numpy as np
import pandas as pd
from f1tyre import numpy_ml as ml
from f1tyre.config import COMPOUNDS


def curve_error(test, pred, max_age=30, min_n=30):
    d = test[["Compound", "TyreLife", "delta"]].assign(pred=pred)
    errors = []
    for compound in COMPOUNDS:
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
    out = {"mae_s": ml.mae(y, pred), "median_ae_s": ml.median_ae(y, pred), "rmse_s": ml.rmse(y, pred), "r2": ml.r2(y, pred), "curve_mae_s": curve_error(test, pred)}
    if fuel_total_s is not None:
        s = stint_slopes(test, pred, fuel_total_s)
        out["slope_mae_s_per_lap"] = float(np.abs(s[:, 0] - s[:, 1]).mean())
        out["slope_corr"] = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) if np.std(s[:, 1]) > 0 else 0.0
    return {k: round(float(v), 4) for k, v in out.items()}
