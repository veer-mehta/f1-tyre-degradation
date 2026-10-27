import base64
import json
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"median_baseline": "Median baseline", "formula_a_calculator_style": "Formula A (calculator style)", "formula_b_with_fuel_and_form": "Formula B (fuel and form)", "ridge_pre_lap_features": "Ridge regression, pre-lap features", "gb_pre_lap": "Gradient boosting, pre-lap", "gb_pre_stint": "Gradient boosting, pre-stint", "persistence_last3_laps": "Persistence (last 3 laps)", "gb_live": "Gradient boosting, live"}
ORDER = ["median_baseline", "formula_a_calculator_style", "formula_b_with_fuel_and_form", "ridge_pre_lap_features", "gb_pre_lap", "gb_pre_stint", "persistence_last3_laps", "gb_live"]
CSS = """
@page { size: A4; margin: 18mm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
.bar { width: 48px; height: 4px; background: #e10600; margin-bottom: 10px; }
h1 { font-size: 21pt; letter-spacing: 0.02em; margin: 0 0 4px 0; }
.sub { color: #666; margin-bottom: 22px; }
h2 { font-size: 11pt; text-transform: uppercase; letter-spacing: 0.12em; margin: 24px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid #ddd; }
p { margin: 6px 0; }
ul { margin: 6px 0; padding-left: 18px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 4px 0; font-size: 9.5pt; }
th { text-align: left; font-weight: 600; color: #555; border-bottom: 1.5px solid #1a1a1a; padding: 5px 6px; }
td { border-bottom: 1px solid #e6e6e6; padding: 5px 6px; }
td.n, th.n { text-align: right; }
tr.key td { font-weight: 600; }
table.meta { margin: 0 0 18px 0; font-size: 10pt; }
table.meta td { border: none; padding: 1px 14px 1px 0; }
table.meta td.k { color: #777; text-transform: uppercase; letter-spacing: 0.08em; font-size: 8.5pt; width: 70px; }
.cap { color: #666; font-size: 9pt; margin: 2px 0 10px 0; }
img { width: 100%; margin: 8px 0 2px 0; }
.fig, table { break-inside: avoid; }
"""


def details_html():
    with open(os.path.join(ROOT, "report", "details.json")) as f:
        d = json.load(f)
    rows = [("Name", d["name"], "Roll No.", d["roll_no"]), ("College", d["college"], "Semester", d["semester"]), ("Subject", d["subject"], "Professor", d["professor"]), ("Date", d["date"], "", "")]
    body = "".join(f'<tr><td class="k">{a}</td><td>{b}</td><td class="k">{c}</td><td>{e}</td></tr>' for a, b, c, e in rows)
    return f'<table class="meta">{body}</table>'


def image_tag(name, width=100):
    with open(os.path.join(ROOT, "report" if name.startswith("app_") else "results", name), "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img style="width:{width}%" src="data:image/png;base64,{data}">'


def table(headers, rows, key_rows=()):
    html = "<table><tr>" + "".join(f'<th class="{"n" if i else ""}">{h}</th>' for i, h in enumerate(headers)) + "</tr>"
    for row_id, cells in rows:
        cls = ' class="key"' if row_id in key_rows else ""
        html += f"<tr{cls}>" + "".join(f'<td class="{"n" if i else ""}">{c}</td>' for i, c in enumerate(cells)) + "</tr>"
    return html + "</table>"


DESCRIPTOR_TEXT = {"length_km": "lap length", "avg_speed_kmh": "average speed", "top_speed_kmh": "top speed", "min_speed_kmh": "slowest-corner speed", "full_throttle_pct": "full-throttle share", "brake_pct": "braking share", "heavy_braking_zones": "heavy braking zones", "mean_lat_g": "mean cornering load", "high_lat_pct": "share of the lap above 2.5 g", "slow_corners": "slow corners", "medium_corners": "medium corners", "fast_corners": "fast corners", "climb_m_per_km": "elevation change per kilometre"}
VARIANT_NAMES = [("compound, curvature, temperature", "Compound, tyre age and track temperature"), ("+ selected circuit characteristics", "+ elevation change (selected)"), ("+ all 13 circuit characteristics", "+ all 13 circuit characteristics"), ("+ circuit identity (each circuit's own history)", "+ each circuit's own history"), ("gradient boosting, pre-lap without circuit feature", "Gradient boosting, pre-lap, no circuit feature"), ("gradient boosting, pre-lap (lap-level model)", "Gradient boosting, pre-lap, with elevation change")]


def degradation_facts(deg):
    v = deg["holdout_2025"]
    selected = deg["selection"]["selected_descriptors"]
    slopes = v["+ selected circuit characteristics"]["mean_slope_by_compound"]
    base = v["compound, curvature, temperature"]
    chosen = v["+ selected circuit characteristics"]
    every = v["+ all 13 circuit characteristics"]
    history = v["+ circuit identity (each circuit's own history)"]
    year_pairs = list(deg["severity_year_to_year_correlation"].values())
    variant_rows = [(k, [label, f"{v[k]['within_r2']:.2f}", f"{v[k]['slope_corr']:.2f}", f"{v[k]['slope_mae_s_per_lap']:.4f}"]) for k, label in VARIANT_NAMES if k in v]
    compound_rows = [(c, [c.title(), f"{slopes['true'][c]:.3f}", f"{slopes['pred'][c]:.3f}"]) for c in ["SOFT", "MEDIUM", "HARD"]]
    return {"selected_text": " and ".join(DESCRIPTOR_TEXT[c] for c in selected), "rho_selected": deg["descriptor_spearman_with_severity"][selected[0]], "slopes": slopes, "ratio": slopes["true"]["SOFT"] / slopes["true"]["HARD"], "base": base, "chosen": chosen, "every": every, "history": history, "year_pairs": year_pairs, "loo": deg["severity_predicted_from_all_descriptors"], "curve_quad": deg["curve_check_2025"]["by_shape_2025"]["quadratic"], "curve_alt_low": min(x for k, x in deg["curve_check_2025"]["by_shape_2025"].items() if k != "quadratic"), "curve_alt_high": max(x for k, x in deg["curve_check_2025"]["by_shape_2025"].items() if k != "quadratic"), "gb_without": v["gradient boosting, pre-lap without circuit feature"], "gb_with": v["gradient boosting, pre-lap (lap-level model)"], "variant_rows": variant_rows, "compound_rows": compound_rows, "selp": deg["descriptor_selection_p_value"], "cliff": deg["cliff"]["summary"], "fit": deg["stint_fit_in_sample"]}


def build_html(report, deg):
    hold = report["holdout_2025"]
    cv = report["mixed_year_cv"]
    loyo = report["leave_one_year_out"]
    checks = report["sanity_checks"]
    ablation = checks["ablation_pre_lap_test_mae_s"]
    ci = checks["mae_ci95"]
    gain = checks["gain_over_formula_b"]
    gap = checks["train_vs_test"]
    intervals = report["prediction_intervals_90"]
    r2 = lambda v: f"{v if v > 0.005 else 0.0:.2f}"
    key = ("gb_pre_lap", "gb_pre_stint", "gb_live")
    hold_rows = [(k, [NAMES[k], f"{hold[k]['mae_s']:.3f}", f"{hold[k]['median_ae_s']:.3f}", r2(hold[k]["r2"]), f"{hold[k]['curve_mae_s']:.3f}", f"{hold[k]['slope_corr']:.2f}"]) for k in ORDER]
    cv_rows = [(k, [NAMES[k], f"{cv[k]['mae_s']:.3f}", r2(cv[k]["r2"]), f"{cv[k]['fold_mae_std']:.3f}"]) for k in ORDER]
    loyo_rows = [(y, [y, f"{m['gb_pre_lap']['mae_s']:.3f} ({m['gb_pre_lap']['r2']:.2f})", f"{m['gb_pre_stint']['mae_s']:.3f} ({m['gb_pre_stint']['r2']:.2f})", f"{m['gb_live']['mae_s']:.3f} ({m['gb_live']['r2']:.2f})"]) for y, m in loyo.items()]
    later = report["holdout_2025_stint_2_plus"]
    later_rows = [(k, [NAMES[k], f"{m['mae_s']:.3f}", f"{m['median_ae_s']:.3f}", f"{m['r2']:.2f}"]) for k, m in later["scores"].items()]
    ablation_rows = [(k, [k.replace("drop_", "Without ").replace("_", " ").replace("all features", "All features"), f"{v:.3f}"]) for k, v in ablation.items() if k != "drop_earlier_stints"]
    persist_gain = 100 * (1 - hold["gb_live"]["mae_s"] / hold["persistence_last3_laps"]["mae_s"])
    d = degradation_facts(deg)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>F1 Tyre Degradation Report</title><style>{CSS}</style></head><body>
<div class="bar"></div>
<h1>Predicting F1 Lap Time and Tyre Degradation</h1>
<div class="sub">Machine learning on FastF1 race data, 2022 to 2025. All models implemented from scratch with NumPy and pandas.</div>
{details_html()}

<h2>1. Problem</h2>
<p>As a tyre wears, a car gets slower, so teams need to know how much time a driver loses per lap of tyre age to decide when to pit. We predict how many seconds slower or faster a lap will be than the race's field-median lap, from the tyre, the fuel load, the weather, the car's recent form and the traffic. Degradation appears as the effect of tyre age once fuel burn is removed.</p>

<h2>2. Data</h2>
<p>Lap-by-lap timing, tyre and weather data from the FastF1 API for 92 races from 2022 to 2025. We keep dry, green-flag laps only: lap 1, pit laps, safety car, virtual safety car and yellow-flag laps, rain, deleted laps and laps slower than 104% of the driver's best are removed. That leaves 70,830 laps from 90 races. Fuel burn (about {report['fuel_effect_total_s_fitted']:.2f} s over a full race) is estimated from the data.</p>

<h2>3. Method</h2>
<p><b>Target.</b> Lap time minus the race's field-median lap time, in seconds. This removes circuit length and year-to-year pace differences.</p>
<p><b>Features.</b> Tyre compound and age (with non-linear terms and an age-by-compound interaction), lap number and race fraction as a fuel proxy, stint, track and air temperature, humidity, wind, driver and team form (their average pace over earlier races only), traffic (gaps to the cars ahead and behind and position at the end of the previous lap), and field context (the field's pace trend and laps since the last safety car or yellow flag).</p>
<p><b>Model.</b> Gradient boosting with regression trees, written from scratch (Friedman's LAD-TreeBoost generalised to a pinball, or quantile, loss). Features are binned into histograms; each round fits a tree to the negative gradient of the loss, with row and column subsampling, and each leaf is set to the quantile of the residuals it contains. At the median this is the absolute-error loss, which is robust to traffic and mistakes. Quantile versions at 5% and 95%, calibrated with split conformal prediction, give 90% prediction intervals. Ridge regression and least squares are also implemented in NumPy. Against XGBoost on the same split the implementation differs by under 0.002 s in MAE.</p>
<p><b>Three tiers.</b> The pre-lap model uses everything known at the start of the lap except the driver's own previous lap times. The pre-stint model adds how the driver went in the earlier stints of the same race (their mean pace, the pace at the end of the previous stint and its length). The live model also adds the driver's previous clean laps in the same stint, like a pit-wall forecast.</p>
<p><b>Baselines.</b> The median lap; persistence (the driver's last 3 laps averaged); Formula A, the structure of an online degradation calculator (base plus compound, track temperature, air temperature and abrasiveness factors times laps) with constants fitted to our data; Formula B, Formula A plus fuel and form; and ridge regression on the same features as the pre-lap model.</p>
<p><b>Validation.</b> Training on 2022 to 2024 (52,092 laps, 66 races) and testing on all of 2025 (18,738 laps, 24 unseen races). Mixed-year 5-fold cross-validation by race, where each fold holds out about 20% of every year's races, and leave-one-year-out. Laps from one race are never split between training and test.</p>

<h2>4. Results</h2>
<p>Unseen 2025 races. MAE is the typical error in seconds; curve error compares predicted and actual average lap-time loss by tyre age.</p>
{table(["Model", "MAE (s)", "Median AE (s)", "R&sup2;", "Curve error (s)", "Slope corr."], hold_rows, key)}
<div class="cap">Persistence is the fair baseline for the live model. The live model improves on it by {persist_gain:.0f}% in MAE.</div>
<p>Mixed-year cross-validation across all 90 races, the main performance estimate:</p>
{table(["Model", "MAE (s)", "R&sup2;", "Fold MAE std"], cv_rows, key)}
<p>Laps in stint 2 or later ({100 * later["share_of_test_laps"]:.0f}% of 2025 laps), where the pre-stint model can use the driver's earlier stints:</p>
{table(["Model", "MAE (s)", "Median AE (s)", "R&sup2;"], later_rows, key)}
<div class="fig">{image_tag("degradation_curves.png")}<div class="cap">Figure 1. Fuel-corrected lap-time loss by tyre age on unseen 2025 races: actual and predicted.</div></div>
<div class="fig">{image_tag("pred_vs_actual.png", 85)}<div class="cap">Figure 2. Predicted against actual lap time relative to the field median, 2025 test laps.</div></div>

<h2>5. Tyre degradation and circuit characteristics</h2>
<p>The lap-level model predicts lap time, so tyre age is only one of several effects it learns, and comparing different drivers and stints confounds the compound with car, driver and strategy. Asking it what-if questions, such as a soft tyre running 30 laps in a late stint, also falls outside the data it has seen, which flattened the compound differences in an early version of our app. We therefore estimate degradation directly: every stint is compared only with itself (stint fixed effects) after removing fuel burn, so driver pace, car and strategy cancel out.</p>
<p><b>Degradation model.</b> A least-squares regression written in NumPy explains the within-stint lap-time change by tyre age for each compound (linear and quadratic terms), its interaction with track temperature, and its interaction with circuit characteristics.</p>
<p>We also tried flexible piecewise-linear curve shapes, which can show a flat start or a late cliff. Judged by how well the curve reproduces the observed average tyre-age curves on held-out races, the parabola was best (curve error {d['curve_quad']:.3f} s on unseen 2025 stints against {d['curve_alt_low']:.3f} to {d['curve_alt_high']:.3f} s for the flexible shapes), so it is kept; the data do not support a cliff or a flat start beyond what it captures. Figure 4 compares the observed averages with the model. The shaded bands in Figure 3 are 95% intervals from resampling whole races.</p>
<p><b>Circuit characteristics.</b> For each of the 25 circuits we measured 13 characteristics from the telemetry of the fastest race lap: lap length, average speed, top speed, slowest-corner speed, the shares of the lap at full throttle and on the brakes, the number of heavy braking zones, mean cornering load in g, the share of the lap above 2.5 g, the numbers of slow, medium and fast corners, and elevation change per kilometre. Which ones to include was decided by cross-validation inside the training seasons only.</p>
<p>Average degradation rate on unseen 2025 stints (at least 15 laps), in seconds per lap of tyre age:</p>
{table(["Compound", "Actual", "Predicted"], d["compound_rows"])}
<p>Within-stint accuracy on unseen 2025 stints:</p>
{table(["Model", "Within-stint R&sup2;", "Slope corr.", "Slope error (s/lap)"], d["variant_rows"])}
<div class="fig">{image_tag("degradation_by_circuit.png")}<div class="cap">Figure 3. Predicted lap-time loss by tyre age and compound at the circuits with the lowest and highest elevation change (shaded: 95% interval).</div></div>
<div class="fig">{image_tag("curve_check.png")}<div class="cap">Figure 4. Observed average lap-time loss by tyre age (dots, sized by the number of stints) against the model (line) on unseen 2025 stints, relative to tyre age 3.</div></div>
<div class="fig">{image_tag("circuit_characteristics.png", 68)}<div class="cap">Figure 5. Correlation of each circuit characteristic with measured degradation severity across 25 circuits (selected characteristic in red).</div></div>
<div class="fig">{image_tag("cliff.png")}<div class="cap">Figure 6. Tyre cliffs in stints of 20 or more laps: tyre age at which the loss speeds up (left) and the share of stints that speed up late against the noise reference of stints that slow down late (right).</div></div>
<ul>
<li><b>Compounds separate clearly.</b> Soft tyres lose {d['slopes']['true']['SOFT']:.3f} s per lap, medium {d['slopes']['true']['MEDIUM']:.3f} and hard {d['slopes']['true']['HARD']:.3f}, so soft degrades about {d['ratio']:.1f} times faster than hard, and the model reproduces this ({d['slopes']['pred']['SOFT']:.3f}, {d['slopes']['pred']['MEDIUM']:.3f}, {d['slopes']['pred']['HARD']:.3f}).</li>
<li><b>One circuit characteristic held up.</b> Cross-validation on the training seasons selected {d['selected_text']} (correlation {d['rho_selected']:.2f} with measured circuit severity). The link is statistically real but modest: it stays significant after allowing for picking the best of the 13 characteristics (p = {d['selp']['p_value_best_of_columns']:.2f}) and after dropping any one circuit ({d['selp']['elevation_rho_leave_one_circuit_out_range'][0]:.2f} to {d['selp']['elevation_rho_leave_one_circuit_out_range'][1]:.2f}). No mechanism is proven: elevation change may stand in for other features of those circuits, such as layout and surface. Adding it improves the held-out stint slope correlation from {d['base']['slope_corr']:.2f} to {d['chosen']['slope_corr']:.2f}. Using all 13 characteristics or each circuit's own history does not help: within-stint R&sup2; {d['every']['within_r2']:.2f} and {d['history']['within_r2']:.2f} against {d['base']['within_r2']:.2f}.</li>
<li><b>Circuit degradation is not predictable from these characteristics alone.</b> Predicting a circuit's severity from all 13 gives R&sup2; {d['loo']['leave_one_circuit_out_r2']:.2f} on circuits left out. Tyre allocation, weather and race incidents also matter, and a circuit's severity is not stable from year to year (correlation between consecutive seasons: {", ".join(f"{x:.2f}" for x in d['year_pairs'])}).</li>
<li><b>Tyre cliffs exist, but in a minority of stints.</b> Each of the {d['cliff']['ALL']['stints']} stints of 20 or more laps was fitted with a straight line and with a line that steepens at one age; a cliff needs that age to be in the second half of the stint. {100 * d['cliff']['ALL']['cliff_share']:.1f}% show a clear cliff (at least 0.1 s per lap of extra loss), against {100 * d['cliff']['ALL']['relief_share_noise_reference']:.1f}% that slow down by the same amount, which is what chance alone gives. About {100 * (d['cliff']['ALL']['cliff_share'] - d['cliff']['ALL']['relief_share_noise_reference']):.0f}% of long stints therefore have a real cliff, typically {d['cliff']['ALL']['median_extra_slope_s_per_lap']:.2f} s per lap steeper afterwards, starting at a median age of {d['cliff']['ALL']['median_onset_age']:.0f} laps (soft {d['cliff']['SOFT']['median_onset_age']:.0f}, medium {d['cliff']['MEDIUM']['median_onset_age']:.0f}, hard {d['cliff']['HARD']['median_onset_age']:.0f}). Most stints never reach one because teams pit first, and onset ages vary, so an average curve cannot show it.</li>
<li><b>The curve describes the average lap, not every stint.</b> Over all {d['fit']['stints']:,} stints the model sits {d['fit']['model_below_average_stint_pct']:.0f}% below the average stint slope (soft {d['fit']['by_compound']['SOFT']['model']:.3f} against {d['fit']['by_compound']['SOFT']['actual']:.3f} s per lap) and {100 * d['fit']['share_of_stints_steeper_than_model']:.0f}% of stints are steeper than it. Stints of 18 laps or fewer average {d['fit']['stints_18_laps_or_fewer']['actual']:.3f} against {d['fit']['stints_18_laps_or_fewer']['model']:.3f} modelled; stints over 25 laps average {d['fit']['stints_over_25_laps']['actual']:.3f} against {d['fit']['stints_over_25_laps']['model']:.3f}. The fit weighs laps equally, so long stints dominate, and long stints are the gentle ones because tyres that wear fast are changed early. Stint slopes also differ a great deal (modelled and actual slopes correlate at {d['fit']['slope_corr']:.2f}). Weighting stints equally removes the gap but fits unseen 2025 stints worse, so it was not used.</li>
<li><b>In the lap-level model it helps tyre behaviour, not lap time.</b> Adding elevation change raises the stint slope correlation of the pre-lap gradient boosting model from {d['gb_without']['slope_corr']:.2f} to {d['gb_with']['slope_corr']:.2f}, but its lap-time error is not better (MAE {ablation['drop_circuit']:.3f} s without, {ablation['all_features']:.3f} s with).</li>
</ul>

<h2 style="break-before:page">6. The app</h2>
<p>A Streamlit app with four tabs makes the models explorable. <b>Predictor</b> draws the degradation curves for the three compounds at any circuit and track temperature, with a band for the average (95%) or for single stints (middle 80%) and a second view that adds fuel burn, so lap times can fall while the tyre wears. <b>Race explorer</b> plots the real stints of any driver in any race from 2022 to 2025 against what the model expects and marks detected tyre cliffs. <b>Circuits</b> ranks the 25 circuits by predicted tyre loss. <b>Model</b> shows the scores from this report, the tyre-cliff results and the circuit characteristics.</p>
<div class="fig">{image_tag("app_predictor.png", 62)}<div class="cap">Figure 7. Predictor tab: curves and 95% bands for the three compounds at Bahrain, with the circuit profile measured from telemetry.</div></div>
<div class="fig">{image_tag("app_explorer.png", 62)}<div class="cap">Figure 8. Race explorer tab: one driver's real stints (dots) against the degradation model (solid line), the stint's own trend (dashed) and a detected tyre cliff (red dotted line).</div></div>
<div class="fig">{image_tag("app_circuits.png", 62)}<div class="cap">Figure 9. Circuits tab: predicted lap-time loss after 20 laps on soft tyres at 35 C.</div></div>
<h2>7. Validation checks</h2>
<ul>
<li><b>No leakage.</b> A model trained on shuffled targets scores R&sup2; {checks['label_shuffle']['r2']:.2f} and MAE {checks['label_shuffle']['mae_s']:.2f}, the same as the baseline.</li>
<li><b>Not a lucky season.</b> Leave-one-year-out gives similar errors for every year (below), so regulation changes do not break the model.</li>
<li><b>Overfitting.</b> Pre-lap train MAE {gap['pre_lap']['train_mae_s']:.3f} against test {gap['pre_lap']['test_mae_s']:.3f}; live {gap['live']['train_mae_s']:.3f} against {gap['live']['test_mae_s']:.3f}.</li>
<li><b>Not by chance.</b> Resampling the 24 test races 1,000 times, the pre-lap MAE has a 95% interval of {ci['gb_pre_lap'][0]:.2f} to {ci['gb_pre_lap'][1]:.2f} s and beats Formula B by {gain['gb_pre_lap']['mean_s']:.2f} s ({gain['gb_pre_lap']['ci95'][0]:.2f} to {gain['gb_pre_lap']['ci95'][1]:.2f}) in every resample.</li>
<li><b>Intervals.</b> The calibrated 90% prediction intervals cover {100 * intervals['pre_lap']['calibrated_coverage']:.0f}% (pre-lap) and {100 * intervals['pre_stint']['calibrated_coverage']:.0f}% (pre-stint) and {100 * intervals['live']['calibrated_coverage']:.0f}% (live) of 2025 laps, with mean widths of {intervals['pre_lap']['calibrated_mean_width_s']:.1f} s, {intervals['pre_stint']['calibrated_mean_width_s']:.1f} s and {intervals['live']['calibrated_mean_width_s']:.1f} s.</li>
</ul>
<p>Leave-one-year-out, MAE in seconds (R&sup2;):</p>
{table(["Held-out year", "Pre-lap", "Pre-stint", "Live"], loyo_rows)}
<p>Feature-group ablation, pre-lap model, 2025 MAE in seconds:</p>
{table(["Features", "MAE (s)"], ablation_rows, ("all_features",))}
<div class="cap">Removing the tyre features costs about 0.04 s, about as much as removing fuel progress or traffic (0.05 s): tyre age is one contributor among several.</div>

<h2>8. Limitations</h2>
<ul>
<li>A single lap carries about 0.33 s of random noise from traffic and mistakes, so lap-level R&sup2; has a ceiling near 0.9 for any model; most pre-lap error is each driver's race-specific offset, which no pre-lap information reveals.</li>
<li>Circuit characteristics are measured once per circuit from one fast lap (the latest season available), so layout changes between seasons are ignored, and with only 25 circuits their effects cannot be separated reliably.</li>
<li>The target is relative to the race's field-median lap, which comes from the session itself, not from a separate forecast.</li>
<li>Compound differences in the average curves are small next to circuit, fuel and traffic effects, and drivers manage their tyres, so the model captures typical behaviour, not a strategy decision.</li>
<li>The live model is mostly persistence: the plain last-3-lap average already reaches R&sup2; {hold['persistence_last3_laps']['r2']:.2f}.</li>
<li>Some design choices (the lap filter and feature set) were made while looking at 2025 results, so the 2025 holdout is slightly optimistic; cross-validation is the better estimate.</li>
</ul>

<h2>9. Conclusion</h2>
<p>A from-scratch gradient boosting model predicts lap time relative to the field with MAE {hold['gb_pre_lap']['mae_s']:.3f} s (R&sup2; {hold['gb_pre_lap']['r2']:.2f}) using only information available before the lap, against {hold['median_baseline']['mae_s']:.3f} s for a naive baseline and {hold['formula_b_with_fuel_and_form']['mae_s']:.3f} s for a fitted formula, and reproduces real tyre-age curves on unseen races. Knowing how the driver went in earlier stints cuts the error to {hold['gb_pre_stint']['mae_s']:.3f} s (R&sup2; {hold['gb_pre_stint']['r2']:.2f}). With the driver's recent laps, error drops to {hold['gb_live']['mae_s']:.3f} s, {persist_gain:.0f}% better than simple persistence.</p>
</body></html>"""


def render_pdf(html_path, pdf_path):
    js = "const { chromium } = require('playwright'); (async () => { const b = await chromium.launch(); const p = await b.newPage(); await p.goto('file://' + process.argv[1]); await p.pdf({ path: process.argv[2], format: 'A4', printBackground: true, margin: { top: '18mm', bottom: '18mm', left: '18mm', right: '18mm' } }); await b.close(); })();"
    try:
        subprocess.run(["node", "-e", js, html_path, pdf_path], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("PDF not rendered (needs node and playwright); open report.html in a browser and print to PDF.")
        return False
    return True


def main():
    with open(os.path.join(ROOT, "results", "metrics.json")) as f:
        report = json.load(f)
    with open(os.path.join(ROOT, "results", "degradation.json")) as f:
        deg = json.load(f)
    out_dir = os.path.join(ROOT, "report")
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(build_html(report, deg))
    if render_pdf(html_path, os.path.join(out_dir, "report.pdf")):
        print("wrote report/report.html and report/report.pdf")


if __name__ == "__main__":
    main()
