// Builds report/report.docx (editable Word version of the report) from results/metrics.json.
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat,
  Packer, PageNumber, Paragraph, ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const report = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "metrics.json"), "utf8"));
const NAMES = {
  median_baseline: "Median baseline",
  formula_a_calculator_style: "Formula A (calculator style)",
  formula_b_with_fuel_and_form: "Formula B (fuel and form)",
  ridge_pre_lap_features: "Ridge regression, pre-lap features",
  gb_pre_lap: "Gradient boosting, pre-lap",
  gb_pre_stint: "Gradient boosting, pre-stint",
  persistence_last3_laps: "Persistence (last 3 laps)",
  gb_live: "Gradient boosting, live",
};
const ORDER = Object.keys(NAMES);
const KEY_ROWS = new Set(["gb_pre_lap", "gb_pre_stint", "gb_live"]);

const FONT = "Arial";
const RED = "E10600";
const GREY = "666666";
const PAGE = { width: 11906, height: 16838 };
const MARGIN = 1020;
const CONTENT = PAGE.width - 2 * MARGIN;

const hold = report.holdout_2025;
const cv = report.mixed_year_cv;
const loyo = report.leave_one_year_out;
const checks = report.sanity_checks;
const ablation = checks.ablation_pre_lap_test_mae_s;
const ci = checks.mae_ci95;
const gain = checks.gain_over_formula_b;
const gap = checks.train_vs_test;
const intervals = report.prediction_intervals_90;
const persistGain = Math.round(100 * (1 - hold.gb_live.mae_s / hold.persistence_last3_laps.mae_s));
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const deg = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "degradation.json"), "utf8"));
const DESCRIPTOR_TEXT = {
  length_km: "lap length", avg_speed_kmh: "average speed", top_speed_kmh: "top speed", min_speed_kmh: "slowest-corner speed",
  full_throttle_pct: "full-throttle share", brake_pct: "braking share", heavy_braking_zones: "heavy braking zones",
  mean_lat_g: "mean cornering load", high_lat_pct: "share of the lap above 2.5 g", slow_corners: "slow corners",
  medium_corners: "medium corners", fast_corners: "fast corners", climb_m_per_km: "elevation change per kilometre",
};
const VARIANT_NAMES = [
  ["compound, curvature, temperature", "Compound, tyre age and track temperature"],
  ["+ selected circuit characteristics", "+ elevation change (selected)"],
  ["+ all 13 circuit characteristics", "+ all 13 circuit characteristics"],
  ["+ circuit identity (each circuit's own history)", "+ each circuit's own history"],
  ["gradient boosting, pre-lap without circuit feature", "Gradient boosting, pre-lap, no circuit feature"],
  ["gradient boosting, pre-lap (lap-level model)", "Gradient boosting, pre-lap, with elevation change"],
];
const dv = deg.holdout_2025;
const selected = deg.selection.selected_descriptors;
const slopes = dv["+ selected circuit characteristics"].mean_slope_by_compound;
const base = dv["compound, curvature, temperature"];
const chosen = dv["+ selected circuit characteristics"];
const every = dv["+ all 13 circuit characteristics"];
const history = dv["+ circuit identity (each circuit's own history)"];
const cl = deg.cliff.summary;
const fit = deg.stint_fit_in_sample;
const yearPairs = Object.values(deg.severity_year_to_year_correlation).map((x) => f2(x)).join(", ");
const selectedText = selected.map((c) => DESCRIPTOR_TEXT[c]).join(" and ");
const compoundRows = ["SOFT", "MEDIUM", "HARD"].map((c) => [c, [c[0] + c.slice(1).toLowerCase(), f3(slopes.true[c]), f3(slopes.pred[c])]]);
const variantRows = VARIANT_NAMES.filter(([k]) => dv[k]).map(([k, label]) => [k, [label, f2(dv[k].within_r2), f2(dv[k].slope_corr), dv[k].slope_mae_s_per_lap.toFixed(4)]]);
const details = JSON.parse(fs.readFileSync(path.join(__dirname, "details.json"), "utf8"));
const r2 = (v) => f2(v > 0.005 ? v : 0);

function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: 21, ...opts });
}

// Paragraph from segments: strings are plain, [text] arrays are bold.
function para(segments, opts = {}) {
  const children = segments.map((s) => (Array.isArray(s) ? run(s[0], { bold: true }) : run(s)));
  return new Paragraph({ children, spacing: { after: 110, line: 300 }, ...opts });
}

function bullet(segments) {
  return para(segments, { numbering: { reference: "bullets", level: 0 }, spacing: { after: 70, line: 290 } });
}

function heading(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT })] });
}

function caption(text) {
  return new Paragraph({ children: [run(text, { size: 18, color: GREY })], spacing: { before: 80, after: 160 } });
}

function table(headers, rows, widths, keyIds = new Set()) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const cellBorders = (bottom) => ({ top: none, left: none, right: none, bottom });
  const headerBorder = { style: BorderStyle.SINGLE, size: 12, color: "1A1A1A" };
  const rowBorder = { style: BorderStyle.SINGLE, size: 4, color: "E6E6E6" };
  const cell = (text, i, width, bold, border) =>
    new TableCell({
      width: { size: width, type: WidthType.DXA },
      borders: cellBorders(border),
      shading: { fill: "FFFFFF", type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ alignment: i ? AlignmentType.RIGHT : AlignmentType.LEFT, children: [run(String(text), { size: 19, bold, color: bold && border === headerBorder ? "555555" : undefined })] })],
    });
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, widths[i], true, headerBorder)) });
  const body = rows.map(([id, cells]) => new TableRow({ children: cells.map((c, i) => cell(c, i, widths[i], keyIds.has(id), rowBorder)) }));
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}

function pngSize(buffer) {
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

function figure(file, captionText, scale = 1) {
  const buffer = fs.readFileSync(path.join(ROOT, file.startsWith("app_") ? "report" : "results", file));
  const { width, height } = pngSize(buffer);
  const targetWidth = Math.round((CONTENT / 1440) * 96 * scale);
  return [
    new Paragraph({ keepNext: true, spacing: { before: 120, after: 40 }, children: [new ImageRun({ type: "png", data: buffer, transformation: { width: targetWidth, height: Math.round((targetWidth * height) / width) }, altText: { title: captionText, description: captionText, name: file } })] }),
    caption(captionText),
  ];
}

function detailsTable() {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const borders = { top: none, bottom: none, left: none, right: none };
  const widths = [1300, 3633, 1300, 3633];
  const rows = [["Name", details.name, "Roll No.", details.roll_no], ["College", details.college, "Semester", details.semester], ["Subject", details.subject, "Professor", details.professor], ["Date", details.date, "", ""]];
  const cell = (text, i, isLabel) => new TableCell({ width: { size: widths[i], type: WidthType.DXA }, borders, margins: { top: 20, bottom: 20, left: 0, right: 120 }, children: [new Paragraph({ children: [run(text, isLabel ? { size: 16, color: "777777", allCaps: true, characterSpacing: 20 } : { size: 20 })] })] });
  return new Table({ width: { size: CONTENT, type: WidthType.DXA }, columnWidths: widths, rows: rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, i, i % 2 === 0)) })) });
}

const holdRows = ORDER.map((k) => [k, [NAMES[k], f3(hold[k].mae_s), f3(hold[k].median_ae_s), r2(hold[k].r2), f3(hold[k].curve_mae_s), f2(hold[k].slope_corr)]]);
const cvRows = ORDER.map((k) => [k, [NAMES[k], f3(cv[k].mae_s), r2(cv[k].r2), f3(cv[k].fold_mae_std)]]);
const loyoRows = Object.entries(loyo).map(([y, m]) => [y, [y, `${f3(m.gb_pre_lap.mae_s)} (${f2(m.gb_pre_lap.r2)})`, `${f3(m.gb_pre_stint.mae_s)} (${f2(m.gb_pre_stint.r2)})`, `${f3(m.gb_live.mae_s)} (${f2(m.gb_live.r2)})`]]);
const later = report.holdout_2025_stint_2_plus;
const laterRows = Object.entries(later.scores).map(([k, m]) => [k, [NAMES[k], f3(m.mae_s), f3(m.median_ae_s), f2(m.r2)]]);
const curveFit = deg.curve_check_2025.by_shape_2025;
const curveAlt = Object.entries(curveFit).filter(([k]) => k !== "quadratic").map(([, v]) => v);
const ablationRows = Object.entries(ablation).filter(([k]) => k !== "drop_earlier_stints").map(([k, v]) => [k, [k === "all_features" ? "All features" : "Without " + k.replace("drop_", "").replace(/_/g, " "), f3(v)]]);

const children = [
  new Paragraph({ indent: { right: CONTENT - 900 }, border: { bottom: { style: BorderStyle.SINGLE, size: 24, color: RED, space: 1 } }, spacing: { after: 160 }, children: [] }),
  new Paragraph({ children: [new TextRun({ text: "Predicting F1 Lap Time and Tyre Degradation", font: FONT, size: 42, bold: true })], spacing: { after: 60 } }),
  new Paragraph({ children: [run("Machine learning on FastF1 race data, 2022 to 2025. All models implemented from scratch with NumPy and pandas.", { color: GREY })], spacing: { after: 200 } }),
  detailsTable(),
  new Paragraph({ spacing: { after: 120 }, children: [] }),

  heading("1. Problem"),
  para(["As a tyre wears, a car gets slower, so teams need to know how much time a driver loses per lap of tyre age to decide when to pit. We predict how many seconds slower or faster a lap will be than the race's field-median lap, from the tyre, the fuel load, the weather, the car's recent form and the traffic. Degradation appears as the effect of tyre age once fuel burn is removed."]),

  heading("2. Data"),
  para([`Lap-by-lap timing, tyre and weather data from the FastF1 API for 92 races from 2022 to 2025. We keep dry, green-flag laps only: lap 1, pit laps, safety car, virtual safety car and yellow-flag laps, rain, deleted laps and laps slower than 104% of the driver's best are removed. That leaves 70,830 laps from 90 races. Fuel burn (about ${f2(report.fuel_effect_total_s_fitted)} s over a full race) is estimated from the data.`]),

  heading("3. Method"),
  para([["Target. "], "Lap time minus the race's field-median lap time, in seconds. This removes circuit length and year-to-year pace differences."]),
  para([["Features. "], "Tyre compound and age (with non-linear terms and an age-by-compound interaction), lap number and race fraction as a fuel proxy, stint, track and air temperature, humidity, wind, driver and team form (their average pace over earlier races only), traffic (gaps to the cars ahead and behind and position at the end of the previous lap), and field context (the field's pace trend and laps since the last safety car or yellow flag)."]),
  para([["Model. "], "Gradient boosting with regression trees, written from scratch (Friedman's LAD-TreeBoost generalised to a pinball, or quantile, loss). Features are binned into histograms; each round fits a tree to the negative gradient of the loss, with row and column subsampling, and each leaf is set to the quantile of the residuals it contains. At the median this is the absolute-error loss, which is robust to traffic and mistakes. Quantile versions at 5% and 95%, calibrated with split conformal prediction, give 90% prediction intervals. Ridge regression and least squares are also implemented in NumPy. Against XGBoost on the same split the implementation differs by under 0.002 s in MAE."]),
  para([["Three tiers. "], "The pre-lap model uses everything known at the start of the lap except the driver's own previous lap times. The pre-stint model adds how the driver went in the earlier stints of the same race (their mean pace, the pace at the end of the previous stint and its length). The live model also adds the driver's previous clean laps in the same stint, like a pit-wall forecast."]),
  para([["Baselines. "], "The median lap; persistence (the driver's last 3 laps averaged); Formula A, the structure of an online degradation calculator (base plus compound, track temperature, air temperature and abrasiveness factors times laps) with constants fitted to our data; Formula B, Formula A plus fuel and form; and ridge regression on the same features as the pre-lap model."]),
  para([["Validation. "], "Training on 2022 to 2024 (52,092 laps, 66 races) and testing on all of 2025 (18,738 laps, 24 unseen races). Mixed-year 5-fold cross-validation by race, where each fold holds out about 20% of every year's races, and leave-one-year-out. Laps from one race are never split between training and test."]),

  heading("4. Results"),
  para(["Unseen 2025 races. MAE is the typical error in seconds; curve error compares predicted and actual average lap-time loss by tyre age."]),
  table(["Model", "MAE (s)", "Median AE (s)", "R²", "Curve error (s)", "Slope corr."], holdRows, [3000, 1300, 1500, 1000, 1700, 1366], KEY_ROWS),
  caption(`Persistence is the fair baseline for the live model. The live model improves on it by ${persistGain}% in MAE.`),
  para(["Mixed-year cross-validation across all 90 races, the main performance estimate:"]),
  table(["Model", "MAE (s)", "R²", "Fold MAE std"], cvRows, [4400, 1800, 1800, 1866], KEY_ROWS),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
  para([`Laps in stint 2 or later (${Math.round(100 * later.share_of_test_laps)}% of 2025 laps), where the pre-stint model can use the driver's earlier stints:`], { keepNext: true }),
  table(["Model", "MAE (s)", "Median AE (s)", "R²"], laterRows, [4400, 1800, 1800, 1866], KEY_ROWS),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
  ...figure("degradation_curves.png", "Figure 1. Fuel-corrected lap-time loss by tyre age on unseen 2025 races: actual and predicted."),
  ...figure("pred_vs_actual.png", "Figure 2. Predicted against actual lap time relative to the field median, 2025 test laps.", 0.85),

  heading("5. Tyre degradation and circuit characteristics"),
  para(["The lap-level model predicts lap time, so tyre age is only one of several effects it learns, and comparing different drivers and stints confounds the compound with car, driver and strategy. Asking it what-if questions, such as a soft tyre running 30 laps in a late stint, also falls outside the data it has seen, which flattened the compound differences in an early version of our app. We therefore estimate degradation directly: every stint is compared only with itself (stint fixed effects) after removing fuel burn, so driver pace, car and strategy cancel out."]),
  para([["Degradation model. "], "A least-squares regression written in NumPy explains the within-stint lap-time change by tyre age for each compound (linear and quadratic terms), its interaction with track temperature, and its interaction with circuit characteristics."]),
  para([`We also tried flexible piecewise-linear curve shapes, which can show a flat start or a late cliff. Judged by how well the curve reproduces the observed average tyre-age curves on held-out races, the parabola was best (curve error ${f3(curveFit.quadratic)} s on unseen 2025 stints against ${f3(Math.min(...curveAlt))} to ${f3(Math.max(...curveAlt))} s for the flexible shapes), so it is kept; the data do not support a cliff or a flat start beyond what it captures. Figure 4 compares the observed averages with the model. The shaded bands in Figure 3 are 95% intervals from resampling whole races.`]),
  para([["Circuit characteristics. "], "For each of the 25 circuits we measured 13 characteristics from the telemetry of the fastest race lap: lap length, average speed, top speed, slowest-corner speed, the shares of the lap at full throttle and on the brakes, the number of heavy braking zones, mean cornering load in g, the share of the lap above 2.5 g, the numbers of slow, medium and fast corners, and elevation change per kilometre. Which ones to include was decided by cross-validation inside the training seasons only."]),
  para(["Average degradation rate on unseen 2025 stints (at least 15 laps), in seconds per lap of tyre age:"], { keepNext: true }),
  table(["Compound", "Actual", "Predicted"], compoundRows, [4000, 2933, 2933]),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
  para(["Within-stint accuracy on unseen 2025 stints:"], { keepNext: true }),
  table(["Model", "Within-stint R²", "Slope corr.", "Slope error (s/lap)"], variantRows, [4600, 1700, 1500, 2066]),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
  ...figure("degradation_by_circuit.png", "Figure 3. Predicted lap-time loss by tyre age and compound at the circuits with the lowest and highest elevation change (shaded: 95% interval)."),
  ...figure("curve_check.png", "Figure 4. Observed average lap-time loss by tyre age (dots, sized by the number of stints) against the model (line) on unseen 2025 stints, relative to tyre age 3."),
  ...figure("circuit_characteristics.png", "Figure 5. Correlation of each circuit characteristic with measured degradation severity across 25 circuits (selected characteristic in red).", 0.68),
  ...figure("cliff.png", "Figure 6. Tyre cliffs in stints of 20 or more laps: tyre age at which the loss speeds up (left) and the share of stints that speed up late against the noise reference of stints that slow down late (right)."),
  bullet([["Compounds separate clearly. "], `Soft tyres lose ${f3(slopes.true.SOFT)} s per lap, medium ${f3(slopes.true.MEDIUM)} and hard ${f3(slopes.true.HARD)}, so soft degrades about ${(slopes.true.SOFT / slopes.true.HARD).toFixed(1)} times faster than hard, and the model reproduces this (${f3(slopes.pred.SOFT)}, ${f3(slopes.pred.MEDIUM)}, ${f3(slopes.pred.HARD)}).`]),
  bullet([["One circuit characteristic held up. "], `Cross-validation on the training seasons selected ${selectedText} (correlation ${f2(deg.descriptor_spearman_with_severity[selected[0]])} with measured circuit severity). The link is statistically real but modest: it stays significant after allowing for picking the best of the 13 characteristics (p = ${f2(deg.descriptor_selection_p_value.p_value_best_of_columns)}) and after dropping any one circuit (${f2(deg.descriptor_selection_p_value.elevation_rho_leave_one_circuit_out_range[0])} to ${f2(deg.descriptor_selection_p_value.elevation_rho_leave_one_circuit_out_range[1])}). No mechanism is proven: elevation change may stand in for other features of those circuits, such as layout and surface. Adding it improves the held-out stint slope correlation from ${f2(base.slope_corr)} to ${f2(chosen.slope_corr)}. Using all 13 characteristics or each circuit's own history does not help: within-stint R² ${f2(every.within_r2)} and ${f2(history.within_r2)} against ${f2(base.within_r2)}.`]),
  bullet([["Circuit degradation is not predictable from these characteristics alone. "], `Predicting a circuit's severity from all 13 gives R² ${f2(deg.severity_predicted_from_all_descriptors.leave_one_circuit_out_r2)} on circuits left out. Tyre allocation, weather and race incidents also matter, and a circuit's severity is not stable from year to year (correlation between consecutive seasons: ${yearPairs}).`]),

  bullet([["Tyre cliffs exist, but in a minority of stints. "], `Each of the ${cl.ALL.stints} stints of 20 or more laps was fitted with a straight line and with a line that steepens at one age; a cliff needs that age to be in the second half of the stint. ${(100 * cl.ALL.cliff_share).toFixed(1)}% show a clear cliff (at least 0.1 s per lap of extra loss), against ${(100 * cl.ALL.relief_share_noise_reference).toFixed(1)}% that slow down by the same amount, which is what chance alone gives. About ${Math.round(100 * (cl.ALL.cliff_share - cl.ALL.relief_share_noise_reference))}% of long stints therefore have a real cliff, typically ${f2(cl.ALL.median_extra_slope_s_per_lap)} s per lap steeper afterwards, starting at a median age of ${Math.round(cl.ALL.median_onset_age)} laps (soft ${Math.round(cl.SOFT.median_onset_age)}, medium ${Math.round(cl.MEDIUM.median_onset_age)}, hard ${Math.round(cl.HARD.median_onset_age)}). Most stints never reach one because teams pit first, and onset ages vary, so an average curve cannot show it.`]),
  bullet([["The curve describes the average lap, not every stint. "], `Over all ${fit.stints.toLocaleString("en-US")} stints the model sits ${Math.round(fit.model_below_average_stint_pct)}% below the average stint slope (soft ${f3(fit.by_compound.SOFT.model)} against ${f3(fit.by_compound.SOFT.actual)} s per lap) and ${Math.round(100 * fit.share_of_stints_steeper_than_model)}% of stints are steeper than it. Stints of 18 laps or fewer average ${f3(fit.stints_18_laps_or_fewer.actual)} against ${f3(fit.stints_18_laps_or_fewer.model)} modelled; stints over 25 laps average ${f3(fit.stints_over_25_laps.actual)} against ${f3(fit.stints_over_25_laps.model)}. The fit weighs laps equally, so long stints dominate, and long stints are the gentle ones because tyres that wear fast are changed early. Stint slopes also differ a great deal (modelled and actual slopes correlate at ${f2(fit.slope_corr)}). Weighting stints equally removes the gap but fits unseen 2025 stints worse, so it was not used.`]),
  bullet([["In the lap-level model it helps tyre behaviour, not lap time. "], `Adding elevation change raises the stint slope correlation of the pre-lap gradient boosting model from ${f2(dv["gradient boosting, pre-lap without circuit feature"].slope_corr)} to ${f2(dv["gradient boosting, pre-lap (lap-level model)"].slope_corr)}, but its lap-time error is not better (MAE ${f3(ablation.drop_circuit)} s without, ${f3(ablation.all_features)} s with).`]),

  heading("6. The app"),
  para(["A Streamlit app with four tabs makes the models explorable. ", ["Predictor"], " draws the degradation curves for the three compounds at any circuit and track temperature, with a band for the average (95%) or for single stints (middle 80%) and a second view that adds fuel burn, so lap times can fall while the tyre wears. ", ["Race explorer"], " plots the real stints of any driver in any race from 2022 to 2025 against what the model expects and marks detected tyre cliffs. ", ["Circuits"], " ranks the 25 circuits by predicted tyre loss. ", ["Model"], " shows the scores from this report, a check of the curve against every stint, and the tyre-cliff results."]),
  ...figure("app_predictor.png", "Figure 7. Predictor tab: curves and 95% bands for the three compounds at Bahrain, with the circuit profile measured from telemetry.", 0.62),
  ...figure("app_explorer.png", "Figure 8. Race explorer tab: one driver's real stints (dots) against the degradation model (solid line), the stint's own trend (dashed) and a detected tyre cliff (red dotted line).", 0.62),
  ...figure("app_circuits.png", "Figure 9. Circuits tab: predicted lap-time loss after 20 laps on soft tyres at 35 C.", 0.62),
  heading("7. Validation checks"),
  bullet([["No leakage. "], `A model trained on shuffled targets scores R² ${f2(checks.label_shuffle.r2)} and MAE ${f2(checks.label_shuffle.mae_s)}, the same as the baseline.`]),
  bullet([["Not a lucky season. "], "Leave-one-year-out gives similar errors for every year (below), so regulation changes do not break the model."]),
  bullet([["Overfitting. "], `Pre-lap train MAE ${f3(gap.pre_lap.train_mae_s)} against test ${f3(gap.pre_lap.test_mae_s)}; live ${f3(gap.live.train_mae_s)} against ${f3(gap.live.test_mae_s)}.`]),
  bullet([["Not by chance. "], `Resampling the 24 test races 1,000 times, the pre-lap MAE has a 95% interval of ${f2(ci.gb_pre_lap[0])} to ${f2(ci.gb_pre_lap[1])} s and beats Formula B by ${f2(gain.gb_pre_lap.mean_s)} s (${f2(gain.gb_pre_lap.ci95[0])} to ${f2(gain.gb_pre_lap.ci95[1])}) in every resample.`]),
  bullet([["Intervals. "], `The calibrated 90% prediction intervals cover ${Math.round(100 * intervals.pre_lap.calibrated_coverage)}% (pre-lap), ${Math.round(100 * intervals.pre_stint.calibrated_coverage)}% (pre-stint) and ${Math.round(100 * intervals.live.calibrated_coverage)}% (live) of 2025 laps, with mean widths of ${intervals.pre_lap.calibrated_mean_width_s.toFixed(1)} s, ${intervals.pre_stint.calibrated_mean_width_s.toFixed(1)} s and ${intervals.live.calibrated_mean_width_s.toFixed(1)} s.`]),
  para(["Leave-one-year-out, MAE in seconds (R²):"], { keepNext: true }),
  table(["Held-out year", "Pre-lap", "Pre-stint", "Live"], loyoRows, [2400, 2488, 2488, 2488]),
  new Paragraph({ spacing: { after: 100 }, children: [] }),
  para(["Feature-group ablation, pre-lap model, 2025 MAE in seconds:"], { keepNext: true }),
  table(["Features", "MAE (s)"], ablationRows, [7000, 2866], new Set(["all_features"])),
  caption("Removing the tyre features costs about 0.04 s, about as much as removing fuel progress or traffic (0.05 s): tyre age is one contributor among several."),

  heading("8. Limitations"),
  bullet(["A single lap carries about 0.33 s of random noise from traffic and mistakes, so lap-level R² has a ceiling near 0.9 for any model; most pre-lap error is each driver's race-specific offset, which no pre-lap information reveals."]),
  bullet(["Circuit characteristics are measured once per circuit from one fast lap (the latest season available), so layout changes between seasons are ignored, and with only 25 circuits their effects cannot be separated reliably."]),
  bullet(["The target is relative to the race's field-median lap, which comes from the session itself, not from a separate forecast."]),
  bullet(["Compound differences in the average curves are small next to circuit, fuel and traffic effects, and drivers manage their tyres, so the model captures typical behaviour, not a strategy decision."]),
  bullet([`The live model is mostly persistence: the plain last-3-lap average already reaches R² ${f2(hold.persistence_last3_laps.r2)}.`]),
  bullet(["Some design choices (the lap filter and feature set) were made while looking at 2025 results, so the 2025 holdout is slightly optimistic; cross-validation is the better estimate."]),

  heading("9. Conclusion"),
  para([`A from-scratch gradient boosting model predicts lap time relative to the field with MAE ${f3(hold.gb_pre_lap.mae_s)} s (R² ${f2(hold.gb_pre_lap.r2)}) using only information available before the lap, against ${f3(hold.median_baseline.mae_s)} s for a naive baseline and ${f3(hold.formula_b_with_fuel_and_form.mae_s)} s for a fitted formula, and reproduces real tyre-age curves on unseen races. Knowing how the driver went in earlier stints cuts the error to ${f3(hold.gb_pre_stint.mae_s)} s (R² ${f2(hold.gb_pre_stint.r2)}). With the driver's recent laps, error drops to ${f3(hold.gb_live.mae_s)} s, ${persistGain}% better than simple persistence.`]),
];

const doc = new Document({
  title: "Predicting F1 Lap Time and Tyre Degradation",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, bold: true, allCaps: true, characterSpacing: 30 },
        paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0, keepNext: true, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "DDDDDD", space: 4 } } },
      },
    ],
  },
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }],
  },
  sections: [{
    properties: { page: { size: PAGE, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: GREY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = path.join(__dirname, "report.docx");
  fs.writeFileSync(out, buffer);
  console.log("wrote", out);
});
