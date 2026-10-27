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
  gbt_pre_lap: "Boosted trees, pre-lap",
  persistence_last3_laps: "Persistence (last 3 laps)",
  gbt_live: "Boosted trees, live",
};
const ORDER = Object.keys(NAMES);
const KEY_ROWS = new Set(["gbt_pre_lap", "gbt_live"]);

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
const persistGain = Math.round(100 * (1 - hold.gbt_live.mae_s / hold.persistence_last3_laps.mae_s));
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
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

function figure(file, captionText) {
  const buffer = fs.readFileSync(path.join(ROOT, "results", file));
  const { width, height } = pngSize(buffer);
  const targetWidth = Math.round((CONTENT / 1440) * 96);
  return [
    new Paragraph({ keepNext: true, spacing: { before: 120, after: 40 }, children: [new ImageRun({ type: "png", data: buffer, transformation: { width: targetWidth, height: Math.round((targetWidth * height) / width) }, altText: { title: captionText, description: captionText, name: file } })] }),
    caption(captionText),
  ];
}

const holdRows = ORDER.map((k) => [k, [NAMES[k], f3(hold[k].mae_s), f3(hold[k].median_ae_s), r2(hold[k].r2), f3(hold[k].curve_mae_s), f2(hold[k].slope_corr)]]);
const cvRows = ORDER.map((k) => [k, [NAMES[k], f3(cv[k].mae_s), r2(cv[k].r2), f3(cv[k].fold_mae_std)]]);
const loyoRows = Object.entries(loyo).map(([y, m]) => [y, [y, `${f3(m.gbt_pre_lap.mae_s)} (${f2(m.gbt_pre_lap.r2)})`, `${f3(m.gbt_live.mae_s)} (${f2(m.gbt_live.r2)})`]]);
const ablationRows = Object.entries(ablation).map(([k, v]) => [k, [k === "all_features" ? "All features" : "Without " + k.replace("drop_", "").replace(/_/g, " "), f3(v)]]);

const children = [
  new Paragraph({ indent: { right: CONTENT - 900 }, border: { bottom: { style: BorderStyle.SINGLE, size: 24, color: RED, space: 1 } }, spacing: { after: 160 }, children: [] }),
  new Paragraph({ children: [new TextRun({ text: "Predicting F1 Lap Time and Tyre Degradation", font: FONT, size: 42, bold: true })], spacing: { after: 60 } }),
  new Paragraph({ children: [run("Machine learning on FastF1 race data, 2022 to 2025. All models implemented from scratch with NumPy and pandas.", { color: GREY })], spacing: { after: 240 } }),

  heading("1. Problem"),
  para(["As a tyre wears, a car gets slower, so teams need to know how much time a driver loses per lap of tyre age to decide when to pit. We predict how many seconds slower or faster a lap will be than the race's field-median lap, from the tyre, the fuel load, the weather, the car's recent form and the traffic. Degradation appears as the effect of tyre age once fuel burn is removed."]),

  heading("2. Data"),
  para([`Lap-by-lap timing, tyre and weather data from the FastF1 API for 92 races from 2022 to 2025. We keep dry, green-flag laps only: lap 1, pit laps, safety car, virtual safety car and yellow-flag laps, rain, deleted laps and laps slower than 104% of the driver's best are removed. That leaves 70,830 laps from 90 races. Fuel burn (about ${f2(report.fuel_effect_total_s_fitted)} s over a full race) is estimated from the data.`]),

  heading("3. Method"),
  para([["Target. "], "Lap time minus the race's field-median lap time, in seconds. This removes circuit length and year-to-year pace differences."]),
  para([["Features. "], "Tyre compound and age (with non-linear terms and an age-by-compound interaction), lap number and race fraction as a fuel proxy, stint, track and air temperature, humidity, wind, driver and team form (their average pace over earlier races only), traffic (gaps to the cars ahead and behind and position at the end of the previous lap), and field context (the field's pace trend and laps since the last safety car or yellow flag)."]),
  para([["Model. "], "Stochastic gradient-boosted regression trees written from scratch (Friedman's LAD-TreeBoost generalised to a pinball, or quantile, loss). Features are binned into histograms; each round fits a tree to the negative gradient of the loss, with row and column subsampling, and each leaf is set to the quantile of the residuals it contains. At the median this is the absolute-error loss, which is robust to traffic and mistakes. Quantile versions at 5% and 95%, calibrated with split conformal prediction, give 90% prediction intervals. Ridge regression and least squares are also implemented in NumPy. Against XGBoost on the same split the implementation differs by under 0.002 s in MAE."]),
  para([["Two tiers. "], "The pre-lap model uses everything known at the start of the lap except the driver's own previous lap times. The live model adds the driver's previous clean laps in the same stint, like a pit-wall forecast."]),
  para([["Baselines. "], "The median lap; persistence (the driver's last 3 laps averaged); Formula A, the structure of an online degradation calculator (base plus compound, track temperature, air temperature and abrasiveness factors times laps) with constants fitted to our data; Formula B, Formula A plus fuel and form; and ridge regression on the same features as the pre-lap model."]),
  para([["Validation. "], "Training on 2022 to 2024 (52,092 laps, 66 races) and testing on all of 2025 (18,738 laps, 24 unseen races). Mixed-year 5-fold cross-validation by race, where each fold holds out about 20% of every year's races, and leave-one-year-out. Laps from one race are never split between training and test."]),

  heading("4. Results"),
  para(["Unseen 2025 races. MAE is the typical error in seconds; curve error compares predicted and actual average lap-time loss by tyre age."]),
  table(["Model", "MAE (s)", "Median AE (s)", "R²", "Curve error (s)", "Slope corr."], holdRows, [3000, 1300, 1500, 1000, 1700, 1366], KEY_ROWS),
  caption(`Persistence is the fair baseline for the live model. The live model improves on it by ${persistGain}% in MAE.`),
  para(["Mixed-year cross-validation across all 90 races, the main performance estimate:"]),
  table(["Model", "MAE (s)", "R²", "Fold MAE std"], cvRows, [4400, 1800, 1800, 1866], KEY_ROWS),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
  ...figure("degradation_curves.png", "Figure 1. Fuel-corrected lap-time loss by tyre age on unseen 2025 races: actual and predicted."),
  ...figure("pred_vs_actual.png", "Figure 2. Predicted against actual lap time relative to the field median, 2025 test laps."),

  heading("5. Validation checks"),
  bullet([["No leakage. "], `A model trained on shuffled targets scores R² ${f2(checks.label_shuffle.r2)} and MAE ${f2(checks.label_shuffle.mae_s)}, the same as the baseline.`]),
  bullet([["Not a lucky season. "], "Leave-one-year-out gives similar errors for every year (below), so regulation changes do not break the model."]),
  bullet([["Overfitting. "], `Pre-lap train MAE ${f3(gap.pre_lap.train_mae_s)} against test ${f3(gap.pre_lap.test_mae_s)}; live ${f3(gap.live.train_mae_s)} against ${f3(gap.live.test_mae_s)}.`]),
  bullet([["Not by chance. "], `Resampling the 24 test races 1,000 times, the pre-lap MAE has a 95% interval of ${f2(ci.gbt_pre_lap[0])} to ${f2(ci.gbt_pre_lap[1])} s and beats Formula B by ${f2(gain.gbt_pre_lap.mean_s)} s (${f2(gain.gbt_pre_lap.ci95[0])} to ${f2(gain.gbt_pre_lap.ci95[1])}) in every resample.`]),
  bullet([["Intervals. "], `The calibrated 90% prediction intervals cover ${Math.round(100 * intervals.pre_lap.calibrated_coverage)}% (pre-lap) and ${Math.round(100 * intervals.live.calibrated_coverage)}% (live) of 2025 laps, with mean widths of ${intervals.pre_lap.calibrated_mean_width_s.toFixed(1)} s and ${intervals.live.calibrated_mean_width_s.toFixed(1)} s.`]),
  para(["Leave-one-year-out, MAE in seconds (R²):"], { keepNext: true }),
  table(["Held-out year", "Pre-lap", "Live"], loyoRows, [3000, 3433, 3433]),
  new Paragraph({ spacing: { after: 100 }, children: [] }),
  para(["Feature-group ablation, pre-lap model, 2025 MAE in seconds:"], { keepNext: true }),
  table(["Features", "MAE (s)"], ablationRows, [7000, 2866], new Set(["all_features"])),
  caption("Removing the tyre features costs about 0.06 s, as much as removing fuel progress or traffic: tyre age is one contributor among several."),

  heading("6. Limitations"),
  bullet(["A single lap carries about 0.33 s of random noise from traffic and mistakes, so lap-level R² has a ceiling near 0.9 for any model; most pre-lap error is each driver's race-specific offset, which no pre-lap information reveals."]),
  bullet(["The target is relative to the race's field-median lap, which comes from the session itself, not from a separate forecast."]),
  bullet(["Compound differences in the average curves are small next to circuit, fuel and traffic effects, and drivers manage their tyres, so the model captures typical behaviour, not a strategy decision."]),
  bullet([`The live model is mostly persistence: the plain last-3-lap average already reaches R² ${f2(hold.persistence_last3_laps.r2)}.`]),
  bullet(["Some design choices (the lap filter and feature set) were made while looking at 2025 results, so the 2025 holdout is slightly optimistic; cross-validation is the better estimate."]),

  heading("7. Conclusion"),
  para([`A from-scratch gradient-boosted tree model predicts lap time relative to the field with MAE ${f3(hold.gbt_pre_lap.mae_s)} s (R² ${f2(hold.gbt_pre_lap.r2)}) using only information available before the lap, against ${f3(hold.median_baseline.mae_s)} s for a naive baseline and ${f3(hold.formula_b_with_fuel_and_form.mae_s)} s for a fitted formula, and reproduces real tyre-age curves on unseen races. With the driver's recent laps, error drops to ${f3(hold.gbt_live.mae_s)} s, ${persistGain}% better than simple persistence.`]),
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
