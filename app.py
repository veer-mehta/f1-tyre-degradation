import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from f1tyre import config, data, degradation, features, models
from f1tyre.circuits import DESCRIPTOR_LABELS


COMPOUND_COLORS = {"Soft": "#E10600", "Medium": "#FFD12E", "Hard": "#F2F2F2"}
MAX_AGE = 30
PROFILE_STATS = [("length_km", "{:.1f} km"), ("avg_speed_kmh", "{:.0f} km/h"), ("full_throttle_pct", "{:.0f}%"), ("brake_pct", "{:.0f}%"), ("mean_lat_g", "{:.2f} g"), ("climb_m_per_km", "{:.0f} m/km")]
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Titillium Web', sans-serif; }
.stApp { background: #0B0B0C; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.2rem; max-width: 1100px; }
.f1-bar { height: 4px; width: 64px; background: #E10600; margin-bottom: 14px; }
.f1-title { font-size: 1.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; line-height: 1.1; }
.f1-sub { color: #8E8E93; font-size: 0.95rem; margin: 6px 0 26px 0; font-weight: 300; }
.f1-label { color: #8E8E93; font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 4px; }
.f1-card { background: #121214; border: 1px solid #222226; border-left: 3px solid var(--c); padding: 14px 18px; }
.f1-card .name { font-size: 0.8rem; letter-spacing: 0.14em; text-transform: uppercase; color: #BDBDC2; font-weight: 600; }
.f1-card .row { display: flex; gap: 20px; margin-top: 8px; }
.f1-card .num { font-size: 1.35rem; font-weight: 700; line-height: 1.05; white-space: nowrap; }
.f1-card .cap { color: #8E8E93; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; white-space: nowrap; }
.f1-stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 4px; }
.f1-stat { background: #121214; border: 1px solid #222226; padding: 10px 14px; }
.f1-stat.used { border-left: 3px solid #E10600; }
.f1-stat .num { font-size: 1.15rem; font-weight: 700; white-space: nowrap; }
.f1-stat .cap { color: #8E8E93; font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase; }
.f1-note { color: #6F6F76; font-size: 0.82rem; margin-top: 18px; line-height: 1.5; }
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid #222226; }
.stTabs [data-baseweb="tab"] { padding: 8px 0; letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.8rem; font-weight: 600; }
div[data-baseweb="select"] > div { background: #121214; border-color: #222226; }
table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
th { text-align: left; color: #8E8E93; font-weight: 600; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; border-bottom: 1px solid #2A2A2E; padding: 8px 6px; }
td { border-bottom: 1px solid #1B1B1E; padding: 9px 6px; }
</style>
"""


def read_report():
    path = config.RESULTS_DIR / "degradation.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_resource(show_spinner="Fitting the degradation model on 2022-2025 races")
def load_model():
    laps = features.build_lap_table(data.load_laps())
    circuits = data.load_circuits()
    report = read_report()
    panel = degradation.build_panel(laps, models.estimate_fuel_effect(laps))
    model = degradation.model_from_report(report).fit(panel, circuits)
    draws_path = config.RESULTS_DIR / "degradation_bootstrap.csv"
    draws = pd.read_csv(draws_path) if draws_path.exists() else None
    totals = laps.groupby("EventName")["TotalLaps"].max().to_dict()
    return model, circuits, report["final_model"]["circuit_columns"], draws, totals, models.estimate_fuel_effect(laps), panel


def label(text):
    st.markdown(f'<div class="f1-label">{text}</div>', unsafe_allow_html=True)


def chart(curves, y_title):
    colors = alt.Scale(domain=list(COMPOUND_COLORS), range=list(COMPOUND_COLORS.values()))
    x = alt.X("age:Q", title="Tyre age (laps)", scale=alt.Scale(domain=[1, MAX_AGE]))
    color = alt.Color("compound:N", scale=colors, legend=alt.Legend(orient="top", title=None, symbolType="stroke", symbolStrokeWidth=4, symbolOpacity=1, labelFontSize=13))
    line = alt.Chart(curves).mark_line(strokeWidth=3, interpolate="monotone").encode(x=x, y=alt.Y("loss:Q", title=y_title), color=color)
    if "low" not in curves:
        return line.properties(height=380).configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#8E8E93", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E", labelFontSize=12, titleFontSize=12).configure_legend(labelColor="#F2F2F2")
    band = alt.Chart(curves).mark_area(opacity=0.14, interpolate="monotone").encode(x=x, y="low:Q", y2="high:Q", color=color)
    base = band + line
    return base.properties(height=380).configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#8E8E93", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E", labelFontSize=12, titleFontSize=12).configure_legend(labelColor="#F2F2F2")


def card(compound, curves):
    g = curves[curves["compound"] == compound].set_index("age")["loss"]
    color = COMPOUND_COLORS[compound]
    return f'<div class="f1-card" style="--c:{color}"><div class="name">{compound}</div><div class="row"><div><div class="num">{g.loc[10]:+.2f}s</div><div class="cap">10 laps</div></div><div><div class="num">{g.loc[20]:+.2f}s</div><div class="cap">20 laps</div></div><div><div class="num">{g.loc[30]:+.2f}s</div><div class="cap">30 laps</div></div></div></div>'


def circuit_profile(row, selected):
    blocks = ""
    for column, fmt in PROFILE_STATS:
        used = " used" if column in selected else ""
        blocks += f'<div class="f1-stat{used}"><div class="num">{fmt.format(row[column])}</div><div class="cap">{DESCRIPTOR_LABELS[column].split(" (")[0]}</div></div>'
    return f'<div class="f1-stats">{blocks}</div>'


def predictor_tab(model, circuits, selected, draws, totals, fuel_total_s):
    c1, c2, c3 = st.columns([2, 1.3, 1.5])
    with c1:
        label("Circuit")
        events = sorted(circuits.index)
        event = st.selectbox("Circuit", events, index=events.index("Bahrain Grand Prix") if "Bahrain Grand Prix" in events else 0, label_visibility="collapsed")
    with c2:
        label("Track temperature (C)")
        track_temp = st.slider("Track temperature", 15, 55, 35, label_visibility="collapsed")
    with c3:
        label("View")
        view = st.radio("View", ["Tyre effect", "Lap time with fuel burn"], horizontal=True, label_visibility="collapsed")
    ages = np.arange(1, MAX_AGE + 1)
    frames = []
    for c in config.COMPOUNDS:
        frame = pd.DataFrame({"compound": c.title(), "age": ages, "loss": model.curve(c, circuits.loc[event], track_temp, ages)})
        if draws is not None:
            frame["low"], frame["high"] = model.curve_band(c, circuits.loc[event], track_temp, ages, draws)
        frames.append(frame)
    curves = pd.concat(frames, ignore_index=True)
    fuel_view = view.startswith("Lap time")
    if fuel_view:
        burn = fuel_total_s / totals[event] * (curves["age"] - 1)
        for column in [c for c in ["loss", "low", "high"] if c in curves]:
            curves[column] = curves[column] - burn
    st.write("")
    st.altair_chart(chart(curves, "Lap time vs first lap of the stint (s)" if fuel_view else "Lap-time loss vs fresh tyre (s)"), use_container_width=True, theme=None)
    for col, compound in zip(st.columns(3), COMPOUND_COLORS):
        col.markdown(card(compound, curves), unsafe_allow_html=True)
    st.write("")
    label("Circuit profile (from fastest-lap telemetry)")
    st.markdown(circuit_profile(circuits.loc[event], selected), unsafe_allow_html=True)
    used = ", ".join(DESCRIPTOR_LABELS[c].lower() for c in selected) or "none"
    st.markdown(f'<div class="f1-note">Each stint is compared only with itself, so driver, car and strategy cancel out; the model is fitted on 2022-2025 races, and the shaded bands are 95% intervals from resampling races. Besides compound and tyre age it uses track temperature and the circuit characteristics that held up in testing: {used} (marked in red). The other circuit characteristics showed no reliable link to degradation. Treat the curves as typical behaviour, not a strategy call. "Lap time with fuel burn" adds the time gained as the car gets lighter (about {fuel_total_s / totals[event]:.2f} s per lap here), so lap times can fall even while the tyre degrades.</div>', unsafe_allow_html=True)


def lap_model_table(report):
    holdout = report["holdout_2025"]
    rows = [("Median baseline", "median_baseline"), ("Formula (fuel and form)", "formula_b_with_fuel_and_form"), ("Ridge regression", "ridge_pre_lap_features"), ("Gradient boosting, pre-lap", "gb_pre_lap"), ("Gradient boosting, pre-stint", "gb_pre_stint"), ("Persistence, last 3 laps", "persistence_last3_laps"), ("Gradient boosting, live", "gb_live")]
    html = "<table><tr><th>Model</th><th>MAE (s)</th><th>R&sup2;</th><th>Tyre-age curve error (s)</th></tr>"
    for name, key in rows:
        m = holdout[key]
        html += f"<tr><td>{name}</td><td>{m['mae_s']:.3f}</td><td>{m['r2'] if m['r2'] > 0.005 else 0.0:.2f}</td><td>{m['curve_mae_s']:.3f}</td></tr>"
    return html + "</table>"


def degradation_table(deg):
    slopes = deg["holdout_2025"]["+ selected circuit characteristics"]["mean_slope_by_compound"]
    html = "<table><tr><th>Compound</th><th>Actual (s per lap)</th><th>Predicted (s per lap)</th></tr>"
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        html += f"<tr><td>{compound.title()}</td><td>{slopes['true'][compound]:.3f}</td><td>{slopes['pred'][compound]:.3f}</td></tr>"
    return html + "</table>"


def curve_check_chart():
    path = config.RESULTS_DIR / "curve_check.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    table["Compound"] = table["Compound"].str.title()
    colors = alt.Scale(domain=list(COMPOUND_COLORS), range=list(COMPOUND_COLORS.values()))
    x = alt.X("age:Q", title="Tyre age (laps)")
    dots = alt.Chart(table).mark_circle(opacity=0.85).encode(x=x, y=alt.Y("actual:Q", title="Lap-time loss vs age 3 (s)"), size=alt.Size("n:Q", legend=None, scale=alt.Scale(range=[30, 200])), color=alt.Color("Compound:N", scale=colors, legend=None))
    line = alt.Chart(table).mark_line(color="#F2F2F2", strokeWidth=2).encode(x=x, y="predicted:Q")
    return alt.layer(dots, line).properties(width=270, height=220).facet(column=alt.Column("Compound:N", title=None, sort=["Soft", "Medium", "Hard"], header=alt.Header(labelColor="#BDBDC2", labelFontSize=12))).configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#8E8E93", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E")


def correlation_chart(deg):
    selected = deg["selection"]["selected_descriptors"]
    frame = pd.DataFrame({"characteristic": [DESCRIPTOR_LABELS[c] for c in deg["descriptor_spearman_with_severity"]], "correlation": list(deg["descriptor_spearman_with_severity"].values()), "used": ["Used in model" if c in selected else "Not used" for c in deg["descriptor_spearman_with_severity"]]})
    bars = alt.Chart(frame).mark_bar().encode(x=alt.X("correlation:Q", title="Correlation with measured degradation (25 circuits)"), y=alt.Y("characteristic:N", sort="-x", title=None), color=alt.Color("used:N", scale=alt.Scale(domain=["Used in model", "Not used"], range=["#E10600", "#55555A"]), legend=alt.Legend(orient="top", title=None))).properties(height=330)
    return bars.configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#BDBDC2", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E").configure_legend(labelColor="#F2F2F2")


def stint_chart(frame):
    colors = alt.Scale(domain=list(COMPOUND_COLORS), range=list(COMPOUND_COLORS.values()))
    x = alt.X("age:Q", title="Tyre age (laps)")
    dots = alt.Chart(frame).mark_circle(size=70, opacity=0.9).encode(x=x, y=alt.Y("actual:Q", title="Lap time vs stint average, fuel-corrected (s)"), color=alt.Color("Compound:N", scale=colors, legend=None), tooltip=["age", "actual", "predicted"])
    line = alt.Chart(frame).mark_line(color="#F2F2F2", strokeWidth=2).encode(x=x, y="predicted:Q")
    trend = alt.Chart(frame).mark_line(color="#8E8E93", strokeWidth=1.5, strokeDash=[5, 4]).encode(x=x, y="trend:Q")
    return alt.layer(dots, trend, line).properties(height=260).facet(column=alt.Column("stint_label:N", title=None, header=alt.Header(labelColor="#BDBDC2", labelFontSize=12))).configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#8E8E93", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E")


def explorer_tab(model, circuits, panel):
    c1, c2, c3 = st.columns([1, 2, 1.5])
    with c1:
        label("Season")
        year = st.selectbox("Season", sorted(panel["Year"].unique(), reverse=True), label_visibility="collapsed")
    with c2:
        label("Race")
        events = sorted(panel[panel["Year"] == year]["EventName"].unique())
        event = st.selectbox("Race", events, label_visibility="collapsed")
    race = panel[(panel["Year"] == year) & (panel["EventName"] == event)].reset_index(drop=True)
    with c3:
        label("Driver")
        driver = st.selectbox("Driver", sorted(race["Driver"].unique()), label_visibility="collapsed")
    stints = race[race["Driver"] == driver].reset_index(drop=True)
    stints["actual"] = degradation.within_stint(stints["y"], stints)
    stints["predicted"] = degradation.within_stint(pd.Series(model.predict_level(stints, circuits), index=stints.index), stints)
    stints["Compound"] = stints["Compound"].str.title()
    stints["trend"] = degradation.within_stint(stints["age"], stints) * degradation.stint_slopes(stints["y"], stints).reindex(pd.MultiIndex.from_frame(stints[degradation.STINT_KEY])).to_numpy()
    stints["stint_label"] = stints.apply(lambda r: f"Stint {int(r['Stint'])}: {r['Compound']}", axis=1)
    st.write("")
    st.altair_chart(stint_chart(stints), use_container_width=True, theme=None)
    slopes = degradation.stint_slopes(stints["y"], stints)
    fitted = degradation.stint_slopes(stints["predicted"], stints)
    html = "<table><tr><th>Stint</th><th>Compound</th><th>Clean laps</th><th>Actual (s per lap)</th><th>Model (s per lap)</th></tr>"
    for (key, actual), predicted in zip(slopes.items(), fitted.values):
        group = stints[stints["Stint"] == key[2]]
        html += f"<tr><td>{int(key[2])}</td><td>{group['Compound'].iloc[0]}</td><td>{len(group)}</td><td>{actual:+.3f}</td><td>{predicted:+.3f}</td></tr>"
    st.markdown(html + "</table>", unsafe_allow_html=True)
    st.markdown('<div class="f1-note">Real stints from the data: dots are the driver\'s dry, green-flag laps after removing fuel burn and the stint average, the white line is what the degradation model expects for a typical stint and the dashed grey line is this stint\'s own straight-line trend. The Model tab shows how the typical line compares with all stints. Single stints are noisy (traffic, tyre management, a lift for fuel saving); the model describes the typical case. Only stints of at least 8 clean laps are shown.</div>', unsafe_allow_html=True)


def circuits_tab(model, circuits, selected):
    c1, c2, c3 = st.columns([1.5, 1.5, 2])
    with c1:
        label("Compound")
        compound = st.radio("Compound", [c.title() for c in config.COMPOUNDS], horizontal=True, label_visibility="collapsed", key="rank_compound")
    with c2:
        label("Track temperature (C)")
        track_temp = st.slider("Track temperature", 15, 55, 35, label_visibility="collapsed", key="rank_temp")
    with c3:
        label("Laps on the tyre")
        laps_on = st.slider("Laps on the tyre", 10, 30, 20, label_visibility="collapsed")
    ages = np.array([1, laps_on])
    frame = pd.DataFrame({"circuit": [e.replace(" Grand Prix", "") for e in circuits.index], "loss": [float(model.curve(compound.upper(), circuits.loc[e], track_temp, ages)[-1]) for e in circuits.index]})
    st.write("")
    bars = alt.Chart(frame).mark_bar(color=COMPOUND_COLORS[compound]).encode(x=alt.X("loss:Q", title=f"Predicted lap-time loss after {laps_on} laps (s)"), y=alt.Y("circuit:N", sort="-x", title=None), tooltip=["circuit", alt.Tooltip("loss:Q", format=".2f")]).properties(height=620)
    st.altair_chart(bars.configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#BDBDC2", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E"), use_container_width=True, theme=None)
    used = ", ".join(DESCRIPTOR_LABELS[c].lower() for c in selected) or "none"
    st.markdown(f'<div class="f1-note">Circuits ranked by the model for one compound and temperature. With only {used} selected as a circuit effect, differences between circuits are modest and should be read as tendencies, not facts: the 95% bands on the Predictor tab show how much they overlap.</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def slope_table(_model, _circuits, _panel):
    pred = pd.Series(_model.predict_within(_panel, _circuits), index=_panel.index)
    keyed = _panel.groupby(degradation.STINT_KEY)
    return pd.DataFrame({"actual": degradation.stint_slopes(_panel["y"], _panel), "model": degradation.stint_slopes(pred, _panel), "Compound": keyed["Compound"].first().str.title(), "laps": keyed["y"].size()}).reset_index(drop=True)


def slope_check_chart(table):
    colors = alt.Scale(domain=list(COMPOUND_COLORS), range=list(COMPOUND_COLORS.values()))
    x_scale = alt.Scale(domain=[0, 0.12], nice=False, clamp=True)
    y_scale = alt.Scale(domain=[-0.1, 0.3], nice=False, clamp=True)
    dots = alt.Chart(table).mark_circle(size=22, opacity=0.35).encode(x=alt.X("model:Q", title="Model slope (s per lap)", scale=x_scale), y=alt.Y("actual:Q", title="Actual slope of the stint (s per lap)", scale=y_scale), color=alt.Color("Compound:N", scale=colors, legend=alt.Legend(orient="top", title=None, symbolOpacity=1)), tooltip=["Compound", "laps", alt.Tooltip("actual:Q", format=".3f"), alt.Tooltip("model:Q", format=".3f")])
    diagonal = alt.Chart(pd.DataFrame({"a": [0.0, 0.12]})).mark_line(color="#F2F2F2", strokeDash=[5, 4]).encode(x="a:Q", y="a:Q")
    return alt.layer(dots, diagonal).properties(height=340).configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#8E8E93", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E").configure_legend(labelColor="#F2F2F2")


def slope_check(table):
    by_compound = table.groupby("Compound")[["actual", "model"]].mean()
    rows = "".join(f"<tr><td>{c}</td><td>{by_compound.loc[c, 'actual']:.3f}</td><td>{by_compound.loc[c, 'model']:.3f}</td></tr>" for c in ["Soft", "Medium", "Hard"])
    short = table[table["laps"] <= 18]
    long = table[table["laps"] > 25]
    html = f"<table><tr><th>Compound</th><th>Average actual stint (s per lap)</th><th>Model (s per lap)</th></tr>{rows}</table>"
    note = f'Over all {len(table):,} stints the model is on average {100 * (1 - table["model"].mean() / table["actual"].mean()):.0f}% below the average stint slope, and {100 * (table["actual"] > table["model"]).mean():.0f}% of stints are steeper than the line. Stints of 18 laps or fewer average {short["actual"].mean():.3f} s per lap against {short["model"].mean():.3f} modelled; stints over 25 laps average {long["actual"].mean():.3f} against {long["model"].mean():.3f}. The fit weighs every lap equally, so long stints dominate, and long stints are the gentle ones, because tyres that wear fast are changed early. Individual stints vary a lot (correlation between modelled and actual slope {table["actual"].corr(table["model"]):.2f}), so the line describes the typical lap, not each stint. These stints were used to fit the model, so this is not a held-out check.'
    return html, note


def headline(report, deg):
    holdout = report["holdout_2025"]
    tiles = [(f"{holdout['gb_pre_lap']['mae_s']:.2f} s", "Pre-lap error (MAE)"), (f"{holdout['gb_pre_stint']['mae_s']:.2f} s", "Pre-stint error (MAE)"), (f"{holdout['gb_live']['mae_s']:.2f} s", "Live error (MAE)"), (f"{holdout['gb_live']['r2']:.2f}", "Live R&sup2;"), ("1.7x", "Soft vs hard wear")]
    blocks = "".join(f'<div class="f1-stat"><div class="num">{n}</div><div class="cap">{c}</div></div>' for n, c in tiles)
    st.markdown(f'<div class="f1-stats" style="grid-template-columns: repeat(5, 1fr)">{blocks}</div>', unsafe_allow_html=True)
    st.write("")


def model_tab(model, circuits, panel):
    path = config.RESULTS_DIR / "metrics.json"
    deg = read_report()
    if not path.exists() or not deg:
        st.write("Run evaluate.py and evaluate_degradation.py to generate results.")
        return
    report = json.loads(path.read_text())
    headline(report, deg)
    label("Lap-time models: unseen 2025 races, trained on 2022-2024")
    st.markdown(lap_model_table(report), unsafe_allow_html=True)
    st.markdown('<div class="f1-note">Gradient boosting implemented from scratch in NumPy. Pre-lap uses everything known at the start of the lap except the driver\'s own previous lap times; pre-stint adds how the driver went in earlier stints of the race; live adds the recent laps of the current stint. Target is lap time relative to the race\'s field-median lap.</div>', unsafe_allow_html=True)
    st.write("")
    label("Degradation model: average degradation by compound, unseen 2025 stints")
    st.markdown(degradation_table(deg), unsafe_allow_html=True)
    st.write("")
    label("Reality check: observed average loss (dots) vs the model (line), unseen 2025 stints")
    check = curve_check_chart()
    if check is not None:
        st.altair_chart(check, use_container_width=True, theme=None)
    st.write("")
    label("Does the line fit every stint? Actual against modelled slope, all stints 2022-2025")
    table = slope_table(model, circuits, panel)
    st.altair_chart(slope_check_chart(table), use_container_width=True, theme=None)
    html, note = slope_check(table)
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(f'<div class="f1-note">{note}</div>', unsafe_allow_html=True)
    st.write("")
    label("Which circuit characteristics affect degradation?")
    st.altair_chart(correlation_chart(deg), use_container_width=True, theme=None)
    st.markdown('<div class="f1-note">Thirteen characteristics were measured from telemetry. Only elevation change was chosen by cross-validation on the training seasons, and it improves held-out stint slopes modestly. Predicting a circuit\'s degradation from all thirteen fails on circuits left out, because tyre allocation, weather and race events also matter.</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="F1 Tyre Degradation", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="f1-bar"></div><div class="f1-title">Tyre Degradation</div><div class="f1-sub">How much slower a stint gets with every lap on the tyre</div>', unsafe_allow_html=True)
    model, circuits, selected, draws, totals, fuel_total_s, panel = load_model()
    tab_predict, tab_explore, tab_circuits, tab_model = st.tabs(["Predictor", "Race explorer", "Circuits", "Model"])
    with tab_predict:
        predictor_tab(model, circuits, selected, draws, totals, fuel_total_s)
    with tab_explore:
        explorer_tab(model, circuits, panel)
    with tab_circuits:
        circuits_tab(model, circuits, selected)
    with tab_model:
        model_tab(model, circuits, panel)


main()
