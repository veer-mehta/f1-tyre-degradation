import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from f1tyre import config, serving


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
.f1-stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 4px; }
.f1-stat { background: #121214; border: 1px solid #222226; padding: 10px 14px; }
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


@st.cache_resource(show_spinner="Loading the models")
def load_cached():
    return serving.load()


def themed(chart, label_color="#8E8E93"):
    return chart.configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor=label_color, titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E", labelFontSize=12, titleFontSize=12).configure_legend(labelColor="#F2F2F2")


def label(text):
    st.markdown(f'<div class="f1-label">{text}</div>', unsafe_allow_html=True)


def prediction_chart(kept, result):
    colors = alt.Scale(domain=["Actual", "Pre-lap model", "Live model"], range=["#F2F2F2", "#8E8E93", "#E10600"])
    frames = [pd.DataFrame({"lap": kept["LapNumber"], "time": kept["LapTime"], "series": "Actual"}).dropna(), pd.DataFrame({"lap": kept["LapNumber"], "time": result["pre_lap"], "series": "Pre-lap model"}), pd.DataFrame({"lap": kept["LapNumber"], "time": result["live"], "series": "Live model"})]
    long = pd.concat(frames, ignore_index=True)
    x = alt.X("lap:Q", title="Lap", scale=alt.Scale(zero=False), axis=alt.Axis(format="d", tickMinStep=1))
    y = alt.Y("time:Q", title="Lap time (s)", scale=alt.Scale(zero=False))
    color = alt.Color("series:N", scale=colors, legend=alt.Legend(orient="top", title=None, symbolOpacity=1))
    lines = alt.Chart(long[long["series"] != "Actual"]).mark_line(strokeWidth=2.5).encode(x=x, y=y, color=color)
    dots = alt.Chart(long[long["series"] == "Actual"]).mark_circle(size=70, opacity=0.9).encode(x=x, y=y, color=color, tooltip=["lap", "time"])
    return themed(alt.layer(lines, dots).properties(height=320))


def result_table(kept, result):
    html = "<table><tr><th>Lap</th><th>Tyre age</th><th>Compound</th><th>Actual (s)</th><th>Pre-lap (s)</th><th>Live (s)</th><th>Pre-lap error</th><th>Live error</th></tr>"
    for i in range(len(kept)):
        actual = kept["LapTime"].iloc[i]
        known = not np.isnan(actual)
        pre, live = result["pre_lap"][i], result["live"][i]
        html += f"<tr><td>{kept['LapNumber'].iloc[i]:.0f}</td><td>{kept['TyreLife'].iloc[i]:.0f}</td><td>{kept['Compound'].iloc[i].title()}</td><td>{actual:.3f}</td>" if known else f"<tr><td>{kept['LapNumber'].iloc[i]:.0f}</td><td>{kept['TyreLife'].iloc[i]:.0f}</td><td>{kept['Compound'].iloc[i].title()}</td><td>-</td>"
        live_text = f"{live:.3f}" if not np.isnan(live) else "-"
        live_error = f"{live - actual:+.3f}" if known and not np.isnan(live) else "-"
        pre_error = f"{pre - actual:+.3f}" if known else "-"
        html += f"<td>{pre:.3f}</td><td>{live_text}</td><td>{pre_error}</td><td>{live_error}</td></tr>"
    return html + "</table>"


def lap_time_tab(served, defaults, circuits, example_data):
    events = ["None"] + sorted(circuits.index)
    c1, c2 = st.columns([2, 1.3])
    with c1:
        label("Circuit (optional)")
        circuit = st.selectbox("Circuit", events, index=events.index("Bahrain Grand Prix") if "Bahrain Grand Prix" in events else 0, label_visibility="collapsed", key="lap_circuit")
    settings = serving.default_settings(defaults, None if circuit == "None" else circuit)
    climb = float(circuits.loc[circuit, "climb_m_per_km"]) if circuit != "None" else defaults["mean_climb"]
    label("Laps: edit the table, paste rows or upload a CSV (columns LapNumber, LapTime, Compound, TyreLife, TrackTemp). Leave LapTime blank to forecast a lap")
    upload = st.file_uploader("Upload a CSV of laps", type="csv", label_visibility="collapsed")
    example, overrides = example_data
    source = pd.read_csv(upload) if upload is not None else example
    if "Compound" in source:
        source["Compound"] = source["Compound"].astype(str).str.upper().str.strip()
    base = {} if upload is not None else overrides
    columns = {"LapNumber": st.column_config.NumberColumn("Lap", min_value=1, step=1), "LapTime": st.column_config.NumberColumn("Lap time (s)", format="%.3f"), "Compound": st.column_config.SelectboxColumn("Compound", options=config.COMPOUNDS, required=True), "TyreLife": st.column_config.NumberColumn("Tyre life (laps)", min_value=1, step=1), "TrackTemp": st.column_config.NumberColumn("Track temp (C)", format="%.1f")}
    source_id = upload.name if upload is not None else "example"
    table = st.data_editor(source, num_rows="dynamic", use_container_width=True, hide_index=True, column_config=columns, key=f"laps_editor_{source_id}")
    rows = serving.clean_rows(table)
    known_times = rows["LapTime"].dropna()
    fallback = settings["median_s"] or (float(known_times.median()) if len(known_times) else 90.0)
    with c2:
        label("Race median lap time (s)")
        settings["median_s"] = st.number_input("Race median lap time", 40.0, 200.0, round(float(fallback), 3), 0.1, format="%.3f", label_visibility="collapsed", key=f"median_{circuit}_{source_id}")
    if len(known_times) and abs(float(known_times.median()) - settings["median_s"]) > 0.03 * settings["median_s"]:
        st.warning(f"Your lap times are around {float(known_times.median()):.1f} s but the race median lap time is {settings['median_s']:.1f} s. Predictions are relative to that median, so set it to the typical lap time of the race your laps come from, or pick the right circuit.")
    with st.expander("More inputs (for your own laps the defaults are neutral; the example uses the real values)"):
        key = f"_{source_id}_{circuit}"
        a1, a2, a3 = st.columns(3)
        settings["stint"] = a1.number_input("Stint number", 1, 5, int(base.get("stint", 1)), key="stint" + key)
        settings["fresh_tyre"] = a1.checkbox("New tyre set", base.get("fresh_tyre", True), key="fresh" + key)
        settings["total_laps"] = a1.number_input("Race length (laps)", 20, 90, int(settings["total_laps"]), key="total" + key)
        settings["driver_ratio"] = a2.number_input("Driver pace vs field median (ratio, below 1 is faster)", 0.97, 1.05, round(base.get("driver_ratio", 1.0), 3), 0.001, format="%.3f", key="ratio" + key)
        settings["field_vs_cum"] = a2.number_input("Field pace trend (s, negative = field getting faster)", -3.0, 3.0, round(base.get("field_vs_cum", 0.0), 2), 0.05, key="trend" + key)
        settings["laps_since_neutral"] = a2.number_input("Laps since the last safety car or yellow flag", 0, 40, int(round(base.get("laps_since_neutral", 40))), key="neutral" + key)
        settings["position"] = a3.number_input("Position", 1, 20, int(base.get("position", 10)), key="pos" + key)
        settings["gap_ahead_s"] = a3.number_input("Gap to the car ahead (s)", 0.0, 60.0, round(base.get("gap_ahead_s", 5.0), 2), key="ahead" + key)
        settings["gap_behind_s"] = a3.number_input("Gap to the car behind (s)", 0.0, 60.0, round(base.get("gap_behind_s", 5.0), 2), key="behind" + key)
        b1, b2, b3 = st.columns(3)
        air = b1.number_input("Air temperature (C), 0 to estimate", 0.0, 60.0, round(base.get("air_temp", 0.0), 1), key="air" + key)
        settings["air_temp"] = air if air > 0 else None
        settings["humidity"] = b2.number_input("Humidity (%)", 0.0, 100.0, round(base.get("humidity", settings["humidity"]), 1), key="hum" + key)
        settings["wind_speed"] = b3.number_input("Wind speed (m/s)", 0.0, 20.0, round(base.get("wind_speed", settings["wind_speed"]), 1), key="wind" + key)
    kept, built = serving.build_features(table, settings, climb, defaults)
    if kept.empty:
        st.info("Add at least one lap with a compound, tyre life and track temperature.")
        return
    result = serving.predict_laps(served, built, settings["median_s"])
    result["live"] = np.where(built["prev_delta"].isna().to_numpy(), np.nan, result["live"])
    st.write("")
    st.altair_chart(prediction_chart(kept, result), use_container_width=True, theme=None)
    st.markdown(result_table(kept, result), unsafe_allow_html=True)
    has_actual = kept["LapTime"].notna().to_numpy()
    if has_actual.any():
        pre_mae = float(np.abs(result["pre_lap"][has_actual] - kept["LapTime"].to_numpy()[has_actual]).mean())
        live_ok = has_actual & ~np.isnan(result["live"])
        live_mae = float(np.abs(result["live"][live_ok] - kept["LapTime"].to_numpy()[live_ok]).mean()) if live_ok.any() else float("nan")
        st.markdown(f'<div class="f1-stats" style="grid-template-columns: repeat(3, 1fr); margin-top: 14px"><div class="f1-stat"><div class="num">{int(has_actual.sum())}</div><div class="cap">Laps with a time</div></div><div class="f1-stat"><div class="num">{pre_mae:.3f} s</div><div class="cap">Pre-lap error (MAE)</div></div><div class="f1-stat"><div class="num">{live_mae:.3f} s</div><div class="cap">Live error (MAE)</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="f1-note">The gradient boosting models, written from scratch, predict a lap relative to the race median lap, so a race median is needed to give seconds: it is taken from the circuit (latest season) or from your own laps, and you can change it. The pre-lap model ignores lap times; the live model also uses the lap times you entered before each lap, so it needs an earlier lap and shows a dash for the first one. Inputs you leave out (weather, traffic, driver pace, how the field is trending) take neutral defaults, which makes predictions worse: these race conditions matter, and the example shows the models with the real values filled in. The circuit effect is its elevation change. Both models were trained on all races from 2022 to 2025, so the example stint is part of their training data and fits better than new laps would; the honest error on unseen races is on the Model tab.</div>', unsafe_allow_html=True)


def lap_model_table(report):
    holdout = report["holdout_2025"]
    rows = [("Median baseline", "median_baseline"), ("Formula (fuel and form)", "formula_b_with_fuel_and_form"), ("Ridge regression", "ridge_pre_lap_features"), ("Gradient boosting, pre-lap", "gb_pre_lap"), ("Gradient boosting, pre-stint", "gb_pre_stint"), ("Persistence, last 3 laps", "persistence_last3_laps"), ("Gradient boosting, live", "gb_live")]
    html = "<table><tr><th>Model</th><th>MAE (s)</th><th>R&sup2;</th><th>Tyre-age curve error (s)</th></tr>"
    for name, key in rows:
        m = holdout[key]
        html += f"<tr><td>{name}</td><td>{m['mae_s']:.3f}</td><td>{m['r2'] if m['r2'] > 0.005 else 0.0:.2f}</td><td>{m['curve_mae_s']:.3f}</td></tr>"
    return html + "</table>"


def verify_section(verify):
    label("Verify the scores yourself: run saved models on 2025 laps they never saw")
    st.markdown(f'<div class="f1-note" style="margin-top: 0">The models below were trained on 2022 to 2024 only. The button runs them on all {len(verify["y"]):,} clean laps of 2025 and computes MAE and R&sup2; here, with the from-scratch code. R&sup2; is the share of lap-time variation explained (0.82 means 82%), not the share of laps predicted correctly.</div>', unsafe_allow_html=True)
    if st.button("Run on the 2025 test laps"):
        with st.spinner("Predicting"):
            scores = serving.verify_scores(verify)
        html = "<table><tr><th>Model</th><th>MAE (s)</th><th>R&sup2;</th></tr>"
        for name, (mae, r2) in scores.items():
            html += f"<tr><td>{name}</td><td>{mae:.3f}</td><td>{max(r2, 0.0):.3f}</td></tr>"
        st.markdown(html + "</table>", unsafe_allow_html=True)
    st.write("")


def headline(report):
    holdout = report["holdout_2025"]
    coverage = report["prediction_intervals_90"]["live"]["calibrated_coverage"]
    tiles = [(f"{holdout['gb_pre_lap']['mae_s']:.2f} s", "Pre-lap error (MAE)"), (f"{holdout['gb_pre_stint']['mae_s']:.2f} s", "Pre-stint error (MAE)"), (f"{holdout['gb_live']['mae_s']:.2f} s", "Live error (MAE)"), (f"{holdout['gb_live']['r2']:.2f}", "Live R&sup2;"), (f"{100 * coverage:.0f}%", "Coverage of 90% intervals")]
    blocks = "".join(f'<div class="f1-stat"><div class="num">{n}</div><div class="cap">{c}</div></div>' for n, c in tiles)
    st.markdown(f'<div class="f1-stats" style="grid-template-columns: repeat(5, 1fr)">{blocks}</div>', unsafe_allow_html=True)
    st.write("")


def model_tab(verify):
    path = config.RESULTS_DIR / "metrics.json"
    if not path.exists():
        st.write("Run evaluate.py to generate results.")
        return
    report = json.loads(path.read_text())
    headline(report)
    label("Lap-time models: unseen 2025 races, trained on 2022-2024")
    st.markdown(lap_model_table(report), unsafe_allow_html=True)
    st.markdown('<div class="f1-note">Gradient boosting implemented from scratch in NumPy. Pre-lap uses everything known at the start of the lap except the driver\'s own previous lap times; pre-stint adds how the driver went in earlier stints of the race; live adds the recent laps of the current stint. Target is lap time relative to the race\'s field-median lap.</div>', unsafe_allow_html=True)
    st.write("")
    verify_section(verify)


def main():
    st.set_page_config(page_title="F1 Lap Time Prediction", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="f1-bar"></div><div class="f1-title">Lap Time Prediction</div><div class="f1-sub">Predict an F1 lap from tyre, fuel, weather and traffic</div>', unsafe_allow_html=True)
    a = load_cached()
    tab_laps, tab_model = st.tabs(["Lap times", "Model"])
    with tab_laps:
        lap_time_tab(a["served"], a["defaults"], a["circuits"], a["example"])
    with tab_model:
        model_tab(a["verify"])


main()
