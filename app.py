import json
import altair as alt
import pandas as pd
import streamlit as st
from f1tyre import config, data, features, models


COMPOUND_COLORS = {"Soft": "#E10600", "Medium": "#FFD12E", "Hard": "#F2F2F2"}
MAX_AGE = 30
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
.f1-note { color: #6F6F76; font-size: 0.82rem; margin-top: 18px; line-height: 1.5; }
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid #222226; }
.stTabs [data-baseweb="tab"] { padding: 8px 0; letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.8rem; font-weight: 600; }
div[data-baseweb="select"] > div { background: #121214; border-color: #222226; }
table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
th { text-align: left; color: #8E8E93; font-weight: 600; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; border-bottom: 1px solid #2A2A2E; padding: 8px 6px; }
td { border-bottom: 1px solid #1B1B1E; padding: 9px 6px; }
</style>
"""


@st.cache_resource(show_spinner="Training model on 2022-2025 races")
def load_model():
    laps = features.build_lap_table(data.load_laps())
    model = models.fit_gbt(laps, config.TIERS["pre_lap"], "pre_lap")
    fuel_total_s = models.estimate_fuel_effect(laps)
    totals = laps.groupby("EventName")["TotalLaps"].max().astype(int).to_dict()
    return laps, model, fuel_total_s, totals


def label(text):
    st.markdown(f'<div class="f1-label">{text}</div>', unsafe_allow_html=True)


def chart(curves, y_field, y_title):
    colors = alt.Scale(domain=list(COMPOUND_COLORS), range=list(COMPOUND_COLORS.values()))
    base = alt.Chart(curves).mark_line(strokeWidth=3, interpolate="monotone").encode(x=alt.X("age:Q", title="Tyre age (laps)", scale=alt.Scale(domain=[1, MAX_AGE])), y=alt.Y(f"{y_field}:Q", title=y_title), color=alt.Color("compound:N", scale=colors, legend=alt.Legend(orient="top", title=None, symbolType="stroke", symbolStrokeWidth=4, labelFontSize=13)))
    return base.properties(height=380).configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#8E8E93", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E", labelFontSize=12, titleFontSize=12).configure_legend(labelColor="#F2F2F2")


def card(compound, curves):
    g = curves[curves["compound"] == compound].set_index("age")["loss"]
    color = COMPOUND_COLORS[compound]
    return f'<div class="f1-card" style="--c:{color}"><div class="name">{compound}</div><div class="row"><div><div class="num">+{g.loc[10]:.2f}s</div><div class="cap">10 laps</div></div><div><div class="num">+{g.loc[20]:.2f}s</div><div class="cap">20 laps</div></div><div><div class="num">+{g.loc[30]:.2f}s</div><div class="cap">30 laps</div></div></div></div>'


def predictor_tab(laps, model, fuel_total_s, totals):
    c1, c2, c3 = st.columns([2, 1.3, 1.3])
    with c1:
        label("Circuit")
        event = st.selectbox("Circuit", sorted(totals), index=sorted(totals).index("Bahrain Grand Prix") if "Bahrain Grand Prix" in totals else 0, label_visibility="collapsed")
    with c2:
        label("Track temperature (C)")
        track_temp = st.slider("Track temperature", 15, 55, 35, label_visibility="collapsed")
    with c3:
        label("Air temperature (C)")
        air_temp = st.slider("Air temperature", 10, 40, 25, label_visibility="collapsed")
    c4, c5, c6 = st.columns([2, 1.3, 1.3])
    with c4:
        label("View")
        view = st.radio("View", ["Tyre effect (fuel-corrected)", "Raw pace vs field median"], horizontal=True, label_visibility="collapsed")
    with c5:
        label("Stint starts on lap")
        start_lap = st.slider("Stint start", 2, max(3, totals[event] - MAX_AGE), min(10, max(3, totals[event] - MAX_AGE)), label_visibility="collapsed")
    with c6:
        label("Traffic")
        traffic = st.radio("Traffic", ["Clean air", "Dirty air"], horizontal=True, label_visibility="collapsed")
    curves = models.predict_degradation(model, laps, fuel_total_s, event, track_temp, air_temp, start_lap, traffic == "Dirty air")
    st.write("")
    if view.startswith("Tyre"):
        st.altair_chart(chart(curves, "loss", "Lap-time loss vs fresh tyre (s)"), use_container_width=True, theme=None)
    else:
        st.altair_chart(chart(curves, "pace", "Lap time vs field median (s)"), use_container_width=True, theme=None)
    cols = st.columns(3)
    for col, compound in zip(cols, COMPOUND_COLORS):
        col.markdown(card(compound, curves), unsafe_allow_html=True)
    st.markdown('<div class="f1-note">Average-driver prediction from 2022-2025 races. Compound differences are small next to circuit, fuel and traffic effects, and drivers manage their tyres, so treat the curves as typical behaviour, not a strategy call.</div>', unsafe_allow_html=True)


def model_tab():
    path = config.RESULTS_DIR / "metrics.json"
    if not path.exists():
        st.write("Run evaluate.py to generate results.")
        return
    report = json.load(open(path))
    holdout = report["holdout_2025"]
    rows = [("Median baseline", "median_baseline"), ("Formula (fuel and form)", "formula_b_with_fuel_and_form"), ("Ridge regression", "ridge_pre_lap_features"), ("Boosted trees, pre-lap", "gbt_pre_lap"), ("Persistence, last 3 laps", "persistence_last3_laps"), ("Boosted trees, live", "gbt_live")]
    html = "<table><tr><th>Model</th><th>MAE (s)</th><th>R&sup2;</th><th>Tyre-age curve error (s)</th></tr>"
    for name, key in rows:
        m = holdout[key]
        html += f"<tr><td>{name}</td><td>{m['mae_s']:.3f}</td><td>{m['r2'] if m['r2'] > 0.005 else 0.0:.2f}</td><td>{m['curve_mae_s']:.3f}</td></tr>"
    html += "</table>"
    label("Unseen 2025 races, trained on 2022-2024")
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('<div class="f1-note">Gradient-boosted trees implemented from scratch in NumPy. Pre-lap uses everything known at the start of the lap except the driver\'s own previous lap times; live adds them. Target is lap time relative to the race\'s field-median lap.</div>', unsafe_allow_html=True)
    st.write("")
    label("Predicted vs actual tyre-age curves, unseen 2025 races")
    curves = pd.read_csv(config.RESULTS_DIR / "degradation_curves.csv").melt(id_vars=["compound", "age"], value_vars=["actual", "pre_lap", "live"], var_name="series", value_name="loss")
    series_scale = alt.Scale(domain=["actual", "pre_lap", "live"], range=["#F2F2F2", "#E10600", "#8E8E93"])
    dash_scale = alt.Scale(domain=["actual", "pre_lap", "live"], range=[[1, 0], [1, 0], [4, 3]])
    lines = alt.Chart(curves).mark_line(strokeWidth=2.5).encode(x=alt.X("age:Q", title="Tyre age (laps)"), y=alt.Y("loss:Q", title="Lap-time loss (s)"), color=alt.Color("series:N", scale=series_scale, legend=alt.Legend(orient="top", title=None, symbolType="stroke", symbolStrokeWidth=4)), strokeDash=alt.StrokeDash("series:N", scale=dash_scale, legend=None)).properties(width=262, height=230)
    faceted = lines.facet(column=alt.Column("compound:N", title=None, sort=["Soft", "Medium", "Hard"], header=alt.Header(labelColor="#BDBDC2", labelFontSize=12, labelOrient="top")))
    st.altair_chart(faceted.configure(background="transparent").configure_view(strokeWidth=0).configure_axis(gridColor="#1C1C1F", labelColor="#8E8E93", titleColor="#8E8E93", domainColor="#2A2A2E", tickColor="#2A2A2E").configure_legend(labelColor="#F2F2F2"), use_container_width=True, theme=None)


def main():
    st.set_page_config(page_title="F1 Tyre Degradation", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="f1-bar"></div><div class="f1-title">Tyre Degradation</div><div class="f1-sub">How much slower a stint gets with every lap on the tyre</div>', unsafe_allow_html=True)
    laps, model, fuel_total_s, totals = load_model()
    tab_predict, tab_model = st.tabs(["Predictor", "Model"])
    with tab_predict:
        predictor_tab(laps, model, fuel_total_s, totals)
    with tab_model:
        model_tab()


main()
