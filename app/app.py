"""Fleet Maintenance Planner dashboard.

Run from the project root:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_split  # noqa: E402
from src.model import MODEL_PATH, load_bundle, predict  # noqa: E402
from src.planning import CRITICAL, WARNING, fleet_status, smooth_predictions  # noqa: E402

st.set_page_config(page_title="Fleet Maintenance Planner", page_icon="✈️", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_predictions(smoothing: int):
    pred = predict(get_bundle(), load_split("test"))
    return smooth_predictions(pred, smoothing)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` from the project root first.")
    st.stop()

# --- Sidebar: planning settings ---
st.sidebar.header("Planning settings")
margin = st.sidebar.slider("Safety margin (cycles)", 0, 40, 10,
                           help="Plan maintenance this many cycles before the predicted failure.")
critical = st.sidebar.slider("Critical threshold (cycles)", 5, 60, 20)
warning = st.sidebar.slider("Plan-soon threshold (cycles)", critical + 5, 120, 50)
smoothing = st.sidebar.slider("Prediction smoothing (cycles)", 1, 20, 5,
                              help="Average the predictions over the last cycles to reduce noise.")

pred = get_predictions(smoothing)
fleet = fleet_status(pred, margin, critical, warning)

# --- Fleet overview ---
st.title("✈️ Fleet Maintenance Planner")
st.caption("Remaining useful life predictions for the FD001 test fleet (NASA C-MAPSS).")

c1, c2, c3 = st.columns(3)
c1.metric("Engines in fleet", len(fleet))
c2.metric("Critical", int((fleet["Status"] == CRITICAL).sum()))
c3.metric("Plan soon", int((fleet["Status"] == WARNING).sum()))

st.subheader("Maintenance priorities")
show_all = st.checkbox("Show all engines", value=False)
table = fleet if show_all else fleet[fleet["Status"].isin([CRITICAL, WARNING])]
st.dataframe(table, hide_index=True)

# --- Engine detail ---
st.subheader("Engine detail")
engine = st.selectbox("Engine", fleet["Engine"], format_func=lambda u: f"Engine {u}")
d = pred[pred["unit"] == engine].set_index("cycle")
row = fleet[fleet["Engine"] == engine].iloc[0]

e1, e2, e3 = st.columns(3)
e1.metric("Cycles flown", int(row["Cycles flown"]))
e2.metric("Predicted RUL", int(row["Predicted RUL"]))
e3.metric("Status", row["Status"])

st.markdown("**Predicted RUL over time**")
st.line_chart(d[["predicted_RUL", "smoothed_RUL"]], x_label="Cycle", y_label="RUL (cycles)")

bundle = get_bundle()
sensor = st.selectbox("Sensor", bundle["sensors"])
st.line_chart(d[[sensor, f"{sensor}_mean"]], x_label="Cycle", y_label="Sensor value")
