import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

import pandas as pd
import streamlit as st

from lib.score_helper import (
    DEFAULT_LIKERT_MAP,
    compute_mean_scores,
    load_data,
    map_likert_to_numeric,
    plot_mean_scores,
    select_question_columns,
)

st.set_page_config(page_title="Delphi Analysis 2", page_icon="📈")

st.title("📈 Delphi Analysis 2 — Mean Agreement Scores")
st.caption(
    "Upload panelist rating data (CSV or Excel) to map Likert-scale answers "
    "to numbers and view the mean agreement score per question."
)

uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df_raw = load_data(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df_raw.head())
st.caption(f"{len(df_raw)} rows x {len(df_raw.columns)} columns")

st.subheader("Select columns")
columns = list(df_raw.columns)
question_guess = select_question_columns(df_raw)
question_cols = st.multiselect(
    "Question columns to score",
    columns,
    default=question_guess or columns,
)
if not question_cols:
    st.info("Select at least one column to compute scores.")
    st.stop()

st.sidebar.header("Likert scale")
use_default_map = st.sidebar.checkbox(
    "Map text answers (Strongly disagree...Strongly agree) to 1-5",
    value=True,
)
likert_map = DEFAULT_LIKERT_MAP if use_default_map else {}
if use_default_map:
    st.sidebar.table(pd.Series(likert_map, name="Value"))
else:
    st.sidebar.caption(
        "Values will be parsed as numbers as-is; non-numeric text becomes missing."
    )

default_lo = min(likert_map.values()) if likert_map else 1
default_hi = max(likert_map.values()) if likert_map else 5
scale_min = st.sidebar.number_input("Scale minimum", value=float(default_lo))
scale_max = st.sidebar.number_input("Scale maximum", value=float(default_hi))

df_num = map_likert_to_numeric(df_raw[question_cols], likert_map)
mean_scores = compute_mean_scores(df_num).dropna()

if mean_scores.empty:
    st.warning("No numeric scores could be computed from the selected columns.")
    st.stop()

st.subheader("Mean scores")
st.dataframe(mean_scores.rename("Mean score").to_frame())

st.subheader("Plot")
fig = plot_mean_scores(mean_scores, scale_min=scale_min, scale_max=scale_max)
st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "Download mean scores (CSV)",
    mean_scores.to_csv().encode("utf-8"),
    file_name="mean_scores.csv",
    mime="text/csv",
)
