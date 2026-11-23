"""Homework #4 — US Traffic Accidents Dashboard (Streamlit + Plotly)."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_logic import (
    SEVERITY_LEVELS, filtered_data, key_findings, load_data,
    map_chart, severity_chart, weather_chart, trend_chart, visibility_chart,
)

st.set_page_config(page_title="US Traffic Accidents Dashboard", page_icon="🚗", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/us_accidents_sample.csv.gz"
AUDIT = ROOT / "data/preparation_summary.json"

st.markdown("""<style>
.block-container{padding-top:1.4rem;padding-bottom:2.3rem}
h1,h2,h3{color:#163958}
[data-testid="stMetric"]{background:#f5f9fc;border:1px solid #dde9f0;border-radius:12px;padding:13px}
</style>""", unsafe_allow_html=True)

st.title("US Traffic Accidents Dashboard")
st.caption("Explore reported US accident records by state, severity, time, location, and weather — 2016 to 2023.")
st.info("All counts describe records in the selected **dataset sample**, not all US crashes or accident risk. Reporting coverage, traffic exposure, and sampling affect interpretation.", icon="ℹ️")

if not DATA.is_file():
    st.error("The prepared dataset has not been included yet. Generate the real-data sample first; the app does not use fabricated accident records.")
    st.code("python prepare_data.py --input US_Accidents_March23.csv --fraction 0.02", language="bash")
    st.markdown("This project cleans the complete original Kaggle CSV first, saves the full cleaned file locally, then samples 2% of cleaned records. Commit only `data/us_accidents_sample.csv.gz` and its audit JSON to GitHub.")
    st.stop()

@st.cache_data(show_spinner="Loading prepared accident data...")
def read():
    return load_data(DATA)

try:
    all_data = read()
except Exception as exc:
    st.error(f"Failed to load the prepared dataset: {exc}")
    st.stop()

if all_data.empty:
    st.error("There are no valid records in the prepared dataset.")
    st.stop()

states = sorted(all_data["State"].dropna().unique().tolist())
minimum_year, maximum_year = int(all_data["Year"].min()), int(all_data["Year"].max())

with st.sidebar:
    st.header("Explore the data")
    state_mode = st.selectbox("State selection", ["All states", "Choose states"], index=0)
    chosen_states = st.multiselect("States", options=states, default=[], placeholder="Select one or more states", disabled=(state_mode == "All states"))
    if state_mode == "Choose states" and not chosen_states:
        st.caption("Choose at least one state, or switch to All states.")
    levels = st.multiselect("Severity levels", options=SEVERITY_LEVELS, default=SEVERITY_LEVELS,
                            format_func=lambda k: f"Level {k}")
    if minimum_year < maximum_year:
        year_range = st.slider("Year range", minimum_year, maximum_year, (minimum_year, maximum_year), step=1)
    else:
        year_range = (minimum_year, maximum_year)
        st.caption(f"Year: {minimum_year}")
    st.divider()
    st.caption("Severity measures traffic disruption, not injury or fatality severity. Missing weather values are retained as Unknown.")

if state_mode == "Choose states" and not chosen_states:
    st.warning("Please select at least one state to display results.")
    st.stop()
if not levels:
    st.warning("Please select at least one severity level to display results.")
    st.stop()

selected = filtered_data(all_data, states=chosen_states if state_mode == "Choose states" else None,
                         severities=levels, years=year_range)
if selected.empty:
    st.warning("No sampled accident records match the selected states, severity levels, and years. Try expanding the filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sample records", f"{len(selected):,}")
k2.metric("States represented", f"{selected['State'].nunique():,}")
k3.metric("Severity 3–4 share", f"{selected['Severity'].isin([3,4]).mean():.1%}")
k4.metric("Years included", f"{selected['Year'].nunique():,}")

intro, patterns, methods = st.tabs(["Overview & map", "Weather & time", "Data notes"])
with intro:
    left, right = st.columns([1.36, 1])
    with left:
        fig_map, mapped_count = map_chart(selected)
        st.plotly_chart(fig_map, use_container_width=True, config={"displaylogo": False})
        if mapped_count < len(selected):
            st.caption(f"Map displays a reproducible random subset of {mapped_count:,} out of {len(selected):,} filtered records to keep interactions fast. Other charts use all filtered sample rows.")
        else:
            st.caption(f"Map displays all {mapped_count:,} records matching the filters.")
    with right:
        st.plotly_chart(severity_chart(selected), use_container_width=True, config={"displaylogo": False})
        st.caption("1 = least disruption to traffic; 4 = greatest disruption. These are **not** four injury-severity categories.")
    st.subheader("Key findings in the current selection")
    for finding in key_findings(selected):
        st.markdown(f"- {finding}")

with patterns:
    wc, vc = st.columns([1.1, 1])
    with wc:
        st.plotly_chart(weather_chart(selected), use_container_width=True, config={"displaylogo": False})
        st.caption("Raw record counts by reported condition show association only, not that the weather caused a crash. Missing weather is 'Unknown'.")
    with vc:
        visibility = visibility_chart(selected)
        if visibility is not None:
            st.plotly_chart(visibility, use_container_width=True, config={"displaylogo": False})
            st.caption("Visibility measurements are missing for some reports; only measured values are binned above.")
    st.plotly_chart(trend_chart(selected, *year_range), use_container_width=True, config={"displaylogo": False})
    st.caption("Monthly counts of sampled reports are plotted only between the first and last observed months of the current selection. Any internal zero denotes no sampled reports, not zero actual crashes. The source begins February 2016 and ends March 2023; reporting coverage varies over time.")

with methods:
    st.subheader("Data origin and cleaning")
    st.markdown("Source: [US-Accidents: A Countrywide Traffic Accident Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/) by Sobhan Moosavi and collaborators, through March 2023. We **clean the complete source, save the complete cleaned CSV locally, then select a reproducible 2% sample** for the hosted app.")
    info = {}
    if AUDIT.is_file():
        try:
            info = json.loads(AUDIT.read_text(encoding="utf-8"))
            st.write("Preparation order: **Clean all raw records → save full cleaned CSV → sample cleaned records**")
            st.write(f"Original records scanned: {info['cleaning']['raw_rows_inspected']:,}")
            st.write(f"Excluded for invalid essential fields (entire source): {info['cleaning']['removed_missing_or_invalid_required']:,}")
            st.write(f"Duplicate IDs removed (entire source): {info['cleaning']['duplicate_ids_removed_from_full_source']:,}")
            st.write(f"Full cleaned dataset: {info['cleaning']['cleaned_rows']:,} records (saved locally, not hosted)")
            st.write(f"Sampled from cleaned dataset: {info['sampling']['selected_from_cleaned']:,} records ({info['sampling']['fraction_observed_of_cleaned']:.2%})")
            st.write(f"Unknown weather labels in full cleaned data: {info['cleaning']['weather_unknown_after_cleaning']:,}")
            st.download_button("Download preparation audit (JSON)", data=json.dumps(info, indent=2), file_name="preparation_summary.json", mime="application/json")
        except (KeyError, ValueError, TypeError) as exc:
            st.caption(f"Preparation audit could not be read: {exc}")
    else:
        st.caption("Preparation audit unavailable; run prepare_data.py to generate data/preparation_summary.json.")
    if AUDIT.is_file() and "distribution_comparison" in info.get("sampling", {}):
        with st.expander("Sample vs. complete cleaned dataset checks"):
            for dimension, details in info["sampling"]["distribution_comparison"].items():
                st.write(f"{dimension}: largest marginal share difference = {details['max_absolute_percentage_point_difference']:.3f} percentage points")
            st.caption(info["sampling"].get("distribution_comparison_note", ""))
    st.markdown("""**Limitations:** Raw counts are not per-mile-driven or per-driver rates. Coverage changes over time, varies by state and provider, and is affected by source reporting. The sample is deliberately reduced to support responsive hosting; findings should not be generalized to national crash incidence. Weather conditions are observational and do not establish causality.""")
    with st.expander("Inspect a small selection of records"):
        st.dataframe(selected.head(200), use_container_width=True, hide_index=True)

st.divider()
st.caption("Educational data explorer · Streamlit + Plotly · Source credit: Moosavi et al., US-Accidents (2016–2023).")
