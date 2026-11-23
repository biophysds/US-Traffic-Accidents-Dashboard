"""Pure, testable data/plotting routines shared by the Streamlit dashboard."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

REQUIRED = {"State", "Severity", "Start_Time", "Start_Lat", "Start_Lng", "Weather_Condition"}
SEVERITY_LEVELS = [1, 2, 3, 4]
PLOT_BG = "rgba(0,0,0,0)"
PALETTE = {1: "#2c9e89", 2: "#2e70ad", 3: "#e5a13d", 4: "#c64b56"}


def load_data(path):
    path = Path(path)
    df = pd.read_csv(path, compression="infer", dtype={"State": "string", "Severity": "Int64"}, low_memory=False)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Processed data missing: {missing}")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time", "State", "Severity", "Start_Lat", "Start_Lng"])
    df["Severity"] = df["Severity"].astype(int)
    df["Year"] = df["Start_Time"].dt.year
    df["Month_Start"] = df["Start_Time"].dt.to_period("M").dt.to_timestamp()
    df["Weather_Condition"] = df["Weather_Condition"].fillna("Unknown")
    return df


def filtered_data(df, states=None, severities=None, years=None):
    mask = df["Severity"].isin(severities if severities is not None else SEVERITY_LEVELS)
    if states:
        mask &= df["State"].isin(states)
    if years is not None:
        low, high = years
        mask &= df["Year"].between(low, high)
    return df.loc[mask].copy()


def base_layout(fig, height=350):
    fig.update_layout(
        height=height, margin=dict(l=12, r=12, t=56, b=20),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Arial", size=12, color="#243349"),
        title_font=dict(size=18), hoverlabel=dict(font_size=12),
        legend_title_text="", showlegend=True,
    )
    return fig


def map_chart(df, max_points=5000):
    sampled = df.sample(n=min(len(df), max_points), random_state=42).copy() if len(df) > max_points else df.copy()
    sampled["Severity level"] = sampled["Severity"].map(lambda v: f"Level {int(v)}")
    fig = px.scatter_map(
        sampled, lat="Start_Lat", lon="Start_Lng", color="Severity level",
        color_discrete_map={f"Level {k}": v for k, v in PALETTE.items()},
        category_orders={"Severity level": [f"Level {k}" for k in SEVERITY_LEVELS]},
        hover_data={"State": True, "Start_Time": True, "Weather_Condition": True,
                    "Start_Lat": ":.3f", "Start_Lng": ":.3f", "Severity": True},
        title="Recorded accident locations", zoom=2.6,
        center={"lat": 39.6, "lon": -96.8},
        map_style="carto-positron", opacity=0.66,
    )
    return base_layout(fig, height=515), len(sampled)


def severity_chart(df):
    counts = df["Severity"].value_counts().reindex(SEVERITY_LEVELS, fill_value=0)
    values = pd.DataFrame({"Severity": [str(v) for v in SEVERITY_LEVELS], "Records": counts.values})
    fig = px.bar(values, x="Severity", y="Records", color="Severity",
                 color_discrete_map={str(k): v for k, v in PALETTE.items()},
                 text="Records", title="Severity distribution (sample records)")
    fig.update_traces(textposition="outside", hovertemplate="Severity %{x}<br>%{y:,} records<extra></extra>")
    fig.update_yaxes(title_text="Number of sampled records", rangemode="tozero")
    fig.update_xaxes(title_text="Severity level (1 = least traffic impact; 4 = greatest)")
    fig = base_layout(fig)
    fig.update_layout(showlegend=False)
    return fig


def weather_chart(df, top_n=10):
    weather = df["Weather_Condition"].fillna("Unknown").astype(str).replace("", "Unknown")
    counts = weather.value_counts().head(top_n)
    plot = pd.DataFrame({"Weather": counts.index, "Records": counts.values}).sort_values("Records")
    fig = px.bar(plot, x="Records", y="Weather", orientation="h", text="Records",
                 title="Most frequently reported weather conditions")
    fig.update_traces(marker_color="#347ca9", textposition="outside",
                      hovertemplate="%{y}<br>%{x:,} sampled records<extra></extra>")
    fig.update_xaxes(title_text="Number of sampled records", rangemode="tozero")
    fig.update_yaxes(title_text="Reported weather category")
    return base_layout(fig, height=400)


def monthly_trend_table(df, first_year=None, last_year=None):
    """Count observed months, without padding beyond the real source coverage.

    Years restrict the requested selection; a missing month *between* the first
    and last observed month is zero sampled reports, not zero actual crashes.
    """
    if df.empty:
        return pd.DataFrame(columns=["Month_Start", "Records"])
    valid = df.copy()
    if first_year is not None:
        valid = valid.loc[valid["Year"] >= int(first_year)]
    if last_year is not None:
        valid = valid.loc[valid["Year"] <= int(last_year)]
    if valid.empty:
        return pd.DataFrame(columns=["Month_Start", "Records"])

    by_month = valid.groupby("Month_Start").size()
    # Do not prepend January 2016 or append April–December 2023: those months
    # fall outside the source's observed data range in the full-year selection.
    first_observed = valid["Month_Start"].min()
    last_observed = valid["Month_Start"].max()
    months = pd.date_range(first_observed, last_observed, freq="MS")
    return (by_month.reindex(months, fill_value=0)
            .rename_axis("Month_Start")
            .rename("Records")
            .reset_index())


def trend_chart(df, first_year=None, last_year=None):
    table = monthly_trend_table(df, first_year, last_year)
    if table.empty:
        return base_layout(go.Figure())
    fig = px.line(table, x="Month_Start", y="Records", markers=False,
                  title="Monthly accident records in the sample")
    fig.update_traces(line_color="#24769c", line_width=2.7,
                      hovertemplate="%{x|%B %Y}: %{y:,} records<extra></extra>")
    fig.update_xaxes(title_text="Month / year")
    fig.update_yaxes(title_text="Number of sampled records", rangemode="tozero")
    return base_layout(fig, height=375)


def visibility_chart(df):
    if "Visibility(mi)" not in df.columns:
        return None
    visibility = pd.to_numeric(df["Visibility(mi)"], errors="coerce")
    intervals = pd.cut(visibility, bins=[-np.inf, 0.5, 2, 5, 10, np.inf],
                       labels=["≤ 0.5", "0.5–2", "2–5", "5–10", "> 10"])
    counts = intervals.value_counts(sort=False)
    values = pd.DataFrame({"Visibility": counts.index.astype(str), "Records": counts.values})
    fig = px.bar(values, x="Visibility", y="Records", text="Records",
                 title="Recorded visibility at accident locations")
    fig.update_traces(marker_color="#287f8a", textposition="outside")
    fig.update_xaxes(title_text="Visibility in miles (binned)")
    fig.update_yaxes(title_text="Sampled records with a visibility value")
    return base_layout(fig)


def key_findings(df):
    if len(df) == 0:
        return ["No records match the current selection."]
    findings = []
    n = len(df)
    most_common_severity = df["Severity"].value_counts().idxmax()
    sev_count = int((df["Severity"] == most_common_severity).sum())
    findings.append(f"Severity {most_common_severity} is the most common level in this selection ({sev_count:,} of {n:,} sampled records; {sev_count / n:.1%}).")
    non_unknown = df["Weather_Condition"].dropna().astype(str)
    non_unknown = non_unknown[~non_unknown.str.strip().str.lower().isin(["unknown", "", "nan", "none"])]
    if len(non_unknown):
        top_weather = non_unknown.value_counts()
        findings.append(f"The most frequently recorded known weather condition is {top_weather.index[0]} ({int(top_weather.iloc[0]):,} of {len(non_unknown):,} records with a known condition).")
    monthly = df.groupby("Month_Start").size()
    if len(monthly):
        peak = monthly.idxmax()
        findings.append(f"The highest observed month within the current sample and filters is {peak:%B %Y}, with {int(monthly.max()):,} records; this is not an exposure-adjusted risk ranking.")
    return findings
