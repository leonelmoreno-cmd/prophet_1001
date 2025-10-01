#!/usr/bin/env python
# coding: utf-8

# Demand Analysis JC — Google Trends + Prophet Forecasting
# Fixed scope: US, last 5 years, en-US
# Best practices: retries/backoff, caching, validation, structured error messages.

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from pytrends.request import TrendReq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta

# ---------- Streamlit basic setup ----------
st.set_page_config(page_title="Demand Analysis JC", layout="wide")
st.title("Demand Analysis JC")
st.caption("Google Trends (US, last 5y, en-US) → Prophet Forecasting → Plotly → Better decisions")

# ---------- UI inputs ----------
kw = st.text_input("Keyword (required for Request mode)", value="", placeholder="e.g., rocket stove")

# Two explicit actions as requested
col_req, col_csv = st.columns(2)
request_clicked = col_req.button("Request")
uploaded_file = col_csv.file_uploader("Choose Google Trends CSV", type=["csv", "tsv"])
upload_clicked = col_csv.button("Upload CSV")

# Fixed config for Request mode
HL = "en-US"            # interface language for Trends
TZ = 360                # timezone offset param (per pytrends examples)
TIMEFRAME = "today 5-y" # last 5 years
GEO = "US"              # United States

# ---------- Helpers ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trends(keyword: str) -> pd.DataFrame:
    """Call Google Trends via pytrends and return a cleaned dataframe."""
    pytrends = TrendReq(
        hl=HL,
        tz=TZ,
        timeout=(10, 25),     # connect/read
        retries=2,
        backoff_factor=0.1,   # exponential backoff
    )
    pytrends.build_payload([keyword], timeframe=TIMEFRAME, geo=GEO)
    df = pytrends.interest_over_time()
    if df.empty:
        return df
    # Clean: drop last row and 'isPartial' per your workflow
    if len(df) > 0:
        df = df.iloc[:-1]
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def build_figure(df_plot: pd.DataFrame, title_kw: str) -> go.Figure:
    """Build 4-panel Plotly figure."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual")
    )
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["original"], name="Original", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["trend"],   name="Trend",   mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["seasonal"],name="Seasonal",mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["remainder"],name="Residual",mode="lines"), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    fig.update_layout(height=900, title_text=f"STL Decomposition — {title_kw} — Google Trends (US, last 5y)")
    return fig

def _clean_keyword_label(raw: str) -> str:
    """Normalize series label like 'beef tallow for skin: (Estados Unidos)' -> 'beef tallow for skin (United States/Estados Unidos)' without trailing colon artifacts."""
    label = raw.strip()
    # Remove duplicate spaces and trailing colon
    label = re.sub(r":\s*$", "", label)
    # Replace ':\s*(' with ' ('
    label = re.sub(r":\s*\(", " (", label)
    return label

def parse_trends_csv(file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parse a Google Trends CSV (en or es).
    Returns (df, series_label) where df has a DatetimeIndex and a single numeric column.
    """
    text = file_bytes.decode("utf-8-sig", errors="replace")
    
    # Debugging: Show first few lines of the uploaded file to help understand the format
    st.write("First 5 lines of CSV:")
    st.write(text.splitlines()[:5])

    # Find the header line by scanning until we see a row whose first cell is Week/Semana/Date/Fecha
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    header_idx = -1
    for i, ln in enumerate(lines):
        # Try commas first; if only one col, try semicolon and tab
        for sep in [",", ";", "\t"]:
            parts = [p.strip() for p in ln.split(sep)]
            if len(parts) >= 2:
                # Look for column names that might be related to dates
                possible_date_columns = ["week", "semana", "date", "fecha"]
                if any(p.lower() in possible_date_columns for p in parts):
                    header_idx = i
                    delimiter = sep
                    break
        if header_idx != -1:
            break

    if header_idx == -1:
        raise ValueError("Could not locate a header row (expected columns like Week/Semana/Date/Fecha).")

    # Rebuild the CSV from header onward and let pandas parse
    content = "\n".join(lines[header_idx:])
    df_raw = pd.read_csv(io.StringIO(content), sep=None, engine="python")

    # Debugging: Show the columns of the CSV after parsing
    st.write("Columns after parsing CSV:")
    st.write(df_raw.columns)

    # Identify date column name (en/es)
    date_candidates = [c for c in df_raw.columns if c.strip().lower() in {"week", "semana", "date", "fecha"}]
    if not date_candidates:
        raise ValueError("Date column not found (looking for Week/Semana/Date/Fecha).")
    date_col = date_candidates[0]

    # Identify the keyword series column: the one that isn't the date
    value_cols = [c for c in df_raw.columns if c != date_col]
    if not value_cols:
        raise ValueError("Value column not found (expected a keyword like 'beef tallow for skin: (United States)').")

    # If multiple columns exist, take the first non-empty numeric-like column
    chosen = None
    for col in value_cols:
        non_na = pd.to_numeric(df_raw[col], errors="coerce")
        if non_na.notna().sum() > 0:
            chosen = col
            break
    if chosen is None:
        # Fallback: take the first value column
        chosen = value_cols[0]

    series_label = _clean_keyword_label(str(chosen))

    # Build normalized dataframe
    df = pd.DataFrame({
        "date": pd.to_datetime(df_raw[date_col], errors="coerce"),
        series_label: pd.to_numeric(df_raw[chosen], errors="coerce"),
    }).dropna(subset=["date"]).sort_values("date")

    # Some exports may contain future/partial last rows; drop trailing NaN or duplicates
    df = df.dropna(subset=[series_label])

    # Use date as index to match fetch_trends() shape
    df = df.set_index("date")

    # Debugging: Show the dataframe after cleaning
    st.write("Cleaned DataFrame:")
    st.write(df.head())

    return df, series_label

def run_prophet_forecast(df: pd.DataFrame, series_name: str):
    """Run Prophet forecasting for the next 6 months."""
    if df.empty:
        st.warning("No data available after parsing. Please verify the file or keyword.")
        st.stop()

    if series_name not in df.columns:
        st.error(f"Column '{series_name}' not found in the dataset.")
        st.stop()

    df_prophet = df.reset_index()[['date', series_name]]
    df_prophet.columns = ['ds', 'y']

    # Instantiate Prophet model
    model = Prophet(weekly_seasonality=True)
    model.add_country_holidays(country_name='US')  # Add built-in holidays for the US
    
    # Fit the model
    model.fit(df_prophet)

    # Make a future dataframe for 6 months
    future = model.make_future_dataframe(df_prophet, periods=26, freq='W')

    # Forecast
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the forecast components
    fig2 = model.plot_components(forecast)
    st.plotly_chart(fig2, use_container_width=True)

    # Display forecasted data
    st.markdown(f"### Forecast for the next 6 months")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], use_container_width=True)

# ---------- Execution controllers ----------
def run_request_mode():
    if not kw.strip():
        st.error("Please enter a keyword to use Request mode.")
        st.stop()
    with st.spinner("Fetching Google Trends…"):
        try:
            df = fetch_trends(kw.strip())
        except Exception as e:
            st.error(f"Error fetching data from Google Trends: {e}")
            st.info("Tip: pin urllib3<2 and try again if you see 'method_whitelist' errors.")
            st.stop()
    if df.empty:
        st.warning("No data returned by Google Trends for this keyword/timeframe/geo.")
        st.stop()

    # Series selection follows the exact column typed by user
    col_name = kw.strip()
    if col_name not in df.columns:
        st.error(f"Column '{col_name}' not found in Trends result.")
        st.stop()

    run_prophet_forecast(df, col_name)

def run_upload_mode():
    if uploaded_file is None:
        st.error("Please choose a CSV file exported from Google Trends.")
        st.stop()
    try:
        with st.spinner("Parsing CSV…"):
            file_bytes = uploaded_file.read()
            df_csv, series_label = parse_trends_csv(file_bytes)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    # For display, keep the label from the CSV header (keyword + country)
    run_prophet_forecast(df_csv, series_label)

# Trigger actions
if request_clicked:
    run_request_mode()
elif upload_clicked:
    run_upload_mode()

# ---------- Footer ----------
st.markdown(
    """
    <small>
    Data source: Google Trends via <code>pytrends</code> • Forecasting: <code>Prophet</code> • Charts: Plotly • Host: Streamlit Community Cloud
    </small>
    """, unsafe_allow_html=True
)
