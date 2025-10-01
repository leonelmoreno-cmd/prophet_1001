#!/usr/bin/env python
# coding: utf-8

# Análisis de Demanda JC — Google Trends + Prophet
# Alcance fijo: US, últimos 5 años, en-US
# Mejores prácticas: reintentos/retraso, almacenamiento en caché, validación, mensajes de error estructurados.

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_forecast_component
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuración básica de Streamlit
st.set_page_config(page_title="Análisis de Demanda JC", layout="wide")
st.title("Análisis de Demanda JC")
st.caption("Google Trends (US, últimos 5 años, en-US) → Prophet → Plotly → Mejores decisiones")

# Entradas de la interfaz de usuario
kw = st.text_input("Palabra clave (obligatoria para el modo de solicitud)", value="", placeholder="ej. estufa de cohete")

col_req, col_csv = st.columns(2)
request_clicked = col_req.button("Solicitar")
uploaded_file = col_csv.file_uploader("Elegir CSV de Google Trends", type=["csv", "tsv"])
upload_clicked = col_csv.button("Subir CSV")

# Configuración fija para el modo de solicitud
HL = "en-US"            # idioma de la interfaz para Trends
TZ = 360                # parámetro de desfase horario (segundos)
TIMEFRAME = "today 5-y" # últimos 5 años
GEO = "US"              # Estados Unidos

# Función para obtener tendencias de Google
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trends(keyword: str) -> pd.DataFrame:
    """Llamar a Google Trends a través de pytrends y devolver un dataframe limpio."""
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl=HL, tz=TZ, timeout=(10, 25), retries=2, backoff_factor=0.1)
    pytrends.build_payload([keyword], timeframe=TIMEFRAME, geo=GEO)
    df = pytrends.interest_over_time()
    if df.empty:
        return df
    if len(df) > 0:
        df = df.iloc[:-1]
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

# Función para analizar y predecir con Prophet
def run_prophet(df: pd.DataFrame, series_name: str):
    """Aplicar el modelo Prophet para predecir los próximos 6 meses."""
    if df.empty:
        st.warning("No hay datos disponibles después del análisis. Verifique el archivo o la palabra clave.")
        return

    if series_name not in df.columns:
        st.error(f"La columna '{series_name}' no se encuentra en el conjunto de datos.")
        return

    df_prophet = df[[series_name]].reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.add_country_holidays(country_name='US')
    model.fit(df_prophet)

    future = model.make_future_dataframe(df_prophet, periods=26, freq='W')
    forecast = model.predict(future)

    # Gráfica de la predicción
    fig = model.plot(forecast)
    st.plotly_chart(fig)

    # Componentes de la predicción
    st.subheader("Componentes de la predicción")
    fig_components = model.plot_components(forecast)
    st.plotly_chart(fig_components)

    # Descarga de los resultados
    st.download_button(
        label="Descargar resultados",
        data=forecast.to_csv(index=False).encode('utf-8'),
        file_name=f'forecast_{series_name}.csv',
        mime='text/csv'
    )

# Función para analizar datos semanales
def weekly_analysis(df: pd.DataFrame, series_name: str):
    """Análisis de la estacionalidad semanal."""
    df['week'] = df.index.isocalendar().week
    weekly_avg = df.groupby('week')[series_name].mean().reset_index()

    fig = px.line(weekly_avg, x='week', y=series_name, title='Promedio semanal')
    st.plotly_chart(fig)

# Controladores de ejecución
def run_request_mode():
    if not kw.strip():
        st.error("Por favor, ingrese una palabra clave para usar el modo de solicitud.")
        return
    with st.spinner("Obteniendo datos de Google Trends…"):
        try:
            df = fetch_trends(kw.strip())
        except Exception as e:
            st.error(f"Error al obtener datos de Google Trends: {e}")
            return
    if df.empty:
        st.warning("No se devolvieron datos de Google Trends para esta palabra clave.")
        return

    series_name = kw.strip()
    if series_name not in df.columns:
        st.error(f"La columna '{series_name}' no se encuentra en los resultados de Trends.")
        return

    run_prophet(df, series_name)
    weekly_analysis(df, series_name)

def run_upload_mode():
    if uploaded_file is None:
        st.error("Por favor, elija un archivo CSV exportado desde Google Trends.")
        return
    try:
        with st.spinner("Analizando CSV…"):
            file_bytes = uploaded_file.read()
            df_csv = pd.read_csv(io.BytesIO(file_bytes))
            df_csv['date'] = pd.to_datetime(df_csv['date'])
            df_csv.set_index('date', inplace=True)
    except Exception as e:
        st.error(f"Error al analizar el CSV: {e}")
        return

    series_label = df_csv.columns[0]  # Asumimos que solo hay una columna de datos
    run_prophet(df_csv, series_label)
    weekly_analysis(df_csv, series_label)

# Ejecutar según la acción del usuario
if request_clicked:
    run_request_mode()
elif upload_clicked:
    run_upload_mode()

# Pie de página
st.markdown(
    """
    <small>
    Fuente de datos: Google Trends a través de <code>pytrends</code> • Predicción: <code>Prophet</code> • Gráficos: Plotly • Hospedado en Streamlit Community Cloud
    </small>
    """, unsafe_allow_html=True
)
