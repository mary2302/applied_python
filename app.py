# app.py
import os
import asyncio
import concurrent.futures
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import aiohttp

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Ваши модули (с запасным вариантом, чтобы приложение не падало при импорте) ---
try:
    from weather_req import season_from_month
except Exception:
    def season_from_month(m: int) -> str:
        # Простая мапа: Северное полушарие
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"

try:
    from parallel_EDA import default_pd
except Exception:
    # Минимальный фоллбек (если import не доступен)
    def default_pd(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
        df["ma30"] = df.groupby("city")["temperature"].transform(
            lambda s: s.rolling(window=30, min_periods=1).mean()
        )
        season_statistics = (
            df.groupby(["city", "season"])["temperature"]
              .agg(season_mean="mean", season_std="std", n="count")
              .reset_index()
        )
        df = df.merge(season_statistics, on=["city", "season"], how="left")
        df["lower_bound"] = df["season_mean"] - 2 * df["season_std"]
        df["upper_bound"] = df["season_mean"] + 2 * df["season_std"]
        df["is_anomaly"] = (df["temperature"] < df["lower_bound"]) | (df["temperature"] > df["upper_bound"])
        return df


st.set_page_config(page_title="Real-time weather monitoring and analysis", layout="wide")


# ---------------------------
# Data loading / caching
# ---------------------------
@st.cache_data(show_spinner=False)
def load_history_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df_raw = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    return default_pd(df_raw)

@st.cache_data(show_spinner=False)
def load_history_from_path(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)
    return default_pd(df_raw)


# ---------------------------
# OpenWeatherMap async fetch
# ---------------------------
async def fetch_weather_async(city: str, key: str) -> tuple[float | None, dict]:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
    try:
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        connector = aiohttp.TCPConnector(limit=10, force_close=True)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.get(url) as response:
                try:
                    data = await response.json()
                except Exception as e:
                    return None, {"cod": response.status, "message": f"JSON parse error: {e}"}

                if str(data.get("cod")) != "200":
                    return None, data

                try:
                    t = float(data["main"]["temp"])
                    return t, data
                except (KeyError, TypeError, ValueError) as e:
                    return None, {"cod": data.get("cod"), "message": f"Invalid data: {e}"}

    except asyncio.TimeoutError as e:
        return None, {"cod": None, "message": f"Timeout error: {e}"}
    except aiohttp.ClientError as e:
        return None, {"cod": None, "message": f"Network error: {e}"}
    except Exception as e:
        return None, {"cod": None, "message": f"Unexpected error: {e}"}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_weather_wrapper(city: str, key: str):
    """
    Безопасный запуск async-кода в Streamlit.
    Если event loop уже существует, запускаем asyncio.run в отдельном потоке.
    """
    async def _async_fetch():
        return await fetch_weather_async(city, key)

    try:
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(_async_fetch()))
                return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(_async_fetch())
    except Exception as e:
        return None, {"cod": None, "message": str(e)}


# ---------------------------
# Monthly analysis helpers
# ---------------------------
def monthly_series(df_city_raw: pd.DataFrame) -> pd.DataFrame:
    if df_city_raw.empty:
        return pd.DataFrame()

    d = df_city_raw.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"])
    d["month"] = d["timestamp"].dt.to_period("M").dt.to_timestamp()

    m = (
        d.groupby("month", as_index=False)["temperature"]
         .mean()
         .rename(columns={"temperature": "temp_month_mean"})
         .sort_values("month")
    )

    m["trend_12m"] = m["temp_month_mean"].rolling(window=12, min_periods=1).mean()
    return m


def plot_single_city_monthly(df_city_raw: pd.DataFrame, city: str):
    m = monthly_series(df_city_raw)
    if m.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["month"], y=m["temp_month_mean"],
        mode="lines+markers",
        name="Monthly mean",
        hovertemplate="<b>%{x|%Y-%m}</b><br>Mean: %{y:.2f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=m["month"], y=m["trend_12m"],
        mode="lines",
        name="12m trend",
        hovertemplate="<b>%{x|%Y-%m}</b><br>Trend: %{y:.2f}°C<extra></extra>"
    ))

    fig.update_layout(
        title=f"{city}: Monthly mean and 12-month trend",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=420,
        hovermode="x unified"
    )
    return fig


# ---------------------------
# UI
# ---------------------------
st.title("🌦️ Real-time weather monitoring and analysis")

with st.sidebar:
    st.header("Данные")
    uploaded = st.file_uploader("Загрузите temperature_data.csv (опционально)", type=["csv"])
    use_local = st.checkbox("Использовать локальный temperature_data.csv", value=(uploaded is None))

    st.header("API")
    api_key = st.text_input(
        "Введите OpenWeatherMap API ключ для получения актуальной температуры",
        type="password",
        placeholder="API ключ"
    )

# Load data
if uploaded is not None:
    df_all = load_history_from_bytes(uploaded.getvalue())
elif use_local:
    df_all = load_history_from_path("temperature_data.csv")
else:
    st.stop()

if df_all.empty or "city" not in df_all.columns:
    st.error("Данные не загрузились или формат не соответствует ожидаемому (нужны колонки city, timestamp, temperature, season).")
    st.stop()

# City selector
cities = sorted(df_all["city"].dropna().unique().tolist())
city = st.selectbox("Выберите город", cities, index=0)

df_city = df_all[df_all["city"] == city].copy()
df_city["timestamp"] = pd.to_datetime(df_city["timestamp"], errors="coerce")
df_city = df_city.dropna(subset=["timestamp"]).sort_values("timestamp")

anom = df_city[df_city.get("is_anomaly", False) == True].copy()

# Layout columns
left, right = st.columns([1, 1])

# ---------------------------
# Season profile (right)
# ---------------------------
with right:
    st.subheader("Сезонные профили (интерактивные)")
    if not df_city.empty:
        season_profile = (
            df_city.groupby("season", as_index=False)["temperature"]
                  .agg(season_mean="mean", season_std="std", n="count")
        )

        # NaN std -> 0 (если n=1)
        season_profile["season_std"] = season_profile["season_std"].fillna(0.0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=season_profile["season"],
            y=season_profile["season_mean"],
            error_y=dict(
                type="data",
                array=season_profile["season_std"],
                visible=True,
                thickness=1.5,
                width=3
            ),
            name="Mean ± Std",
            hovertemplate=(
                "<b>Сезон:</b> %{x}<br>"
                "<b>Среднее:</b> %{y:.2f}°C<br>"
                "<b>Std:</b> %{customdata[0]:.2f}°C<br>"
                "<b>N:</b> %{customdata[1]}<extra></extra>"
            ),
            customdata=np.c_[season_profile["season_std"].values, season_profile["n"].values],
        ))
        fig.update_layout(
            title=f"{city}: Сезонные средние и стандартные отклонения",
            xaxis_title="Сезон",
            yaxis_title="Температура (°C)",
            hovermode="x unified",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных по выбранному городу.")


# ---------------------------
# Time series with anomalies (left)
# ---------------------------
with left:
    st.subheader("Временной ряд температур (интерактивный)")
    if not df_city.empty:
        fig2 = make_subplots(specs=[[{"secondary_y": False}]])

        fig2.add_trace(
            go.Scatter(
                x=df_city["timestamp"],
                y=df_city["temperature"],
                mode="lines",
                name="Daily Temperature",
                hovertemplate=(
                    "<b>Дата:</b> %{x|%Y-%m-%d}<br>"
                    "<b>Температура:</b> %{y:.2f}°C<br>"
                    "<b>Сезон:</b> %{customdata}<extra></extra>"
                ),
                customdata=df_city["season"],
            ),
            secondary_y=False
        )

        if "ma30" in df_city.columns:
            fig2.add_trace(
                go.Scatter(
                    x=df_city["timestamp"],
                    y=df_city["ma30"],
                    mode="lines",
                    name="MA30",
                    hovertemplate="<b>MA30:</b> %{y:.2f}°C<extra></extra>",
                ),
                secondary_y=False
            )

        if not anom.empty:
            fig2.add_trace(
                go.Scatter(
                    x=anom["timestamp"],
                    y=anom["temperature"],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(size=8, symbol="x"),
                    hovertemplate=(
                        "<b>АНОМАЛИЯ</b><br>"
                        "<b>Дата:</b> %{x|%Y-%m-%d}<br>"
                        "<b>Температура:</b> %{y:.2f}°C<br>"
                        "<b>Сезон:</b> %{customdata}<extra></extra>"
                    ),
                    customdata=anom["season"],
                ),
                secondary_y=False
            )

        # Seasonal normal range (±2σ)
        if "season_mean" in df_city.columns and "season_std" in df_city.columns:
            std = df_city["season_std"].fillna(0.0)
            upper = df_city["season_mean"] + 2 * std
            lower = df_city["season_mean"] - 2 * std

            fig2.add_trace(
                go.Scatter(
                    x=df_city["timestamp"].tolist() + df_city["timestamp"].tolist()[::-1],
                    y=upper.tolist() + lower.tolist()[::-1],
                    fill="toself",
                    hoverinfo="skip",
                    name="Seasonal Normal Range (±2σ)",
                    line=dict(width=0),
                    showlegend=True,
                ),
                secondary_y=False
            )

        fig2.update_layout(
            title=f"{city}: Daily Temperature with Anomalies",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        fig2.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
        )

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Нет данных по выбранному городу.")


# ---------------------------
# Monthly analysis
# ---------------------------
st.markdown("---")
st.header("📅 Месячный анализ температур")

monthly_fig = plot_single_city_monthly(df_city, city)
if monthly_fig is not None:
    st.plotly_chart(monthly_fig, use_container_width=True)

    with st.expander("📈 Анализ трендов"):
        monthly_data = monthly_series(df_city)
        if not monthly_data.empty:
            latest = monthly_data.iloc[-1]
            st.metric(
                label="Текущий 12-месячный тренд",
                value=f"{latest['trend_12m']:.2f}°C",
                delta=f"{(latest['temp_month_mean'] - latest['trend_12m']):.2f}°C от месячного значения"
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Макс. месячная", f"{monthly_data['temp_month_mean'].max():.1f}°C")
            with c2:
                st.metric("Мин. месячная", f"{monthly_data['temp_month_mean'].min():.1f}°C")
            with c3:
                st.metric("Ср. тренд", f"{monthly_data['trend_12m'].mean():.1f}°C")
else:
    st.info("Недостаточно данных для месячного анализа.")


# ---------------------------
# Current temperature analysis (API)
# ---------------------------
st.markdown("---")
st.subheader("Анализ текущей температуры")

if not api_key:
    st.info("Для получения текущей температуры введите ваш API ключ.")
else:
    current_season = season_from_month(datetime.utcnow().month)
    temp, raw = fetch_weather_wrapper(city, api_key)

    if temp is None:
        st.error("Ошибка при обращении к API")
        st.json(raw)
    else:
        st.write(f"Выбранный город: **{city}**")
        st.write(f"Текущая температура: **{temp:.2f} °C**")
        st.write(f"Время года (по UTC): **{current_season}**")

        # Берём сезонные статистики для текущего сезона
        row = df_city.loc[df_city["season"] == current_season, ["season_mean", "season_std"]].head(1)

        if row.empty:
            st.warning("В исторических данных нет статистики для текущего сезона. Невозможно оценить аномальность.")
        else:
            mean = float(row.iloc[0]["season_mean"])
            std = float(row.iloc[0]["season_std"]) if pd.notna(row.iloc[0]["season_std"]) else 0.0

            lower_bound = mean - 2 * std
            upper_bound = mean + 2 * std
            is_anom = (temp < lower_bound) or (temp > upper_bound)

            st.write("Норма для данного времени года:")
            st.write(f"Средняя температура: **{mean:.2f} °C**")
            st.write(f"Std температуры: **{std:.2f} °C**")
            st.write(f"Нормальный диапазон (±2σ): **[{lower_bound:.2f}, {upper_bound:.2f}]**")

            # ВАЖНО: без тернарника (Streamlit иногда ломает AST на таких выражениях)
            if is_anom:
                st.error("Текущая температура **аномальная** (вне mean ± 2σ).")
            else:
                st.success("Текущая температура **нормальная** (в пределах mean ± 2σ).")

        with st.expander("Сырые данные ответа OpenWeatherMap"):
            st.json(raw)
