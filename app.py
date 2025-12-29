import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import aiohttp
import asyncio
import time

from weather_req import season_from_month
from parallel_EDA import default_pd

st.set_page_config(page_title="Temperature EDA + OpenWeatherMap", layout="wide")

@st.cache_data(ttl=60, show_spinner=False)  # кеш на 60 секунд
def fetch_current_temp_sync_wrapper(city: str, key: str):
    # запускаем async внутри sync-кода streamlit
    return asyncio.run(fetch_current_temp(city, key))

#Кэшируем загруженные исторические данные
@st.cache_data(show_spinner=False)
def load_history(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    history_data = default_pd(df)
    return history_data

async def fetch_current_temp(city, key) -> tuple[float | None, dict]:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.get(url) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = {"cod": resp.status, "message": "Non-JSON response"}

                if str(data.get("cod")) != "200":
                    return None, data

                return float(data["main"]["temp"]), data
    except aiohttp.ClientError as e:
        return None, {"cod": None, "message": f"Network error: {e}"}
    

# ---------- UI ----------
st.title("📈 Temperature analysis + OpenWeatherMap (current temp vs seasonal norms)")

uploaded = st.file_uploader("Загрузите temperature_data.csv", type=["csv"])

api_key = st.text_input(
    "OpenWeatherMap API key (если не введён — текущая погода не показывается)",
    type="password",
    placeholder="Введите API ключ…",
)

if not uploaded:
    st.info("Загрузите CSV с колонками: city, timestamp, temperature, season")
    st.stop()

df = load_history(uploaded.getvalue())

# city selector
cities = sorted(df["city"].unique().tolist())
city = st.selectbox("Выберите город", cities)

df_city = df[df["city"] == city].copy()

# ---------- Layout ----------
left, right = st.columns([1, 1])

# ---------- Descriptive stats ----------
with left:
    st.subheader("Описательная статистика (исторические данные)")
    desc = df_city["temperature"].describe()
    st.dataframe(desc.to_frame(name="temperature").T, use_container_width=True)

    st.caption("Количество наблюдений по сезонам")
    season_counts = df_city["season"].value_counts().rename_axis("season").reset_index(name="count")
    st.dataframe(season_counts, use_container_width=True)

# ---------- Seasonal profiles ----------
with right:
    st.subheader("Сезонные профили (mean ± std)")

    season_profile = (
    df_city.groupby("season")["temperature"]
    .agg(season_mean="mean", season_std="std", n="count")
    .reset_index()
    )
    st.dataframe(season_profile, use_container_width=True)
    # plot bar with errorbars
    fig = plt.figure(figsize=(8, 4))
    x = np.arange(len(season_profile))
    means = season_profile["season_mean"].to_numpy()
    errs = season_profile["season_std"].to_numpy()
    plt.bar(x, means)
    plt.errorbar(x, means, yerr=errs, fmt="none", capsize=6)
    plt.xticks(x, season_profile["season"].tolist(), rotation=0)
    plt.xlabel("Season")
    plt.ylabel("Temperature (°C)")
    plt.title(f"{city}: сезонные средние и std")
    plt.tight_layout()
    st.pyplot(fig)

# ---------- Time series with anomalies ----------
st.subheader("Временной ряд температур (аномалии выделены)")

fig2 = plt.figure(figsize=(12, 5))
plt.plot(df_city["timestamp"], df_city["temperature"], label="Daily temperature")
plt.plot(df_city["timestamp"], df_city["ma30"], label="MA30 (rolling 30d)")

anom = df_city[df_city["is_anomaly"]]
if not anom.empty:
    plt.scatter(anom["timestamp"], anom["temperature"], label="Anomalies")

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title(f"{city}: daily temperature + anomalies (mean±2σ by season)")
plt.legend()
plt.tight_layout()
st.pyplot(fig2)

# ---------- Current weather + normality ----------
st.subheader("Текущая температура (OpenWeatherMap) и проверка нормальности")

if not api_key:
    st.info("Введите API-ключ, чтобы показать текущую погоду и проверить нормальность температуры.")
else:
    current_season = season_from_month(datetime.utcnow().month)

    # ВАЖНО: запускаем async корректно
    temp_now, raw = fetch_current_temp_sync_wrapper(city, api_key)

    if temp_now is None:
        st.error("Ошибка при запросе текущей погоды.")
        st.code(raw, language="json")
    else:
        row = season_profile.loc[season_profile["season"] == current_season, ["season_mean", "season_std"]].head(1)

        st.write(f"**Город:** {city}")
        st.write(f"**Текущая температура:** {temp_now:.2f} °C")
        st.write(f"**Сезон (UTC):** {current_season}")

        if row.empty:
            st.warning("Для этого города нет исторических данных по текущему сезону — оценить нормальность нельзя.")
        else:
            mean = float(row.iloc[0]["season_mean"])
            std = float(row.iloc[0]["season_std"]) if pd.notna(row.iloc[0]["season_std"]) else 0.0
            low = mean - 2 * std
            high = mean + 2 * std
            is_anom = (temp_now < low) or (temp_now > high)

            st.write(f"**Норма сезона:** mean={mean:.2f}°C, std={std:.2f}°C → диапазон [{low:.2f}, {high:.2f}]")
            st.error("Текущая температура **аномальная** (вне mean ± 2σ).") if is_anom else st.success(
                "Текущая температура **нормальная** (в пределах mean ± 2σ)."
            )

            with st.expander("Сырые данные ответа OpenWeatherMap"):
                st.code(raw, language="json")