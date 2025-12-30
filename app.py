from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

def season_from_month(month):
    #Функция для определения сезона по дате
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"

def default_pd(df):
    # Обработка данных с помощью классического pandas для создания статистических признаков
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    df["ma30"] = df.groupby("city")["temperature"].transform(lambda ma: ma.rolling(window=30, min_periods=1).mean())

    season_statistics = df.groupby(["city", "season"])["temperature"].agg(season_mean="mean", season_std="std", n="count").reset_index()
    df = df.merge(season_statistics, on=["city", "season"], how="left")

    df["lower_bound"] = df["season_mean"] - 2 * df["season_std"]
    df["upper_bound"] = df["season_mean"] + 2 * df["season_std"]
    df["is_anomaly"] = (df["temperature"] < df["lower_bound"]) | (df["temperature"] > df["upper_bound"])

    return df


#Название сервиса
st.set_page_config(page_title="Анализ и мониторинг температуры в реальном времени", layout="wide")

#Функции для обработки и кэширования загруженной таблицы с историческими температурными данными
@st.cache_data(show_spinner=False)
def load_history_uploaded(file_bytes: bytes) -> pd.DataFrame:
    df_raw = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    return default_pd(df_raw)

@st.cache_data(show_spinner=False)
def load_history_local(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)
    return default_pd(df_raw)


@st.cache_data(ttl=60, show_spinner=False)
def current_temp_sync(city: str, key: str) -> tuple[float | None, dict]:
    """
    Синхронный запрос текущей температуры.
    Возвращает (temp, raw_json) либо (None, raw_json_с_ошибкой).
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"

    try:
        resp = requests.get(url, timeout=(10, 20))
        # даже если статус не 200, API обычно возвращает JSON с cod/message
        try:
            data = resp.json()
        except Exception as e:
            return None, {"cod": resp.status_code, "message": f"JSON parse error: {e}"}

        if resp.status_code != 200 or str(data.get("cod")) != "200":
            return None, data

        try:
            temp = float(data["main"]["temp"])
            return temp, data
        except (KeyError, TypeError, ValueError) as e:
            return None, {"cod": data.get("cod"), "message": f"Invalid data: {e}"}

    except requests.Timeout as e:
        return None, {"cod": None, "message": f"Timeout error: {e}"}
    except requests.RequestException as e:
        return None, {"cod": None, "message": f"Network error: {e}"}
    except Exception as e:
        return None, {"cod": None, "message": f"Unexpected error: {e}"}
    

#Скользящее среднее по месяцам и построение тренда на 12 месяцев
def ma_by_month(df_city):
    df_city["month"] = df_city["timestamp"].dt.to_period("M").dt.to_timestamp()
    m = (
        df_city.groupby("month", as_index=False)["temperature"]
         .mean()
         .rename(columns={"temperature": "temp_month_mean"})
         .sort_values("month")
    )
    m["trend_12m"] = m["temp_month_mean"].rolling(window=12, min_periods=1).mean()
    return m

def plot_single_city_monthly(df_city, city):
    m = ma_by_month(df_city)
    if m.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["month"], 
        y=m["temp_month_mean"],
        mode="lines+markers", 
        name="Среднемесячная температура",
        hovertemplate="<b>%{x|%Y-%m}</b><br>Mean: %{y:.2f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=m["month"], 
        y=m["trend_12m"],
        mode="lines", 
        name="Тренд за 12 месяцев",
        hovertemplate="<b>%{x|%Y-%m}</b><br>Trend: %{y:.2f}°C<extra></extra>"
    ))
    fig.update_layout(
        title=f"{city}: Среднемесячная температура и тренд за 12 месяцев",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=420,
        hovermode="x unified"
    )
    return fig
    
#Делаем боковую панель для загрузки температурных данных из файла 
#Пользователь может выбрать анализ по локальным историческим данным проекта
with st.sidebar:
    st.header("Исторические данные")
    uploaded_data = st.file_uploader("Загрузите temperature_data.csv (опционально)", type=["csv"])
    local_data = st.checkbox("Использовать локальный temperature_data.csv", value=(uploaded_data is None))

    st.header("OpenWeatherMap")
    api_key = st.text_input(
        "Введите OpenWeatherMap API ключ для получения актуальной температуры",
        type="password",
        placeholder="API ключ"
    )

#Обрабатываем исторические данные и выделяем статистики
if uploaded_data is not None:
    history_data = load_history_uploaded(uploaded_data.getvalue())
elif local_data:
    history_data = load_history_local("temperature_data.csv")
else:
    st.stop()

if history_data.empty or "city" not in history_data.columns:
    st.error("Данные пустые или формат не соответствует ожидаемому (нужны колонки city, timestamp, temperature, season).")
    st.stop()

#Выпадающий список для выбора города для анализа
cities = sorted(history_data["city"].dropna().unique().tolist())
city = st.selectbox("Выберите город", cities, index=0)

df_city = history_data[history_data["city"] == city].copy()
anom = df_city[df_city.get("is_anomaly", False) == True].copy()

st.subheader("Описательная статистика для исторических данных")
st.dataframe(df_city["temperature"].describe())


st.subheader("Сезонные профили")
if df_city.empty:
    st.info("Нет данных по выбранному городу.")
else:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_city["season"],
        y=df_city["season_mean"],
        error_y=dict(type="data", array=df_city["season_std"], visible=True, thickness=1.5, width=3),
        name="Mean ± Std",
        hovertemplate=(
            "<b>Сезон:</b> %{x}<br>"
            "<b>Среднее:</b> %{y:.2f}°C<br>"
            "<b>Std:</b> %{customdata[0]:.2f}°C<br>"
            "<b>N:</b> %{customdata[1]}<extra></extra>"
            ),
        customdata=np.c_[df_city["season_std"].values, df_city["n"].values],
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


st.subheader("Временной ряд температур")
if df_city.empty:
    st.info("Нет данных по выбранному городу.")
else:
    fig2 = make_subplots(specs=[[{"secondary_y": False}]])

    fig2.add_trace(go.Scatter(
        x=df_city["timestamp"],
        y=df_city["temperature"],
        mode="lines",
        name="Дневная температура",
        customdata=df_city["season"],
        hovertemplate=(
            "Дата: {x|Y-m-d}",
            "Температура: {y:.2f}°C",
            "Сезон: {customdata}"
        ),
    ))

    if "ma30" in df_city.columns:
        fig2.add_trace(go.Scatter(
            x=df_city["timestamp"],
            y=df_city["ma30"],
            mode="lines",
            name="MA30"
        ))

    if not anom.empty:
        fig2.add_trace(go.Scatter(
            x=anom["timestamp"],
            y=anom["temperature"],
            mode="markers",
            name="Аномалии",
            marker=dict(size=8, symbol="x"),
            customdata=anom["season"],
            hovertemplate=(
                "<b>АНОМАЛИЯ</b><br>"
                "<b>Дата:</b> %{x|%Y-%m-%d}<br>"
                "<b>Температура:</b> %{y:.2f}°C<br>"
                "<b>Сезон:</b> %{customdata}<extra></extra>"
            ),
        ))

    #Интервал нормальных температур среднее±2𝜎
    if "season_mean" in df_city.columns and "season_std" in df_city.columns:
        std = df_city["season_std"].fillna(0.0)
        upper = df_city["upper_bound"]
        lower = df_city["lower_bound"]

        fig2.add_trace(go.Scatter(
            x=df_city["timestamp"].tolist() + df_city["timestamp"].tolist()[::-1],
            y=upper.tolist() + lower.tolist()[::-1],
            fill="toself",
            hoverinfo="skip",
            name="Нормальный интервал температур по временам года (средняя ± 2σ)",
            line=dict(width=0),
            showlegend=True,
        ))

    fig2.update_layout(
        title=f"{city}: Дневная температура с аномалиями",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig2, use_container_width=True)

st.header("Месячный анализ температур")

monthly_fig = plot_single_city_monthly(df_city, city)
if monthly_fig is not None:
    st.plotly_chart(monthly_fig, use_container_width=True)

    st.subheader("Анализ трендов по месяцам")
    monthly_data = ma_by_month(df_city)
    if not monthly_data.empty:
        latest = monthly_data.iloc[-1]
        st.metric(
            label="Текущий 12-месячный тренд",
            value=f"{latest['trend_12m']:.2f}°C",
            delta=f"Отличие в {(latest['temp_month_mean'] - latest['trend_12m']):.2f}°C от среднемесячного значения"
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

#Анализ текущей температуры для выбранного города
st.header("Анализ текущей температуры")

#Без введенного ключа не работаем
if not api_key:
    st.info("Для получения текущей температуры введите ваш API ключ.")
else:
    current_season = season_from_month(datetime.utcnow().month)
    temp, raw = current_temp_sync(city, api_key)

    if temp is None:
        st.error("Ошибка при обращении к API")
        st.json(raw)
    else:
        st.write(f"Выбранный город: {city}")
        st.write(f"Текущая температура: {temp} °C")
        st.write(f"Время года (eng): {current_season}")

        row = df_city.loc[df_city["season"] == current_season, ["season_mean", "season_std"]].head(1)

        if row.empty:
            st.warning("Отсутсвуют данные для данного времени года - нельзя сделать вывод об аномальности температуры.")
        else:
            mean = float(row.iloc[0]["season_mean"])
            std = float(row.iloc[0]["season_std"]) if pd.notna(row.iloc[0]["season_std"]) else 0.0

            lower_bound = float(row.iloc[0]["lower_bound"])
            upper_bound = float(row.iloc[0]["upper_bound"])
            is_anom = row.iloc[0]["is_anom"]

            st.write("Норма для данного времени года:")
            st.write(f"Средняя температура: {mean} °C")
            st.write(f"Дисперсия температуры: {std} °C")
            st.write(f"Нормальный температурный интервал (mean±2σ): [{lower_bound}, {upper_bound}]")

            if is_anom:
                st.error("Текущая температура аномальна!")
            else:
                st.success("Текущая температура в пределах нормы.")

        with st.expander("Сырые данные ответа OpenWeatherMap"):
            st.json(raw)
