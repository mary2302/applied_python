import requests
from dotenv import load_dotenv
import os
from datetime import datetime
import pandas as pd
import aiohttp
import asyncio
from datetime import datetime
import time

from parallel_EDA import default_pd

load_dotenv()
key = os.getenv("API_KEY")
df = pd.read_csv("temperature_data.csv")

history_data = default_pd(df)

def season_from_month(month):
    #Функция для определения сезона по дате
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"

def current_temp_sync(city):
    #Отправляем синхронно запрос в API на получение актуальной температуры
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(f"Погода в {city}: {data['main']['temp']}°C")
    else:
        print("Ошибка при запросе данных")
    return float(data["main"]["temp"])

def is_normal(city, temp, season):
    #Проверяем текущую температуру на нормальность для сезона
    row = history_data.loc[
        (history_data["city"] == city) & (history_data["season"] == season),
        ["season_mean", "season_std"]
    ].head(1)

    mean = float(row.iloc[0]["season_mean"])
    std = float(row.iloc[0]["season_std"])

    low = mean - 2 * std
    high = mean + 2 * std
    is_anom = (temp < low) or (temp > high)

    return {
        "city": city,
        "season": season,
        "temp": temp,
        "season_mean": mean,
        "season_std": std,
        "lower_bound": low,
        "upper_bound": high,
        "is_anomaly": is_anom,
    }

def check_city_sync(city):
    season = season_from_month(datetime.utcnow().month)
    temp = current_temp_sync(city)
    return is_normal(city, temp, season)

async def current_temp_async(session: aiohttp.ClientSession, city: str) -> float:
        #Асинхронно посылаем запросы в API для получения текущей температуры 
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as response:
            response.raise_for_status()  # выбросит исключение при 4xx/5xx
            data = await response.json()
            return float(data["main"]["temp"])

async def check_cities_async(cities: list[str], concurrency: int = 10):
    season = season_from_month(datetime.utcnow().month)
    #Семафор для ограничения одновременных запросов
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        async def one(city: str):
            async with sem:
                temp = await current_temp_async(session, city)
                return is_normal(city, temp, season)

        return await asyncio.gather(*(one(c) for c in cities))

def main():
    cities = history_data["city"].unique()

    #Синхронный прогон
    t0 = time.perf_counter()
    results_sync = [check_city_sync(city) for city in cities]
    t1 = time.perf_counter()
    print(f"Время работы синхронного варианта: {t1-t0}")
    pd.DataFrame(results_sync).to_csv("data/results_sync.csv", index=False)


    #Асинхронный прогон
    t2 = time.perf_counter()
    results_async = asyncio.run(check_cities_async(cities, concurrency=8))
    t3 = time.perf_counter()
    print(f"Время работы aсинхронного варианта: {t3-t2}")
    pd.DataFrame(results_async).to_csv("data/results_async.csv", index=False)



if __name__ == "__main__":
    main()
