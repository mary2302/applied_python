import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
import modin.pandas as mpd
import polars as pl
import time

df = pd.read_csv("temperature_data.csv")

#Соберем обычный подход в функцию, чтобы удобнее было замерять время работы

def default_pd(df):
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

def one_city(c):
    c = c.copy()
    c = c.sort_values("timestamp")    
    c["ma30"] = c["temperature"].rolling(window=30, min_periods=1).mean()

    season_statistics = c.groupby("season")["temperature"].agg(season_mean="mean", season_std="std", n="count").reset_index()
    c = c.merge(season_statistics, on="season", how="left")

    c["lower_bound"] = c["season_mean"] - 2 * c["season_std"]
    c["upper_bound"] = c["season_mean"] + 2 * c["season_std"]
    c["is_anomaly"] = (c["temperature"] < c["lower_bound"]) | (c["temperature"] > c["upper_bound"])

    return c

#Параллельный анализ с помощью модуля multiprocessing
def multiprocessing_pd(df, nprocs):
    nprocs = nprocs or max(1, cpu_count() - 1)

    cities = [c for _, c in df.groupby("city", sort=False)]

    with Pool(processes=nprocs) as pool:
        parts = pool.map(one_city, cities)

    df = pd.concat(parts, ignore_index=True)

    return df

#Параллельный анализ с помощью modin на движке Ray
def modin_pd(df):
    df = mpd.DataFrame(df.copy())
    df["timestamp"] = mpd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    df["ma30"] = df.groupby("city")["temperature"].transform(lambda ma: ma.rolling(window=30, min_periods=1).mean())

    season_statistics = df.groupby(["city", "season"])["temperature"].agg(season_mean="mean", season_std="std", n="count").reset_index()
    df = df.merge(season_statistics, on=["city", "season"], how="left")

    df["lower_bound"] = df["season_mean"] - 2 * df["season_std"]
    df["upper_bound"] = df["season_mean"] + 2 * df["season_std"]
    df["is_anomaly"] = (df["temperature"] < df["lower_bound"]) | (df["temperature"] > df["upper_bound"])
    return df

#Параллельный анализ с помощью polars
def polars_pd(path) -> pl.DataFrame:
    #Загружаем через Polars df
    df = pl.read_csv(path, try_parse_dates=True).with_columns([pl.col("timestamp")])
    df = df.sort(["city", "timestamp"])
    
    #Скользящее среднее ma30 по каждому городу
    df = df.with_columns(pl.col("temperature").rolling_mean(window_size=30, min_samples=1).over("city").alias("ma30"))

    stats = (
        df.group_by(["city", "season"])
          .agg([
              pl.col("temperature").mean().alias("season_mean"),
              pl.col("temperature").std().alias("season_std"),
              pl.len().alias("n")
          ])
    )

    df = df.join(stats, on=["city", "season"], how="left")

    df = df.with_columns([
        (pl.col("season_mean") - 2 * pl.col("season_std")).alias("lower_bound"),
        (pl.col("season_mean") + 2 * pl.col("season_std")).alias("upper_bound"),
    ])

    df = df.with_columns([
        (
            (pl.col("temperature") < pl.col("lower_bound")) |
            (pl.col("temperature") > pl.col("upper_bound"))
        ).alias("is_anomaly")
    ])

    return df

def benchmark(func, label, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(f"{label:22s} ")
    print(f"median={np.median(times):.4f}s  min={np.min(times):.4f}s  max={np.max(times):.4f}s  ")
    print(f"runs={['%.4f' % t for t in times]}")

def main():

    benchmark(lambda: default_pd(df), "default pandas", repeats=5)

    benchmark(lambda: multiprocessing_pd(df, nprocs=4), "pandas multiprocessing", repeats=5)

    benchmark(lambda: modin_pd(df), "modin", repeats=5)

    benchmark(lambda: polars_pd("temperature_data.csv"), "polars", repeats=5)

if __name__ == "__main__":
    main()
