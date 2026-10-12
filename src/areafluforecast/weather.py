"""Weekly weather per area from the Open-Meteo archive (free, no API key).

Weather data by Open-Meteo.com (CC BY 4.0). Daily values at each area's centroid
are aggregated to the same weeks as the flu data.

Weather is optional. In the research behind this package it helped the graph
model a little (mostly at 2 weeks ahead); the models run on rates alone.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import DataError

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLES = {
    # Open-Meteo name: (column, weekly aggregation)
    "temperature_2m_mean": ("temp_mean_c", "mean"),
    "temperature_2m_max": ("temp_max_c", "mean"),
    "temperature_2m_min": ("temp_min_c", "mean"),
    "relative_humidity_2m_mean": ("relative_humidity_mean", "mean"),
    "precipitation_sum": ("precipitation_sum_mm", "sum"),
    "wind_speed_10m_max": ("wind_speed_max_kmh", "mean"),
}
WEATHER_COLS = [column for column, _ in DAILY_VARIABLES.values()]
MAX_RETRIES = 8
# Open-Meteo's historical archive is reanalysis on a grid of roughly 0.1 degrees
# (~9-11 km), so centroids closer than that get the same weather. Rounding to
# the grid lets nearby areas share one request, which keeps a 60-ZIP city under
# the free API's rate limit without losing any real detail.
GRID_DEGREES = 0.1


def fetch_daily(lat: float, lon: float, start: str, end: str, *, cache_dir: Path | None = None,
                delay: float = 1.0) -> pd.DataFrame:
    """Daily weather at one point, indexed by date. Cached on disk when `cache_dir` is set."""
    import requests

    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": start,
              "end_date": end, "daily": ",".join(DAILY_VARIABLES), "timezone": "auto"}
    cache_file = None
    if cache_dir is not None:
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        cache_file = Path(cache_dir) / f"openmeteo_{key}.json"
        if cache_file.exists():
            return _daily_frame(json.loads(cache_file.read_text()))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as exc:  # network errors and bad JSON alike
            status = getattr(getattr(exc, "response", None), "status_code", None) or 0
            limited = status == 429
            if 400 <= status < 500 and not limited:
                # The request itself is wrong (e.g. dates outside the archive); retrying won't help.
                raise RuntimeError(f"Open-Meteo rejected the request ({status}): {exc}") from exc
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Open-Meteo request failed after {MAX_RETRIES} tries: {exc}") from exc
            # The free tier limits calls per minute: on a 429, wait out the minute.
            time.sleep(65 if limited else 2 ** attempt)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
    time.sleep(delay)
    return _daily_frame(payload)


def _daily_frame(payload: dict) -> pd.DataFrame:
    daily = pd.DataFrame(payload["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    return daily.set_index("time")


def to_weeks(daily: pd.DataFrame, week_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate daily rows to the weeks that start on each date of `week_index`."""
    start_dow = int(week_index[0].dayofweek)
    week_start = daily.index - pd.to_timedelta((daily.index.dayofweek - start_dow) % 7, unit="D")
    grouped = daily.groupby(week_start)
    weekly = pd.DataFrame({column: getattr(grouped[name], how)()
                           for name, (column, how) in DAILY_VARIABLES.items()})
    return weekly.reindex(week_index)


def fetch_weather(centroids: dict[str, tuple[float, float]], week_index: pd.DatetimeIndex, *,
                  cache_dir: Path | None = None, verbose: bool = True,
                  grid: float = GRID_DEGREES) -> dict[str, pd.DataFrame]:
    """Weekly weather for every area: {column: DataFrame(weeks x areas)}.

    Areas whose centroids fall in the same `grid`-degree cell share one request
    (set grid=0 to fetch every centroid separately).
    """
    start = str(week_index.min().date())
    last_day = week_index.max() + pd.Timedelta(days=6)
    yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    end = str(min(last_day, yesterday).date())

    def cell(lat: float, lon: float) -> tuple[float, float]:
        if not grid:
            return lat, lon
        return round(round(lat / grid) * grid, 4), round(round(lon / grid) * grid, 4)

    cells = {area: cell(lat, lon) for area, (lat, lon) in centroids.items()}
    unique = sorted(set(cells.values()))
    fetched: dict[tuple[float, float], pd.DataFrame] = {}
    for i, (lat, lon) in enumerate(unique, start=1):
        if verbose:
            print(f"  weather {i}/{len(unique)} ({lat:.2f}, {lon:.2f})", flush=True)
        fetched[(lat, lon)] = to_weeks(fetch_daily(lat, lon, start, end, cache_dir=cache_dir),
                                       week_index)
    per_area = {area: fetched[c] for area, c in cells.items()}
    return {column: pd.DataFrame({area: frame[column] for area, frame in per_area.items()},
                                 index=week_index)
            for column in WEATHER_COLS}


def read_weather_csv(path: str | Path, *, week_col: str = "week", area_col: str = "area"
                     ) -> dict[str, pd.DataFrame]:
    """Weather from a long CSV: week, area, and any of WEATHER_COLS as columns."""
    frame = pd.read_csv(path, parse_dates=[week_col])
    missing = [c for c in WEATHER_COLS if c not in frame.columns]
    if missing:
        raise DataError(f"{path} lacks weather columns {missing}. Expected: {WEATHER_COLS}")
    frame[area_col] = frame[area_col].astype(str)
    return {column: frame.pivot(index=week_col, columns=area_col, values=column).sort_index()
            for column in WEATHER_COLS}


def align_weather(weather: dict[str, pd.DataFrame], week_index: pd.DatetimeIndex,
                  areas: list[str]) -> dict[str, np.ndarray]:
    """{column: (weeks, areas) array} on `week_index`, gaps filled.

    A short gap is carried forward from the area's own last week (the archive
    lags real time by a few days, so the newest week is often missing). Anything
    left is filled with the city-wide median: weather is spatially smooth.
    """
    out = {}
    for column in WEATHER_COLS:
        frame = weather[column].reindex(index=week_index, columns=areas).astype(float)
        frame = frame.ffill(limit=2)
        values = frame.to_numpy(dtype=np.float32)
        out[column] = np.where(np.isnan(values), np.nanmedian(values), values)
    return out
