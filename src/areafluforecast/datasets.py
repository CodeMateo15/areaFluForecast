"""Bundled sample data.

`load_sample("boston")`: weekly emergency-department influenza rates per 100,000
for 14 Boston neighborhoods (Boston Public Health Commission public dashboard),
their borders, 7 anchor nodes for the surrounding towns and water, centroids,
and weekly weather (Open-Meteo, CC BY 4.0). See sample_data/boston/README.md.

`load_sample("chicago")`: weekly flu-like-illness activity levels (1-10, not a
rate) for 58 Chicago ZIP codes and one anchor (Chicago Department of Public
Health, City of Chicago Data Portal), with borders, centroids and weather. See
sample_data/chicago/README.md, which carries the portal's required disclaimer.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pandas as pd

from .city import City
from .formats import check, stamp
from .weather import WEATHER_COLS, read_weather_csv

SAMPLES = ("boston", "chicago")


def _path(name: str, file: str):
    return resources.files("areafluforecast") / "sample_data" / name / file


def load_sample(name: str = "boston", *, weather: bool = True) -> City:
    if name not in SAMPLES:
        raise ValueError(f"Bundled samples: {', '.join(SAMPLES)}. Got {name!r}.")
    with resources.as_file(resources.files("areafluforecast") / "sample_data" / name) as folder:
        return load_city(folder, name=name, weather=weather)


def load_city(folder: str | Path, *, name: str | None = None, weather: bool = True) -> City:
    """A city saved with `save_city`: rates.csv, edges.csv, centroids.csv,
    city.json and, optionally, weather.csv."""
    folder = Path(folder)
    info = check(json.loads((folder / "city.json").read_text()), "city", folder)
    rates = pd.read_csv(folder / "rates.csv", parse_dates=["week"]).set_index("week")
    rates.columns = [str(c) for c in rates.columns]
    edges = pd.read_csv(folder / "edges.csv", dtype=str)
    cents = pd.read_csv(folder / "centroids.csv", dtype={"area": str})
    anchors = info.get("anchors", [])
    area_edges = [(a, b) for a, b in zip(edges["a"], edges["b"])
                  if a not in anchors and b not in anchors]
    anchor_edges = [(a, b) if b in anchors else (b, a) for a, b in zip(edges["a"], edges["b"])
                    if a in anchors or b in anchors]
    kwargs = {}
    if weather and (folder / "weather.csv").exists():
        kwargs["weather"] = read_weather_csv(folder / "weather.csv")
    return City(name=name or info.get("name") or folder.name, label=info["label"], rates=rates,
                edges=area_edges, flu_months=tuple(info["flu_months"]), anchors=anchors,
                anchor_edges=anchor_edges,
                centroids={r.area: (r.lat, r.lon) for r in cents.itertuples()}, **kwargs)


def save_city(city: City, folder: str | Path) -> Path:
    """Write a city in the layout `load_city` reads (weather only if the city has it)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rates = city.rates.copy()
    rates.index.name = "week"
    rates.to_csv(folder / "rates.csv", date_format="%Y-%m-%d")
    pd.DataFrame(city.edges + city.anchor_edges, columns=["a", "b"]).to_csv(
        folder / "edges.csv", index=False)
    pd.DataFrame([(a, lat, lon) for a, (lat, lon) in (city.centroids or {}).items()],
                 columns=["area", "lat", "lon"]).to_csv(folder / "centroids.csv", index=False)
    if city.weather:
        weather = pd.concat({col: city.weather[col].stack() for col in WEATHER_COLS}, axis=1)
        weather.index.names = ["week", "area"]
        weather.round(3).reset_index().to_csv(folder / "weather.csv", index=False,
                                              date_format="%Y-%m-%d")
    (folder / "city.json").write_text(json.dumps(
        {**stamp("city"), "name": city.name, "label": city.label, "flu_months": list(city.flu_months),
         "anchors": list(city.anchors)}, indent=2) + "\n")
    return folder
