"""Build package City objects from the GNN-Influenza-Boston research repo.

The parity check and the bundled sample both need *exactly* the data the paper
used, so they read it through the research repo's own loaders rather than
re-parsing the source files.

    AREAFLU_RESEARCH_REPO=~/GNN-Influenza-Boston python scripts/export_sample.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from areafluforecast import City
from areafluforecast.weather import WEATHER_COLS


def research_repo() -> Path:
    path = Path(os.environ.get("AREAFLU_RESEARCH_REPO", "~/GNN-Influenza-Boston")).expanduser()
    if not (path / "Code" / "influenza").exists():
        raise SystemExit(f"Research repo not found at {path}; set AREAFLU_RESEARCH_REPO.")
    code = str(path / "Code")
    if code not in sys.path:
        sys.path.insert(0, code)
    return path


# The research backtest's data variant per city (run_backtest.DEFAULT_VARIANT).
# Chicago runs on "full": post_covid would start it in July 2022, leaving the live
# track one past season to check itself on. Its 2020 gap (May-Sep, off-season)
# stays NaN, which the package masks like Boston's suppressed cells.
VARIANT = {"boston": "full", "columbus": "post_covid", "buenos_aires": "post_covid",
           "chicago": "full"}


def boston_from_research(*, with_static: bool = True, with_weather: bool = True) -> City:
    """Boston exactly as the paper's backtest saw it (variant "full")."""
    return city_from_research("boston", with_static=with_static, with_weather=with_weather)


def city_from_research(name: str, *, with_static: bool = True, with_weather: bool = True) -> City:
    """A research city exactly as its backtest saw it."""
    research_repo()
    from influenza.cities import get
    from influenza.windows import variant_data

    city = get(name)
    rates = variant_data(city.loaders.load_rates(), VARIANT.get(name, "full")).available
    long_of = dict(zip(city.short_names, city.node_names))
    edges = [(long_of[a], long_of[b]) for a, b in city.geo_edges]
    anchor_edges = [(long_of[a], b) if a in long_of else (long_of[b], a)
                    for a, b in city.geo_edges_anchor]
    centroids = {long_of.get(short, short): (lat, lon) for short, (lon, lat) in city.coords.items()
                 if short in long_of}
    kwargs = {}
    if with_weather:
        arrays = city.loaders.load_weather(rates.index)
        kwargs["weather"] = {col: pd.DataFrame(arrays[col], index=rates.index,
                                               columns=list(city.node_names))
                             for col in WEATHER_COLS}
    if with_static:
        values, cols, _ = city.loaders.load_static_demographics()
        kwargs["static"] = pd.DataFrame(values, index=list(city.node_names), columns=cols)
    return City(name=name, label=city.label, rates=rates, edges=edges,
                flu_months=tuple(sorted(city.flu_months, key=lambda m: (m < 8, m))),
                anchors=list(city.anchor_short), anchor_edges=anchor_edges,
                centroids=centroids, **kwargs)


def maybe_quick(config):
    """AREAFLU_QUICK=1 shrinks a config to seconds, to smoke-test a script before the cluster."""
    from dataclasses import fields, replace

    if not os.environ.get("AREAFLU_QUICK"):
        return config
    names = {f.name for f in fields(config)}
    small = {"n_seeds": 1, "epochs": 2, "mc_samples": 4}
    return replace(config, **{k: v for k, v in small.items() if k in names})
