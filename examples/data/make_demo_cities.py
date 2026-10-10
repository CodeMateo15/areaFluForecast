"""Write the two made-up demo cities used by the example notebooks.

These are SYNTHETIC: no real place, no real surveillance data. They exist so the
examples can show the file formats a health department would supply.

* Riverton: 12 districts on a 3 x 4 grid, about 7 years of weekly rates, plus two
  neighbouring areas with no data (a township and a lake) as anchors.
* Lakeside: 7 districts, only ~2.3 years of data -- the "short history" case.

Each season's epidemic starts in one district and spreads to neighbours with a
delay that grows with distance, so the map carries real information.

    python examples/data/make_demo_cities.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def square(x: float, y: float, size: float) -> list:
    return [[[round(x, 6), round(y, 6)], [round(x + size, 6), round(y, 6)],
             [round(x + size, 6), round(y + size, 6)], [round(x, 6), round(y + size, 6)],
             [round(x, 6), round(y, 6)]]]


def feature(name: str, ring: list, kind: str = "district") -> dict:
    return {"type": "Feature", "properties": {"district": name, "kind": kind},
            "geometry": {"type": "Polygon", "coordinates": ring}}


def simulate(cells: dict[str, tuple[int, int]], weeks: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Weekly ILI rate per 100k: seasonal waves spreading from a random origin district."""
    rng = np.random.default_rng(seed)
    names = list(cells)
    pos = np.array([cells[n] for n in names], dtype=float)
    size = rng.uniform(0.7, 1.4, len(names))          # districts differ in baseline level
    rates = np.zeros((len(weeks), len(names)))
    t = np.arange(len(weeks))
    for year in range(weeks[0].year - 1, weeks[-1].year + 1):
        peak_day = pd.Timestamp(year=year, month=12, day=15) + pd.Timedelta(days=int(rng.integers(0, 70)))
        origin = rng.integers(len(names))
        distance = np.abs(pos - pos[origin]).sum(axis=1)  # border hops on the grid
        height = rng.uniform(35, 75)
        width = rng.uniform(3.0, 5.0)
        peak_week = (peak_day - weeks[0]).days / 7
        for j in range(len(names)):
            centre = peak_week + 1.3 * distance[j]
            rates[:, j] += size[j] * height * np.exp(-0.5 * ((t - centre) / width) ** 2)
    rates += 4 * size[None, :]
    noisy = rng.poisson(np.maximum(rates, 0.1) * 3) / 3      # count noise
    frame = pd.DataFrame(noisy.round(1), index=weeks, columns=names)
    suppressed = (frame < 2.5) & (rng.random(frame.shape) < 0.5)   # small counts suppressed
    return frame.mask(suppressed)


def riverton() -> None:
    names = ["Old Town", "Harbor", "Millbrook", "Eastgate",
             "Westfield", "Central", "Parkview", "Ironworks",
             "Northside", "Hillcrest", "Riverside", "Southpoint"]
    cells = {n: (i % 4, i // 4) for i, n in enumerate(names)}
    lon0, lat0, size = -82.30, 40.10, 0.02
    features = [feature(n, square(lon0 + c * size, lat0 + r * size, size)) for n, (c, r) in cells.items()]
    # Anchors: a township along the north edge and a lake along the east edge. No data.
    # Shared borders repeat every vertex of the districts they touch, as a
    # topologically clean boundary file does.
    south_edge = [[lon0 + c * size, lat0 + 3 * size] for c in range(5)]
    features.append(feature("North Township", [south_edge + [[lon0 + 4 * size, lat0 + 4 * size],
                                                             [lon0, lat0 + 4 * size],
                                                             [lon0, lat0 + 3 * size]]], kind="anchor"))
    features.append(feature("Lake Erin", [[[lon0 + 4 * size, lat0], [lon0 + 5 * size, lat0],
                                           [lon0 + 5 * size, lat0 + 3 * size], [lon0 + 4 * size, lat0 + 3 * size],
                                           [lon0 + 4 * size, lat0 + 2 * size], [lon0 + 4 * size, lat0 + size],
                                           [lon0 + 4 * size, lat0]]], kind="anchor"))
    for f in features:
        f["geometry"]["coordinates"] = [[[round(x, 6), round(y, 6)] for x, y in ring]
                                        for ring in f["geometry"]["coordinates"]]
    (HERE / "riverton_districts.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=1))
    weeks = pd.date_range("2019-06-02", "2026-05-31", freq="7D")
    rates = simulate(cells, weeks, seed=11)
    long = rates.rename_axis("week").reset_index().melt("week", var_name="district", value_name="rate")
    long.to_csv(HERE / "riverton_rates.csv", index=False, date_format="%Y-%m-%d")


def lakeside() -> None:
    names = ["Bayfront", "Cedar", "Dunmore", "Elm Park", "Fairview", "Glenwood", "Highland"]
    cells = {n: (i % 3, i // 3) for i, n in enumerate(names)}
    weeks = pd.date_range("2024-02-04", "2026-05-31", freq="7D")
    rates = simulate(cells, weeks, seed=29)
    rates.rename_axis("week").to_csv(HERE / "lakeside_rates.csv", date_format="%Y-%m-%d")
    edges = sorted({tuple(sorted((a, b))) for a in names for b in names
                    if a != b and sum(abs(p - q) for p, q in zip(cells[a], cells[b])) == 1})
    pd.DataFrame(edges, columns=["district_a", "district_b"]).to_csv(HERE / "lakeside_edges.csv", index=False)


if __name__ == "__main__":
    riverton()
    lakeside()
    print("wrote", *sorted(p.name for p in HERE.glob("*") if p.suffix in (".csv", ".geojson")))
