"""A city: weekly rates per area, the area graph, and the season.

Everything the models need comes from data a health department already has:

* **rates** -- weekly influenza rate per area (per 100,000 residents is what the
  pretrained models expect; any consistent unit works for models you train).
  Missing or suppressed weeks are fine: leave them blank.
* **the graph** -- either a boundary file (GeoJSON) the neighbours are derived
  from, or an explicit edge list.
* **the season** -- which months are flu season (October-March by default;
  Southern-Hemisphere cities pass e.g. April-September).

Optional: weather per area (fetched for you from Open-Meteo), static area
characteristics, and *anchor* nodes -- areas on the boundary that have no data
but border your areas (neighbouring towns). Anchors only carry graph structure.
"""

from __future__ import annotations

import calendar
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import adjacency as adj
from .errors import DataError, DataWarning
from .weather import WEATHER_COLS, fetch_weather, read_weather_csv

DEFAULT_FLU_MONTHS = (10, 11, 12, 1, 2, 3)


@dataclass(kw_only=True)
class City:
    name: str
    rates: pd.DataFrame                              # weeks x areas
    edges: list[tuple[str, str]]                     # area-area borders
    flu_months: tuple[int, ...] = DEFAULT_FLU_MONTHS
    anchors: list[str] = field(default_factory=list)
    anchor_edges: list[tuple[str, str]] = field(default_factory=list)  # (area, anchor)
    centroids: dict[str, tuple[float, float]] | None = None            # (lat, lon)
    weather: dict[str, pd.DataFrame] | None = None   # column -> weeks x areas
    static: pd.DataFrame | None = None               # areas x numeric columns
    label: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rates = _regularise(self.rates, self.notes)
        self.edges = sorted({tuple(sorted((str(a), str(b)))) for a, b in self.edges if a != b})
        self.anchors = [str(a) for a in self.anchors]
        self.anchor_edges = sorted({(str(a), str(b)) for a, b in self.anchor_edges})
        self.flu_months = tuple(int(m) for m in self.flu_months)
        self.label = self.label or self.name
        _validate(self)

    # --- basic facts -------------------------------------------------------
    @property
    def areas(self) -> list[str]:
        return list(self.rates.columns)

    @property
    def n_areas(self) -> int:
        return self.rates.shape[1]

    @property
    def has_weather(self) -> bool:
        return self.weather is not None

    @property
    def season_first_month(self) -> int:
        months = set(self.flu_months)
        return next(m for m in range(1, 13) if m in months and ((m - 2) % 12) + 1 not in months)

    @property
    def flu_season_label(self) -> str:
        first = self.season_first_month
        last = ((first - 1 + len(self.flu_months) - 1) % 12) + 1
        return f"{calendar.month_abbr[first]}-{calendar.month_abbr[last]}"

    def season_window(self, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The scored window of season `year`: a full year opening four months
        before flu season starts (June-May for an October season), so it carries
        the off-season run-up. Matches the research backtest folds."""
        start = ((self.season_first_month - 1 - 4) % 12) + 1
        begin = pd.Timestamp(year=year, month=start, day=1)
        end = begin + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        return begin, end

    def season_label(self, year: int) -> str:
        begin, end = self.season_window(year)
        return f"{begin.year}-{str(end.year)[2:]}" if begin.year != end.year else str(begin.year)

    def complete_seasons(self, min_history_years: float = 2.0) -> list[int]:
        """Season years whose whole window is observed and has enough history before it."""
        first, last = self.rates.index.min(), self.rates.index.max()
        years = []
        for year in range(first.year, last.year + 1):
            begin, end = self.season_window(year)
            if end <= last and (begin - first).days >= 365.25 * min_history_years:
                years.append(year)
        return years

    def summary(self) -> str:
        lines = [f"{self.label}: {self.n_areas} areas, {len(self.rates)} weeks "
                 f"({self.rates.index.min().date()} to {self.rates.index.max().date()}), "
                 f"flu season {self.flu_season_label}",
                 f"  graph: {len(self.edges)} borders between areas"
                 + (f", {len(self.anchors)} anchors with {len(self.anchor_edges)} borders"
                    if self.anchors else ""),
                 f"  missing area-weeks: {float(self.rates.isna().mean().mean()):.1%}",
                 f"  weather: {'yes' if self.has_weather else 'no'}"
                 f" | static characteristics: "
                 f"{', '.join(self.static.columns) if self.static is not None else 'none'}",
                 f"  complete seasons available to backtest: "
                 f"{', '.join(self.season_label(y) for y in self.complete_seasons()) or 'none'}"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    # --- optional inputs ---------------------------------------------------
    def fetch_weather(self, *, cache_dir: str | Path | None = None, verbose: bool = True) -> City:
        """Fetch weekly weather at each area's centroid from Open-Meteo (no key needed)."""
        if not self.centroids:
            raise DataError("No area centroids to fetch weather at. Build the city from a "
                            "boundary file, or pass centroids={area: (lat, lon)}.")
        missing = [a for a in self.areas if a not in self.centroids]
        if missing:
            raise DataError(f"No centroid for areas {missing}.")
        cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "areafluforecast"
        self.weather = fetch_weather({a: self.centroids[a] for a in self.areas},
                                     self.rates.index, cache_dir=cache, verbose=verbose)
        return self

    def with_rates(self, rates: pd.DataFrame) -> City:
        """A copy with different rates (same areas), e.g. extended with future weeks."""
        return replace(self, rates=rates, notes=list(self.notes))

    # --- constructors ------------------------------------------------------
    @classmethod
    def from_frames(cls, name: str, rates: pd.DataFrame, *,
                    edges: Iterable[tuple[str, str]] | None = None,
                    polygons: list[dict] | None = None, id_col: str | None = None,
                    anchors: Iterable[str] = (), anchor_edges: Iterable[tuple[str, str]] = (),
                    flu_months: Iterable[int] = DEFAULT_FLU_MONTHS,
                    centroids: dict[str, tuple[float, float]] | None = None,
                    weather: dict[str, pd.DataFrame] | None = None,
                    static: pd.DataFrame | None = None, label: str | None = None) -> City:
        """From in-memory objects.

        rates      weeks x areas frame of weekly rates.
        polygons   a list of GeoJSON features; neighbours are derived from them,
                   and `id_col` names the property holding the area id.
        edges      alternatively, (area, area) pairs.
        weather    {column: weeks x areas frame} for each of the weather columns.
        static     areas x numeric characteristics.
        The rest are as in `City`.
        """
        if polygons is not None:
            if id_col is None:
                raise DataError("Pass id_col: the GeoJSON property that holds the area id.")
            derived = adj.adjacency_from_polygons(polygons, id_col)
            centroids = centroids or adj.centroids_from_polygons(polygons, id_col)
            area_set, anchor_set = set(map(str, rates.columns)), set(map(str, anchors))
            edges = [p for p in derived if p[0] in area_set and p[1] in area_set]
            anchor_edges = list(anchor_edges) + [
                (a, b) if a in area_set else (b, a) for a, b in derived
                if (a in area_set) != (b in area_set) and ({a, b} & anchor_set)]
        if edges is None:
            raise DataError("Give the graph: either polygons (a boundary file) or edges.")
        return cls(name=name, rates=rates, edges=list(edges), anchors=list(anchors),
                   anchor_edges=list(anchor_edges), centroids=centroids, flu_months=flu_months,
                   weather=weather, static=static, label=label)

    @classmethod
    def from_files(cls, rates: str | Path, *, areas: str | Path | None = None,
                   edges: str | Path | None = None, id_col: str | None = None,
                   name: str | None = None, flu_months: Iterable[int] = DEFAULT_FLU_MONTHS,
                   week_col: str = "week", area_col: str = "area", rate_col: str = "rate",
                   anchors: Iterable[str] = (), weather: str | Path | None = None,
                   static: str | Path | None = None,
                   centroids: str | Path | None = None, label: str | None = None) -> City:
        """Build a city from files.

        rates   CSV, either long (columns `week`, `area`, `rate`) or wide (first
                column the week start date, one column per area).
        areas   GeoJSON boundary file; neighbours are derived from it. `id_col`
                names the property holding the area id used in `rates`.
        edges   alternatively, a CSV edge list (first two columns = two areas).
                Edges to names listed in `anchors` become anchor edges.
        weather long CSV with `week`, `area` and the six weather columns.
        static  CSV with an `area` column and numeric area characteristics.
        centroids CSV with `area`, `lat`, `lon` (only needed to fetch weather
                without a boundary file).
        """
        frame = read_rates_csv(rates, week_col=week_col, area_col=area_col, rate_col=rate_col)
        anchors = [str(a) for a in anchors]
        kwargs = dict(flu_months=flu_months, label=label)
        if weather is not None:
            kwargs["weather"] = read_weather_csv(weather, area_col=area_col)
        if static is not None:
            table = pd.read_csv(static, dtype={area_col: str}).set_index(area_col)
            kwargs["static"] = table
        if centroids is not None:
            table = pd.read_csv(centroids, dtype={area_col: str})
            kwargs["centroids"] = {r[area_col]: (float(r["lat"]), float(r["lon"]))
                                   for _, r in table.iterrows()}
        city_name = name or Path(rates).stem
        if areas is not None:
            return cls.from_frames(city_name, frame, polygons=adj.read_geojson(areas),
                                   id_col=id_col, anchors=anchors, **kwargs)
        if edges is None:
            raise DataError("Give the graph: areas='boundaries.geojson' or edges='edges.csv'.")
        pairs = adj.read_edges_csv(edges)
        anchor_set = set(anchors)
        area_edges = [p for p in pairs if not (set(p) & anchor_set)]
        anchor_edges = [(a, b) if b in anchor_set else (b, a) for a, b in pairs
                        if set((a, b)) & anchor_set]
        return cls.from_frames(city_name, frame, edges=area_edges, anchors=anchors,
                               anchor_edges=anchor_edges, **kwargs)


# --- reading and checking ----------------------------------------------------

def read_rates_csv(path: str | Path, *, week_col: str = "week", area_col: str = "area",
                   rate_col: str = "rate") -> pd.DataFrame:
    """Weekly rates as a weeks x areas frame, from a long or a wide CSV."""
    raw = pd.read_csv(path)
    if {week_col, area_col, rate_col} <= set(raw.columns):
        raw[week_col] = pd.to_datetime(raw[week_col])
        raw[area_col] = raw[area_col].astype(str)
        duplicated = raw.duplicated([week_col, area_col])
        if duplicated.any():
            example = raw.loc[duplicated, [week_col, area_col]].iloc[0].tolist()
            raise DataError(f"{path}: {int(duplicated.sum())} duplicate week/area rows, "
                            f"e.g. {example}. One rate per area per week.")
        return raw.pivot(index=week_col, columns=area_col, values=rate_col)
    first = raw.columns[0]
    try:
        index = pd.to_datetime(raw[first])
    except (ValueError, TypeError) as exc:
        raise DataError(
            f"{path}: expected either long columns {week_col!r}, {area_col!r}, {rate_col!r} "
            f"or a wide table whose first column is the week date. Got columns "
            f"{list(raw.columns)[:6]}.") from exc
    frame = raw.drop(columns=[first]).set_index(index)
    frame.index.name = "week"
    frame.columns = [str(c) for c in frame.columns]
    return frame


def rates_from_counts(counts: pd.DataFrame, population: pd.Series | dict,
                      per: float = 100_000) -> pd.DataFrame:
    """Weekly counts (weeks x areas) to rates per `per` residents."""
    population = pd.Series(population, dtype=float)
    missing = [c for c in counts.columns if c not in population.index]
    if missing:
        raise DataError(f"No population for areas {missing}.")
    return counts.astype(float).div(population[counts.columns]) * per


def _regularise(rates: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    frame = rates.copy()
    frame.columns = [str(c) for c in frame.columns]
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise DataError("The same week appears twice in the rates.")
    try:
        frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise DataError(f"Rates must be numbers (blank for missing weeks): {exc}") from exc
    weekdays = frame.index.dayofweek.unique()
    if len(weekdays) > 1:
        raise DataError(
            "Week dates fall on different weekdays "
            f"({sorted(calendar.day_abbr[d] for d in weekdays)}). Use one date per week, "
            "e.g. the Sunday the week starts.")
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq="7D")
    if len(grid) != len(frame):
        notes.append(f"{len(grid) - len(frame)} whole weeks were absent and are treated as missing.")
        frame = frame.reindex(grid)
    frame.index.name = "week"
    return frame


def _validate(city: City) -> None:
    rates = city.rates
    if city.n_areas < 2:
        raise DataError("Need at least two areas: the graph models forecast areas jointly.")
    if (rates < 0).any().any():
        raise DataError("Rates cannot be negative.")
    if len(rates) < 104:
        raise DataError(f"Only {len(rates)} weeks of data; at least two years (104 weeks) "
                        "are needed to learn a season.")
    if not set(city.flu_months) <= set(range(1, 13)) or not city.flu_months:
        raise DataError(f"flu_months must be calendar months 1-12, got {city.flu_months}.")
    try:
        _ = city.season_first_month
    except StopIteration:
        raise DataError(f"flu_months {city.flu_months} must be one contiguous run of months.") from None

    areas, anchors = set(city.areas), set(city.anchors)
    if areas & anchors:
        raise DataError(f"{sorted(areas & anchors)} are both areas with data and anchors.")
    unknown = {n for pair in city.edges for n in pair} - areas
    if unknown:
        raise DataError(
            f"Edges mention {sorted(unknown)}, which are not columns of the rates. Fix the "
            "spelling, or list boundary nodes without data in anchors=[...].")
    for a, b in city.anchor_edges:
        if a not in areas or b not in anchors:
            raise DataError(f"Anchor edge ({a}, {b}) must join an area to a listed anchor.")
    unused = anchors - {b for _, b in city.anchor_edges}
    if unused:
        raise DataError(f"Anchors {sorted(unused)} border no area.")

    lonely = adj.isolated(city.areas, city.edges)
    if lonely:
        warnings.warn(f"{lonely} border no other area, so the graph tells the model nothing "
                      "about them. Check the boundary file or edge list.", DataWarning, stacklevel=3)
    sparse = rates.columns[rates.isna().mean() > 0.5].tolist()
    if sparse:
        warnings.warn(f"{sparse} are missing more than half their weeks.", DataWarning,
                      stacklevel=3)
    if city.weather is not None:
        absent = [c for c in WEATHER_COLS if c not in city.weather]
        if absent:
            raise DataError(f"Weather lacks columns {absent}.")
        missing = [a for a in city.areas if a not in city.weather[WEATHER_COLS[0]].columns]
        if missing:
            raise DataError(f"No weather for areas {missing}.")
    if city.static is not None:
        city.static.index = city.static.index.astype(str)
        missing = [a for a in city.areas if a not in city.static.index]
        if missing:
            raise DataError(f"No static characteristics for areas {missing}.")
        city.static = city.static.loc[city.areas].apply(pd.to_numeric, errors="raise")
        if city.static.isna().any().any():
            raise DataError("Static characteristics contain blanks; fill or drop those columns.")


def scaled_static(city: City) -> tuple[np.ndarray, list[str]] | tuple[None, list[str]]:
    """Static characteristics min-max scaled to [0, 1] across areas."""
    if city.static is None:
        return None, []
    values = city.static.to_numpy(dtype=np.float64)
    low, high = values.min(axis=0), values.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    return ((values - low) / span).astype(np.float32), list(city.static.columns)
