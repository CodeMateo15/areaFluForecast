# Preparing your data

You need **two files**: weekly rates, and a map. Everything else is optional.

!!! note "What you bring, and what the package does"
    The package doesn't look up a city by name or download data for one. It
    builds everything from the two files below, so it works for any city,
    district or county, in any country, that has weekly rates per area and a
    map of those areas. The only cities it knows by name are the two bundled
    samples, `af.load_sample("boston")` and `af.load_sample("chicago")`.

## 1. Weekly rates

This is a CSV with one value per area per week, in either of two layouts.

**Long:** one row per area per week. The default column names are `week`,
`area` and `rate`; pass `week_col=`, `area_col=` or `rate_col=` if yours differ.

| week | area | rate |
|---|---|---|
| 2024-01-07 | Riverside | 41.2 |
| 2024-01-07 | Old Town | 28.9 |
| 2024-01-14 | Riverside | 47.5 |

**Wide:** the week start date in the first column, then one column per area.

| week | Riverside | Old Town |
|---|---|---|
| 2024-01-07 | 41.2 | 28.9 |
| 2024-01-14 | 47.5 | 31.0 |

**Rules:**
- **Units.** Rates per 100,000 residents are what the pretrained models expect.
  Any consistent unit works for models you train yourself.
- **One date per week**, all on the same weekday (for example the Sunday that
  starts the week).
- **Missing or suppressed weeks stay blank.** Don't fill them with 0: a zero is
  a real observation and would teach the models something false. Blank weeks
  are skipped when scoring and filled from the past only when used as inputs.
- **At least two years** of weeks, so the models can learn a season. Three or
  more is better, and backtesting a season needs two years of history before
  it.

**If you have counts rather than rates:**

```python
rates = af.rates_from_counts(counts, population)   # counts: weeks x areas; population: {area: residents}
```

## 2. The map

The graph model needs to know which areas border each other. Give it one of
these.

**A boundary file (GeoJSON)**, with one feature per area and a property holding
the area name used in your rates:

```python
city = af.City.from_files("rates.csv", areas="areas.geojson", id_col="NAME")
```

The package finds the borders itself. Two areas are neighbours when their
outlines share a stretch of boundary; touching at a single corner doesn't count.
Several features may share a name (ZIP codes grouped into one area, say) and
are merged.

**Or an edge list**: a CSV whose first two columns are pairs of bordering areas.

```python
city = af.City.from_files("rates.csv", edges="borders.csv")
```

!!! tip "Check the map"
    `city.summary()` reports how many borders were found, and the package warns
    about any area with no neighbours. Plot the borders once ([example
    2](examples/02_your_own_data.ipynb) shows how). A wrong border feeds the
    model wrong information.

    The default rule needs neighbouring outlines to share exactly the same
    corner points. Census TIGER files and most city open-data portals are drawn
    that way. A map whose areas were drawn separately is not: its borders sit a
    few metres apart, and the default rule **silently misses them** rather than
    failing. An area with no neighbours, or far fewer borders than you expect,
    is the sign. For such a map use
    `af.adjacency.adjacency_from_polygons(features, "NAME", method="touches")`,
    which needs `pip install shapely` and allows a small gap, and pass the
    result as `edges=`.

## The season

`flu_months` sets which calendar months are flu season. The default is
October–March. A Southern-Hemisphere city passes its own, for example
`flu_months=(4, 5, 6, 7, 8, 9)`. The backtest scores whole seasons that open four
months before flu season starts (June–May for an October season), so the
off-season run-up is included.

## Optional inputs

| input | what it is | how |
|---|---|---|
| **weather** | weekly temperature, humidity, rain and wind per area | `city.fetch_weather()` downloads it from Open-Meteo (free, no key, cached), or pass `weather="weather.csv"` |
| **anchors** | neighbouring towns or water that border your areas but have no data; they tell the model where the city's edge is | `anchors=["North Township", "Lake"]`; their borders come from the boundary file or edge list |
| **static characteristics** | per-area numbers such as population density | `static="areas.csv"` with an `area` column; off by default, because they didn't help in the paper |

## Checks the package runs

When you build a city, the package refuses data it can't use and says what to
fix:

```text
DataError: Edges mention ['Old Twon'], which are not columns of the rates. Fix the
spelling, or list boundary nodes without data in anchors=[...].
```

It also catches:
- negative rates
- duplicate weeks
- dates on mixed weekdays
- fewer than two years of data
- an anchor with no borders
- missing weather or static values for an area

It warns, without stopping, about areas missing more than half their weeks and
areas with no neighbours.

Whole weeks absent from the file are added back as blank weeks, and
`city.summary()` says how many.
