# areaFluForecast

**Documentation: https://areafluforecast.readthedocs.io** (available once the repository is
public and imported on Read the Docs; build it locally with
`pip install -r docs/requirements.txt && mkdocs serve`).

Neighborhood-scale influenza forecasts, 1 to 4 weeks ahead, for public health
departments. You bring weekly flu rates for your areas (neighborhoods, ZIP-code
groups, districts) and their boundaries. The package trains three models, checks
each against the simplest possible forecast on your own past seasons, and only
then gives you forecasts, with honest uncertainty bands.

The three models:
- **`gnn_st`**: a graph neural network that uses which areas border each other
  and whose flu curves move together. This is the model from the accompanying paper.
- **`lstm`**: a recurrent network that sees every area's series but no map.
- **`colagnn`**: Cola-GNN (Deng et al. 2020), a published graph model for flu.
  This is our reimplementation from the paper, not the authors' code.

Every model is compared with **persistence**, "next week looks like this week".
A forecaster that can't beat that on your data has learned nothing you can use.
The package tells you, for each model and horizon, whether it did.

> **What the evidence says.** In Boston the graph model's error was 18–28% below
> persistence on seasons it never saw. It tied a carefully tuned LSTM at 1 week
> and led it on average at 2–4 weeks. In Buenos Aires it did no better than
> persistence. Your city may be either, and the backtest is how you find out.

## Install

```bash
pip install areaFluForecast            # once published; until then:
pip install git+https://github.com/CodeMateo15/areaFluForecast
pip install "areaFluForecast[plot]"    # adds plotting
```

Python 3.10+, CPU only. Installing the package also installs PyTorch and
PyTorch Geometric.

## Try it on Boston

```python
import areafluforecast as af

city = af.load_sample("boston")
print(city.summary())
bt = af.backtest(city, seasons=1, horizons=(1, 2), configs={"gnn_st": {"n_seeds": 3}})
fc = af.forecast(city, horizons=(1, 2))
fc.summary()
```

Or from the shell: `areaflu backtest --sample boston --seasons 1 --quick`.

**Worked examples:** [`examples/`](examples/) has five notebooks, saved with
their outputs:
- a Boston quickstart
- bringing your own files
- a city with only two years of data
- reading the results and choosing a model
- a real city (Chicago) built from public open data

## Use your own data

You need two things.

**1. Weekly rates**, as a CSV. Either long format:

| week | area | rate |
|---|---|---|
| 2024-01-07 | Riverside | 41.2 |
| 2024-01-07 | Old Town | 28.9 |

or wide format, with one column per area and the week start date first. Rates
per 100,000 residents are expected. Leave missing or suppressed weeks blank.
Don't fill them with 0. You need at least two years of data, and three or more
is better. If you have counts, `af.rates_from_counts(counts, population)`
converts them.

**2. The map**, given one of two ways:
- a GeoJSON boundary file whose features carry the area name (`id_col`). The
  package works out which areas share a border.
- or a CSV edge list with one row per pair of bordering areas.

```python
city = af.City.from_files("rates.csv", areas="areas.geojson", id_col="NAME",
                          flu_months=(10, 11, 12, 1, 2, 3))
city.fetch_weather()          # optional: Open-Meteo, free, no API key
print(city.summary())         # check this before training
```

Southern-Hemisphere cities pass their own flu months, e.g. `(4, 5, 6, 7, 8, 9)`.

**Optional inputs:**
- **Anchors** are neighbouring towns or water that border your areas but have
  no data. List them in `anchors=[...]`, and the edge list or boundary file
  supplies their borders.
- **Static area characteristics** (a CSV with an `area` column). They are off
  by default, because in the paper they did not help.

## The workflow

```python
bt = af.backtest(city)        # would it have worked here? (last 2 complete seasons)
bt.report()                   # each model vs persistence, per horizon
fc = af.forecast(city)        # weeks 1-4 after your last observed week
fc.to_csv("forecast.csv")     # median, 95% band, 23 FluSight quantiles per area
fc.to_flusight()              # FluSight hub layout
fc.plot("gnn_st")
```

`forecast()` reads the most recent backtest for the city. It adds a `trust`
column and raises a `TrustWarning` for any model or horizon that did not
clearly beat persistence, or was never backtested.

How to read the backtest report:
- **WIS** (weighted interval score) asks how good the whole forecast was, the
  band as well as the point. Lower is better, and it's in rate units. CDC's
  FluSight ranks forecasts by it.
- **Relative WIS** is the model's WIS divided by persistence's, on the same
  weeks. Below 1 means the model beat persistence.
- **p (Holm)** asks whether the weekly gap could be week-to-week noise. It is
  tested over weeks, not area-weeks, because neighbouring areas move together,
  and it is corrected for the number of comparisons. "Better than persistence"
  means a relative WIS below 1 and p below 0.05.
- **Coverage** is how often the truth fell inside the 95% band. It should be
  near 95%.

## How long it takes

On a laptop CPU, for about 15 areas and 8 years of weekly data:
- One graph-model ensemble member takes about 3 minutes, so the default
  10-member ensemble takes about 30 minutes per horizon.
- The LSTM and Cola-GNN take a few minutes per horizon.
- A default backtest (2 seasons × 4 horizons) takes about 4–5 hours.
- A forecast takes about 2–3 hours.

To go faster, use `configs={"gnn_st": {"n_seeds": 3}}`, fewer horizons, or
`--quick` to check the pipeline end to end. Everything is single-threaded, so
separate horizons or cities can run in parallel. `tune=True` picks each model's
settings from 12 candidates on validation weeks, the way the paper did, and
takes about 12× longer.

## Starting from Boston's model

A department with only a season or two of data can start from a model trained
elsewhere:

```python
fc = af.forecast(city, init_from="boston_gnn_st_noadapt_h1.pt", zero_shot=True, horizons=(1,))
```

In the paper, a Boston model applied to Columbus with no Columbus training tied
a model trained on Columbus itself and beat persistence. The reverse direction
(Columbus to Boston, with a third of the history) was worse at 2 and 4 weeks.

Limits:
- **The inputs must match.** The pretrained model uses weather and no static
  characteristics, so call `city.fetch_weather()` first.
- **Backtest it on your data before trusting it.** Pretrained Boston weights
  are published with the release.

## Limits to know

- **The forecasts run late.** Every model here, like persistence, rises after
  the curve rises and peaks after it peaks. It runs about *h* weeks behind at
  *h* weeks ahead. The models beat persistence by getting the size of the
  epidemic right, not by seeing turns earlier. This happens because the models
  only see past rates, and a turn isn't visible in weekly data until after it
  happens. A growth-rate feature and a median loss were tested and didn't
  change it. The backtest report has a **timing** section that measures this
  on your data. Earlier warning would need inputs that move before ED visits
  do, such as wastewater or a shorter reporting delay.
- **Reporting delays are not simulated.** A 1-week forecast assumes last week's
  data is complete. If your data arrives L weeks late, treat the h-week forecast
  as roughly an (h + L)-week one.
- **The backtest scores the models, not your data pipeline.** If your counts get
  revised after the fact, the backtest saw final numbers and your real-time
  forecasts will not.
- **One season is one season.** Two or more backtest seasons are better than one.

## Reproducing the paper

`scripts/parity/` reruns the paper's tuned Boston backtest with this package and
compares the forecasts week by week. `tests/test_research_parity.py` checks,
against the research code, that the inputs, graph, initial weights and trained
forecasts are identical, down to the last bit for training.

## Data availability and credits

- **The bundled Boston sample** comes from the Boston Public Health Commission's
  public dashboard.
- **Weather data** is by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).
- **The bundled Chicago sample** (`load_sample("chicago")`) comes from the Chicago
  Department of Public Health's "Influenza Risk Level by ZIP Code" on the City of
  Chicago Data Portal. It is a 1-10 activity level, not a rate; its README carries
  the portal's required disclaimer.
- **Nothing else is bundled.** The package contains no data from any other
  health department.
- **Paper data.** The Columbus and Buenos Aires data used in the accompanying
  paper are described in its data availability statement. Columbus weekly
  area-level counts are public in the
  [research repository](https://github.com/CodeMateo15/GNN-Influenza-Boston).
  Columbus Public Health's visit-level extracts are available from them on
  request.
- **Your data stays with you.** The package sends nothing anywhere. Its only
  network call is the optional weather fetch, which sends area centroids to
  Open-Meteo.
- This work was supported by Boston Public Health Commission, Columbus Public
  Health, the PEAK Trailblazer fellowship, and Prof. Ariel Hamlin (Northeastern
  University).

## Citing

See `CITATION.cff`.
