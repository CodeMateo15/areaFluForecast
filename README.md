# areaFluForecast

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.23327109-3B97DB.svg)](https://doi.org/10.5281/zenodo.23327109)
[![PyPI](https://img.shields.io/pypi/v/areaFluForecast.svg)](https://pypi.org/project/areaFluForecast/)
[![tests](https://github.com/CodeMateo15/areaFluForecast/actions/workflows/ci.yml/badge.svg)](https://github.com/CodeMateo15/areaFluForecast/actions/workflows/ci.yml)

**areaFluForecast** is an open-source Python package that gives public health
departments 1–4 week influenza forecasts for their neighborhoods, ZIP codes or
districts. You bring weekly flu rates and area boundaries. It trains three
models, checks each against persistence ("next week looks like this week") on
your own past seasons, and gives forecasts with calibrated uncertainty bands
and a trust label for each model.

In Boston, the graph model's error was 18–28% below persistence on seasons it
never saw. In Columbus it was 12–23% below on average, but not clearly. The
backtest tells you which kind of city yours is.

---

## Documentation

Documentation: [areafluforecast.readthedocs.io](https://areafluforecast.readthedocs.io)

PyPI: [areaFluForecast](https://pypi.org/project/areaFluForecast/)

Data sources and credits: [Data and credits](https://areafluforecast.readthedocs.io/en/latest/data-and-credits/)

---

## Install

```bash
pip install areaFluForecast             # everything except plots
pip install "areaFluForecast[plot]"     # + forecast plots
pip install "areaFluForecast[geo]"      # + messy boundary files (shapely)
```

Requires Python 3.10 or later. Runs on a laptop CPU; no GPU needed.

---

## A tour

```python
import areafluforecast as af

# Your data: weekly rates per area, plus a boundary file
city = af.City.from_files("rates.csv", areas="areas.geojson", id_col="NAME")
city.fetch_weather()                 # optional; Open-Meteo, no API key
print(city.summary())

# Would the models have worked here? Score past seasons against persistence
bt = af.backtest(city)
bt.report()

# Forecast the next 1-4 weeks, with a trust label from that backtest
fc = af.forecast(city)
fc.to_csv("forecast.csv")            # median, 95% band, 23 FluSight quantiles
fc.plot("gnn_st")

# Short history? Start from the bundled Boston models
weights = af.pretrained("boston")
fc = af.forecast(city, init_from=weights, zero_shot=True)
```

Or from the shell:

```bash
areaflu check    --rates rates.csv --areas areas.geojson --id-col NAME
areaflu backtest --rates rates.csv --areas areas.geojson --id-col NAME --weather
areaflu forecast --rates rates.csv --areas areas.geojson --id-col NAME --weather
areaflu backtest --sample boston --seasons 1 --quick      # try it on bundled data
```

---

## What's in it

| Model | What it is |
| --- | --- |
| `gnn_st` | Graph neural network over bordering areas and correlated curves; the model from the accompanying paper |
| `lstm` | Multivariate LSTM: every area's series, no map |
| `colagnn` | Cola-GNN (Deng et al. 2020), reimplemented from the paper |
| `arima` | Seasonal ARIMA per area (annual cycle as harmonic regressors), the paper's statistical baseline |
| `persistence`, `seasonal_naive` | Baselines every model is compared with |
| `af.pretrained("boston")` | Transferable graph models trained on Boston, one per horizon |

| Bundled data | Coverage |
| --- | --- |
| Boston sample | 14 neighborhoods, weekly, 2017–2026, with weather |
| Chicago sample | 58 ZIP codes, weekly activity level (1–10), 2019–2026, with weather |

---

## Modules

### City and data
`City.from_files`, `City.from_frames`, `load_sample`, `load_city`, `save_city`,
`rates_from_counts`, `read_rates_csv`

### Backtest and forecast
`backtest`, `BacktestResult`, `latest_backtest`, `combine`, `forecast`,
`ForecastResult`

Every forecast carries a `trust` label per model and horizon, such as "better
than persistence" or "not backtested". Anything short of "better" raises a
`TrustWarning`.

### Models and settings
`GNNConfig`, `LSTMConfig`, `ColaGNNConfig`, `BaselineConfig`, `PRESETS`,
`resolve`, `Checkpoint`, `pretrained`

### Scores, timing, maps, weather
`af.scoring` (WIS, interval score, PIT, Diebold-Mariano), `af.timing` (weeks
behind the curve, peak offsets), `af.adjacency` (borders from GeoJSON),
`af.weather` (Open-Meteo)

---

## Limits

- **Forecasts run late.** Like persistence, every model runs about *h* weeks
  behind the curve at *h* weeks ahead. The models win on the size of the
  epidemic, not on its timing. The backtest report measures this on your
  data.
- **Reporting delays aren't simulated.** If your data arrives *L* weeks late,
  treat an *h*-week forecast as an (*h* + *L*)-week one.
- **The backtest sees final data.** Revised counts make past accuracy look
  better than real time will be.

More in [Limits to know](https://areafluforecast.readthedocs.io/en/latest/limits/).

---

## Examples

Five notebooks, saved with their outputs, are in [`examples/`](https://github.com/CodeMateo15/areaFluForecast/tree/main/examples):
a Boston quickstart, bringing your own files, a city with two years of data,
reading the results, and a real city (Chicago) from public open data.

## Reproducing the paper

`scripts/parity/` reruns the paper's Boston backtest with this package.
`tests/test_research_parity.py` checks that the package matches the research
code bit for bit. See [Reproducing the paper](https://areafluforecast.readthedocs.io/en/latest/reproducibility/).

---

## Citing

Cite the version you used:
[10.5281/zenodo.23327109](https://doi.org/10.5281/zenodo.23327109) always
resolves to the latest version. Details are in [`CITATION.cff`](https://github.com/CodeMateo15/areaFluForecast/blob/main/CITATION.cff),
or use GitHub's **Cite this repository** button.

## References

The package implements methods from published work. The core ones:

- Cola-GNN: Deng et al. 2020, CIKM '20, [doi:10.1145/3340531.3411975](https://doi.org/10.1145/3340531.3411975)
- Graph convolution: Kipf & Welling 2017, ICLR, [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
- LSTM: Hochreiter & Schmidhuber 1997, *Neural Comput* 9:1735, [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
- Weighted interval score: Bracher et al. 2021, *PLoS Comput Biol* 17:e1008618, [doi:10.1371/journal.pcbi.1008618](https://doi.org/10.1371/journal.pcbi.1008618)
- FluSight quantiles and relative WIS: Mathis et al. 2024, *Nat Commun* 15:6289, [doi:10.1038/s41467-024-50601-9](https://doi.org/10.1038/s41467-024-50601-9)
- Diebold-Mariano test: Diebold & Mariano 1995, *J Bus Econ Stat* 13:253, [doi:10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)

The full list, with what each paper is used for, is in
[References](https://areafluforecast.readthedocs.io/en/latest/references/).

Found a bug or want a feature? Open an [issue](https://github.com/CodeMateo15/areaFluForecast/issues).

## Credits

Supported by the Boston Public Health Commission, Columbus Public Health, the
PEAK Trailblazer fellowship, and Prof. Ariel Hamlin (Northeastern University).
Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). Your
data stays with you: the only network call is the optional weather fetch.

## Contact

Mateo Biggs, biggs.m@northeastern.edu
