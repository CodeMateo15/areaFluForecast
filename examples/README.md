# Examples

Each notebook is saved with its outputs, so you can read the results on GitHub
without running anything. They run in order from this folder. Notebook 4 reuses
notebook 1's backtest, and runs a small one if it's missing.

| notebook | what it shows | data | run time* |
|---|---|---|---|
| [01_quickstart_boston](01_quickstart_boston.ipynb) | the whole workflow: data, backtest against persistence, forecast with a trust check, FluSight export | bundled Boston sample (real, BPHC) | ~75 min |
| [02_your_own_data](02_your_own_data.ipynb) | file formats, building the map from a boundary file, anchors, checking the map, common errors, counts to rates | Riverton (synthetic) | ~35 min |
| [03_short_history_transfer](03_short_history_transfer.ipynb) | a city with only 2 years of data: backtesting and using a model trained elsewhere | Lakeside + Riverton (synthetic) | ~15 min |
| [04_reading_the_results](04_reading_the_results.ipynb) | relative WIS, band coverage, calibration (PIT), errors over the season, whether forecasts run late, choosing which model to use | notebook 1's backtest | ~1 min |
| [05_real_city_chicago](05_real_city_chicago.ipynb) | a real city from two public downloads: weekly ILI activity by ZIP plus ZIP boundaries, weather, backtest with timing, forecast | Chicago open data (real, CDPH / IDPH ESSENCE) | ~75 min |

\* On a laptop CPU, with the 2-member ensembles the notebooks use for speed. The
default is 10 members.

**Riverton and Lakeside are made up.** `data/make_demo_cities.py` generates them.
Their epidemics spread along district borders by construction, so they show the
formats and the mechanics but say nothing about how the models do on real data.
The Boston and Chicago notebooks use real surveillance data. Chicago's is a 1–10
activity level per ZIP code, not a rate; the notebook explains what that means.

**To regenerate the notebooks after an API change:**

```bash
python scripts/build_example_notebooks.py          # rewrite the cells
python scripts/run_example_notebooks.py --quick    # dry run with 2-epoch models
python scripts/run_example_notebooks.py            # execute and save outputs
```
