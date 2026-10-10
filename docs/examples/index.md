# Examples

Five notebooks, shown here with the outputs from a real run. To run them
yourself, download them from the repository's
[`examples/`](https://github.com/CodeMateo15/areaFluForecast/tree/main/examples)
folder.

| notebook | what it shows | data |
|---|---|---|
| [1. Boston quickstart](01_quickstart_boston.ipynb) | the whole workflow: data, backtest against persistence, forecast with a trust check, FluSight export | Boston sample (real) |
| [2. Your own data](02_your_own_data.ipynb) | file formats, building the map from a boundary file, anchors, checking the map, common errors | Riverton (synthetic) |
| [3. Short history and transfer](03_short_history_transfer.ipynb) | a city with 2.3 years of data, using a model trained elsewhere | Lakeside and Riverton (synthetic) |
| [4. Reading the results](04_reading_the_results.ipynb) | relative WIS, band coverage, calibration, timing, choosing a model | notebook 1's backtest |
| [5. A real city: Chicago](05_real_city_chicago.ipynb) | a city built from two public downloads, with weather, backtest and forecast | Chicago open data (real) |

!!! note "About the numbers"
    - **Small ensembles:** to keep run times down, the notebooks use 1–2
      ensemble members instead of the default 10, without tuning. Their numbers
      are noisier than a full run.
    - **Synthetic cities:** Riverton and Lakeside are made up, and their
      epidemics spread along borders by construction. They show the mechanics,
      not how the models do on real data.
