# Changelog

All notable changes to areaFluForecast are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) as described in the
[versioning policy](docs/versioning.md).

## [Unreleased]

### Added
- `explain(city, models_dir)` splits a frozen graph model's forecast, per area,
  horizon and week, into what moved it away from a usual week at that time of
  year: this week's reading, the three weeks before, earlier weeks, weather, the
  seasonal curve, and each other area (what spills over through the graph).
  Values are Owen values in rate units that add up exactly to the forecast,
  with their sampling error; each area takes about a second.
- `scripts/live_forecast.py explain --models-dir` writes `drivers.csv` beside
  the newest archived forecast, reusing the weeks an earlier folder already
  explained with the same models, plus the gap to the published (MC-Dropout)
  median as its own row.

### Changed
- `forecast(save_models=True)` and `forecast(frozen=)` now cover LSTM, Cola-GNN
  and ARIMA as well as the graph models: the sequence models load their saved
  weights, and ARIMA keeps its saved orders (no order search) and refits on the
  new weeks. `live_forecast.py train` can add models to a folder that already
  holds others, so a weekly run with `--models-dir` trains nothing.

## [1.2.0] - 2026-10-18

### Added
- A [References](docs/references.md) page listing the papers whose methods the
  package implements, with DOIs and what each is used for. The same papers are
  in `CITATION.cff` under `references`, the core ones in the README, and the
  modules that implement each method cite it in their docstrings.
- `forecast(anchor=...)` trains on the weeks before a date (e.g. the season
  start) instead of before the first forecast week, as `backtest()` does for
  that season; with `save_models=True` the saved files also record the anchor
  and a fingerprint of the rates before it.
- `forecast(frozen=folder)` loads each graph model from those files instead of
  training it. The forecast is exactly the one retraining with the same anchor
  would give (same split, scaling, seasonal baseline and intervals), in seconds
  rather than hours. It refuses files from another city, other settings, or
  rates that were revised before the anchor.
- `scripts/live_forecast.py train` saves the season-start graph models;
  `run`/`forecast-part`/`assemble --models-dir` load them, and `run --jobs K`
  forecasts the horizons in K processes.

## [1.1.0] - 2026-10-13

### Added
- `arima` preset (`ArimaConfig`): one seasonal ARIMA per area, the order chosen
  by walk-forward validation error at the forecast horizon, with the annual
  cycle as 3 sin/cos regressors. A port of the research `run_arima.py`; on
  Boston 2025-26 its WIS matches the paper's within 0.7% at 1, 2 and 4 weeks.
  It is in `DEFAULT_MODELS`, and `statsmodels` is now a dependency.

### Changed
- ARIMA reports progress once per area, so long runs show a live percentage.
- `scripts/live_forecast.py assemble` stops with a clear message when saved
  backtest parts were made before a model was added, instead of combining them.

## [1.0.0] - 2026-10-12

The first stable release. The public API, the command line and the saved-file
formats are now covered by the [versioning policy](docs/versioning.md). The
changes below are relative to 0.1.1; the breaking ones are listed first.

### Changed (breaking)
- `City`, `BacktestResult` and `ForecastResult` take keyword arguments only.
- `City.from_frames` takes explicit `flu_months`, `centroids`, `weather`,
  `static` and `label` arguments instead of passing `**kwargs` through.
- Errors in user input raise `DataError` (a `ValueError`) everywhere: bad
  boundary files and edge lists (these raised `KeyError` or `ValueError`),
  weather CSVs, too little history, an empty last week, and a pretrained model
  whose inputs don't match the city.
- Data-quality warnings use the new `DataWarning` category instead of
  `UserWarning`. `TrustWarning` is still exported as `af.TrustWarning`.
- Models run on the CPU unless `device=` says otherwise. Before, a GPU was used
  whenever one was present.
- A single saved model is used only for the horizon it was trained for. Before,
  `init_from` applied the same checkpoint at every horizon, so a 1-week model
  silently made the 4-week forecast. Pass `{horizon: checkpoint}` for several.
- A forecast's trust verdict carries over from a backtest only when the
  backtest ran on the same data (the same areas and the same rates up to the end
  of its last season) and the same settings, tuning, and pretrained file and
  mode. Otherwise the trust label reads "backtested with other settings" and a
  `TrustWarning` is raised. Before, backtests were matched by city name alone,
  so two datasets that shared a name shared verdicts. Seeds don't count as
  settings.
- `forecast(out_dir=None)` no longer looks for backtests in `./areaflu_runs`.
- Checkpoints are opened with `torch.load(weights_only=True)`, which cannot run
  code. Research checkpoints that need full unpickling load only with
  `Checkpoint.load(path, trust_research_file=True)`.
- Command line:
  - `--seasons N` only counts seasons. Use `--season-years 2023 2024` for
    specific seasons.
  - Errors print one line and exit with code 2 (wrong options) or 3 (data
    problems), instead of a traceback.
  - Paths in a `--city` JSON file resolve relative to that file.

### Added
- `af.pretrained("boston")`: transferable graph models trained on Boston, one
  per horizon (1-4 weeks), bundled with the package. Pass them as
  `init_from=af.pretrained("boston")`, or `--pretrained boston` on the command
  line.
- `af.Checkpoint`: a saved model, with `Checkpoint.load(path)` and
  `checkpoint.save(path)`, its horizon, input features, source city, and the
  package and PyTorch versions that saved it.
- A saved model's input features are checked against the city's. A mismatch is
  an error with `zero_shot=True` and a warning when fine-tuning.
- `seed=` on `backtest()` and `forecast()` sets every model's random seeds from
  one number.
- `ForecastResult.save` and `ForecastResult.load`. Forecast runs also save
  `history.csv` and `forecast.json`.
- Saved city, backtest and forecast folders record a format version and the
  package version. A newer format is refused with "upgrade the package".
  Folders saved by 0.1 still load.
- `af.combine`, `af.BaselineConfig`, `af.ModelConfig`, `af.DataWarning`, and
  the `af.adjacency`, `af.scoring`, `af.timing` and `af.weather` submodules are
  public.
- Command line: `--version`, `--pretrained`, `--init-from`, `--zero-shot`,
  `--seed`, `--device`, `--no-progress`, `--save-models`, `--backtest`,
  `--centroids` and the column-name options, plus help text for every option.
- Type hints on every public function and class, and a `py.typed` marker.
- A [versioning and stability](docs/versioning.md) policy. It marks `refit`,
  `drop_edge`, `corr_binary` and `two_sided_intervals` as experimental.
- Release automation (`RELEASING.md`): pushing a version bump to `main` tests
  the built wheel, publishes to PyPI by trusted publishing, tags the commit and
  creates the GitHub release that Zenodo archives.
- CI covers Python 3.10-3.13 on Linux and macOS, the oldest dependency versions
  `pyproject.toml` allows, and the installed wheel. Every Monday it reruns
  against the newest dependencies together with a live Open-Meteo test.

### Fixed
- Training no longer changes the caller's session: Python's, NumPy's and
  PyTorch's global random generators and PyTorch's thread count are restored
  afterwards, and `PYTHONHASHSEED` is no longer written.
- A weather request the server rejects (HTTP 4xx other than 429) fails at once
  instead of retrying for about four minutes.
- Saving a backtest with an empty timing table no longer writes a file that
  can't be read back.
- `forecast` on the command line no longer prints each trust note twice.
- The source distribution includes `tests/conftest.py`, so its tests run.

### Deprecated
- `af.save_checkpoint` and `af.load_checkpoint`; use `af.Checkpoint`. They
  still work, with a `DeprecationWarning`, and will be removed in 2.0.

## [0.1.1] - 2026-10-10

- Documentation and packaging fixes.

## [0.1.0] - 2026-10-10

- First release.
