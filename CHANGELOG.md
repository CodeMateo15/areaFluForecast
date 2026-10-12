# Changelog

All notable changes to areaFluForecast are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may
change the API.

## [1.0.0rc1] - unreleased

The release candidate for 1.0: the API and file formats are what 1.0 will
ship, as covered by the [versioning policy](docs/versioning.md).

### Removed
- `af.save_checkpoint` and `af.load_checkpoint`, deprecated in 0.3.0. Use
  `af.Checkpoint.load(path)` and `checkpoint.save(path)`.

### Added
- Type hints on every public function and class, and a `py.typed` marker.
- The version lives in one place (`areafluforecast.__version__`). Tests check
  that `CITATION.cff` and this changelog match it.
- Release automation (`RELEASING.md`): tagging builds the package, tests the
  built wheel on Linux and macOS, publishes to PyPI by trusted publishing, and
  creates the GitHub release that Zenodo archives.
- CI tests Python 3.10-3.13 on Linux and macOS, the oldest dependency versions
  `pyproject.toml` allows, and the installed wheel rather than the checkout.
  It runs ruff, and every Monday it reruns against the newest dependencies
  together with a live Open-Meteo test.
- Package metadata: Python version, `Development Status :: 5 -
  Production/Stable` and `Typing :: Typed` classifiers, and a changelog link.

## [0.3.0] - unreleased

This release settles the API ahead of 1.0. The breaking changes are listed
first.

### Changed (breaking)
- `City`, `BacktestResult` and `ForecastResult` take keyword arguments only.
- `City.from_frames` takes explicit `flu_months`, `centroids`, `weather`,
  `static` and `label` arguments instead of passing `**kwargs` through.
- Errors in user input now raise `DataError` (a `ValueError`) everywhere: bad
  boundary files and edge lists (these raised `KeyError` or `ValueError`),
  weather CSVs, too little history, an empty last week, and a pretrained model
  whose inputs don't match the city.
- Data-quality warnings use the new `DataWarning` category instead of
  `UserWarning`. `TrustWarning` moved to `areafluforecast.errors` and is still
  exported as `af.TrustWarning`.
- Models run on the CPU unless `device=` says otherwise. Before, a GPU was used
  whenever one was present.
- Command line:
  - `--seasons N` only counts seasons. Use `--season-years 2023 2024` for
    specific seasons.
  - Errors print one line and exit with code 2 (wrong options) or 3 (data
    problems), instead of a traceback.
  - Paths in a `--city` JSON file resolve relative to that file.

### Added
- `af.pretrained("boston")`: transferable graph models trained on Boston, one
  per horizon (1-4 weeks), bundled with the package. Pass them as
  `init_from=af.pretrained("boston")`.
- `af.Checkpoint`: a saved model, with `Checkpoint.load(path)` and
  `checkpoint.save(path)`, its horizon, input features, source city and
  package version. `init_from` accepts `Checkpoint` objects.
- `seed=` on `backtest()` and `forecast()` sets every model's random seeds from
  one number.
- `ForecastResult.save` and `ForecastResult.load`. Forecast runs now also save
  `history.csv` and `forecast.json`.
- Saved city, backtest and forecast folders record a format version and the
  package version. A newer format is refused with "upgrade the package".
- `af.combine`, `af.BaselineConfig`, `af.ModelConfig`, `af.DataWarning`, and
  the `af.adjacency`, `af.scoring`, `af.timing` and `af.weather` submodules are
  public.
- Command line: `--version`, `--pretrained`, `--init-from`, `--zero-shot`,
  `--seed`, `--device`, `--no-progress`, `--save-models`, `--backtest`,
  `--centroids` and the column-name options, plus help text for every option.
- A [versioning and stability](docs/versioning.md) policy, which marks
  `refit`, `drop_edge`, `corr_binary` and `two_sided_intervals` as
  experimental.

### Fixed
- Training no longer changes the caller's session: Python's, NumPy's and
  PyTorch's global random generators and PyTorch's thread count are restored
  afterwards, and `PYTHONHASHSEED` is no longer written.
- A weather request the server rejects (HTTP 4xx other than 429) fails at once
  instead of retrying for about four minutes.
- The trust check ignores seed differences: a backtest run with one seed
  vouches for a forecast run with another.
- `forecast` on the command line no longer prints each trust note twice.

### Deprecated
- `af.save_checkpoint` and `af.load_checkpoint`; use `Checkpoint`. They will be
  removed in 1.0.

## [0.2.0] - unreleased

### Fixed
- A saved model is no longer reused across horizons. Before, `forecast()` and
  `backtest()` with `init_from` applied the same checkpoint at every horizon, so
  a 1-week model silently made the 4-week forecast. A single file now serves
  only the horizon it was trained for; pass `init_from={1: "h1.pt", 2: "h2.pt", ...}`
  for several.
- The trust check no longer matches a backtest by city name alone. A saved
  backtest now counts only if it ran on the same areas and the same rates up to
  the end of its last season, so two datasets that share a name (every
  `rates.csv` is called "rates") no longer share verdicts.
- `forecast(out_dir=None)` no longer looks for backtests in `./areaflu_runs`.
- Saving a backtest with an empty timing table no longer writes a file that
  can't be read back.
- The source distribution now includes `tests/conftest.py`, so its tests run.

### Changed
- Checkpoints are opened with `torch.load(weights_only=True)`, which cannot run
  code. Research checkpoints that need full unpickling load only with
  `load_checkpoint(path, trust_research_file=True)`.
- A checkpoint saved in a newer format now raises "upgrade the package" instead
  of being read as a research file with default settings.
- A forecast's verdict now carries over from the backtest only for the same
  settings, tuning, and pretrained file and mode (zero-shot or fine-tuned).
  Otherwise its trust reads "backtested with other settings" and it raises a
  `TrustWarning`.
- A saved model's input features are checked against the city's: a mismatch
  is an error with `zero_shot=True` and a warning when fine-tuning. Fine-tuning
  also now warns when some weights could not be loaded.

### Added
- Checkpoints record the package and PyTorch versions.
- Saved backtests record a data fingerprint, the settings each model ran with,
  and the pretrained files used. Backtests saved by 0.1 are still used when
  nothing newer matches, with a note that they can't be checked.

## [0.1.1] - 2026-10-10

- Documentation and packaging fixes.

## [0.1.0] - 2026-10-10

- First release.
