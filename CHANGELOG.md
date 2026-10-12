# Changelog

All notable changes to areaFluForecast are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may
change the API.

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
