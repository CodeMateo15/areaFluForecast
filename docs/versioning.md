# Versioning and stability

areaFluForecast follows [Semantic Versioning](https://semver.org/). From 1.0:

- **Patch releases** (1.0.x) fix bugs. They don't change results except to fix
  a bug, and the changelog says when they do.
- **Minor releases** (1.x) add features. Code that worked on 1.0 keeps working.
- **Major releases** (2.0) may break things. Each break is listed in the
  changelog with what to change.

Before 1.0, minor releases (0.x) may change the API, and the changelog lists
every change.

## What is covered

- Everything in `areafluforecast.__all__`: `City`, `backtest`, `forecast`,
  their results, `Checkpoint`, `pretrained`, the config classes, the error and
  warning types, and the loaders.
- The public functions of the `adjacency`, `scoring`, `timing` and `weather`
  submodules.
- The `areaflu` command line: its commands, options and exit codes.
- The columns of `ForecastResult.table` and `BacktestResult.comparison`, and
  the trust labels (`better than persistence`, `not backtested`, and the rest).
- Saved files: checkpoints, and city, backtest and forecast folders. A newer
  version reads folders saved by an older one. An older version refuses a newer
  format with a message, rather than misreading it.

Not covered: modules and names starting with `_`, anything not listed above
(such as `engine`, `training` and `samples`), and the experimental settings
listed on the [model settings](api/configs.md) page.

## Deprecation

Before anything covered is removed or changed incompatibly, it gives a
`DeprecationWarning` for at least one minor release, and the changelog says
what replaces it.

## Numbers

The same version, settings, seed and data give the same forecasts on the same
machine and CPU. Different PyTorch versions or hardware can change results in
the last digits. GPU runs are not bit-reproducible.
