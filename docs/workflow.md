# Backtest and forecast

There are two calls. Run the backtest first; the forecast reads it.

## Backtest: would it have worked here?

```python
bt = af.backtest(city)      # last 2 complete seasons, 1-4 weeks ahead, all three models
bt.report()
```

For each past season, every model trains only on the weeks before that season,
then forecasts every week of it, just as it would have in real time.
Persistence and a "same week last year" baseline are always included for
comparison.

| argument | default | meaning |
|---|---|---|
| `models` | `("gnn_st", "lstm", "colagnn")` | preset names or config objects ([Model settings](api/configs.md)) |
| `seasons` | `2` | the last *n* complete seasons, or season years such as `[2023, 2024]` |
| `horizons` | `(1, 2, 3, 4)` | weeks ahead |
| `configs` | `None` | per-model overrides, e.g. `{"gnn_st": {"n_seeds": 3}}` |
| `tune` | `False` | pick each model's settings from 12 candidates on validation weeks only, as the paper did; about 12× slower |
| `init_from` | `None` | also backtest a saved model from another city ([Short history](transfer.md)) |

**Saved to `areaflu_runs/<city>/backtest_<time>/`:**
- `predictions.csv`: every forecast, with quantiles
- `comparison.csv`: each model against persistence
- `scores.csv`: per season and horizon
- `timing.csv`: how far behind the curve each model runs
- `report.md`: the plain-language report

`af.BacktestResult.load(path)` reopens a saved backtest.

## Forecast: the next weeks

```python
fc = af.forecast(city)                 # 1-4 weeks after the last observed week
fc.summary()                           # median and 95% band per model, horizon, area
fc.to_csv("forecast.csv")              # plus all 23 FluSight quantiles
fc.to_flusight("gnn_st")               # FluSight hub layout
fc.plot("gnn_st")                      # needs the [plot] extra
```

Every model trains on all weeks whose outcome is already known, then forecasts
from your last observed week.

**The trust check.** `forecast()` finds the most recent backtest of the
city's data in `out_dir` and adds a `trust` column per model and horizon. A
saved backtest counts only if it was run on the same areas and the same weekly
rates up to the end of its last season. Weeks added since don't matter, so one
backtest keeps vouching for your weekly forecasts until past data changes.

| trust | meaning |
|---|---|
| better than persistence | lower WIS, and the gap is bigger than week-to-week noise |
| not clearly better than persistence | lower WIS on average, but the gap could be noise |
| no better / worse than persistence | don't use this model at this horizon |
| not backtested | no evidence either way; run a backtest that includes this horizon |
| backtested with other settings | the backtest ran this model differently (other settings, tuning, or pretrained use), so its verdict doesn't apply |

Anything other than "better than persistence" also raises a `TrustWarning`.

Pass `backtest=None` to skip the check, or `backtest=bt` to use a particular
backtest. With `out_dir=None`, nothing is looked up. `save_models=True` writes each trained model to the run folder.

## How long it takes

Everything runs single-threaded on a CPU. For a city of about 15 areas and 8
years of weekly data:

| step | time |
|---|---|
| one graph-model ensemble member | ~3 min (≈11 min for 60 areas) |
| default 10-member graph model, one horizon | ~30 min |
| LSTM or Cola-GNN, one horizon | a few minutes |
| default backtest (2 seasons × 4 horizons) | 4–5 hours |
| default forecast (4 horizons) | 2–3 hours |

**To go faster:**
- Use a smaller ensemble: `configs={"gnn_st": {"n_seeds": 3}}`. Fewer members
  are noisier, so expect less stable numbers.
- Ask for fewer horizons.
- Run separate horizons or cities in parallel. Each run uses one core.
- On a laptop, keep it plugged in and awake: sleep pauses the run.

## Settings

Every model setting can be overridden through `configs`; see
[Model settings](api/configs.md) for the full list. Common ones:

```python
af.backtest(city, configs={
    "gnn_st": {"n_seeds": 5, "use_weather": False},   # smaller ensemble, no weather
    "lstm":   {"lookback": 16, "target": "blend"},
})
```

Extra presets you can add to `models=`:
- `"gnn_st_nograph"`: the graph model with the map removed, to see what the map
  adds on your data
- `"gnn_st_paper"`: the paper's exact setup, which needs static characteristics
- `"seasonal_naive"`
