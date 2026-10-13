# Model settings

Every model is described by a settings object. Pass a preset name such as
`"gnn_st"` to `backtest`/`forecast`, override individual settings with
`configs={"gnn_st": {"n_seeds": 3}}`, or build a config with `resolve`.

**Presets:**

| name | model |
|---|---|
| `gnn_st` | the graph model, the package default; uses weather when the city has it, no static characteristics |
| `gnn_st_paper` | the paper's exact graph-model configuration |
| `gnn_st_nograph` | the graph model with the map removed, to measure what the map adds |
| `gnn_st_noadapt` | the graph model without city-specific learned links; transferable between cities |
| `lstm` | multivariate LSTM |
| `colagnn` | Cola-GNN |
| `arima` | seasonal ARIMA, one per area (needs no training; uses `statsmodels`) |
| `persistence`, `seasonal_naive` | baselines, always included in backtests |

**Experimental settings.** A few `GNNConfig` and interval settings come from
research ablations and are not covered by the versioning policy. They may
change or be removed in a minor release: `refit`, `drop_edge`, `corr_binary`
and `two_sided_intervals`. Every other setting is stable.

::: areafluforecast.configs.resolve

::: areafluforecast.configs.GNNConfig

::: areafluforecast.configs.LSTMConfig

::: areafluforecast.configs.ColaGNNConfig

::: areafluforecast.configs.ArimaConfig

::: areafluforecast.configs.BaselineConfig
