# Reproducing the paper

The package runs the same models the paper evaluated. It was checked against
the research code ([GNN-Influenza-Boston](https://github.com/CodeMateo15/GNN-Influenza-Boston))
in two ways.

## Identical computation

`tests/test_research_parity.py` runs the package and the research code side by
side on Boston's 2022-23 season and checks that these are identical:
- the model inputs and the graph
- the starting weights
- the weights after training and the forecasts, which match exactly, for the
  graph model, the LSTM and Cola-GNN
- forecasts, 95% bands and all 23 quantiles from a full research run, which
  match to within 10⁻⁶

```bash
AREAFLU_RESEARCH_REPO=~/GNN-Influenza-Boston pytest -m research
```

## The paper's backtest, rerun with the package

`scripts/parity/` reruns the paper's tuned Boston backtest (3 seasons × 3
horizons × 3 models, 10 seeds each, with the paper's tuned settings) and
compares the forecasts week by week.

Result: all 27 combinations match the paper.
- 23 are numerically identical.
- The other 4 differ by at most 0.4% in WIS, probably from float rounding
  across processor types.

## Settings that match the paper

- `gnn_st_paper` is the paper's exact graph-model configuration. It uses static
  demographic characteristics and similarity edges.
- The package default, `gnn_st`, leaves those out, because the paper's
  ablations found them no help at 1–2 weeks and harmful at 4.
- The tuning search spaces and the 12 candidate settings are the paper's own.
