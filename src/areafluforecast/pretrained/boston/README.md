# Pretrained Boston models

`af.pretrained("boston")` loads these four files, one per horizon (1-4 weeks
ahead).

**Model.** `gnn_st_noadapt`, the transferable graph model. It has no learned
adjacency, so it fits a city with any number of areas. It uses the package
default settings and an ensemble of 10 seeds.

**Inputs it expects.**
- weekly rates per 100,000 residents
- the six Open-Meteo weather columns (call `city.fetch_weather()` first)
- no static characteristics

The package checks a city's input features against the model's and stops
(zero-shot) or warns (fine-tuning) if they differ.

**Training data.** Every Boston week through 2026-05-03: 14 neighborhoods plus
7 anchors, from Boston Public Health Commission emergency-department data,
with weather from Open-Meteo (CC BY 4.0). The models were trained on
2026-10-02 by `scripts/pretrained/train_pretrained.py` in the source
repository.

**How well it transfers.** Applied with no local training to Columbus's
2024-25 season, its weighted interval score relative to persistence was:

| weeks ahead | relative WIS | Holm-adjusted p |
|---|---|---|
| 1 | 0.86 | 0.048 |
| 2 | 0.93 | 1.0 |
| 3 | 0.92 | 1.0 |
| 4 | 0.82 | 0.89 |

It was clearly better than persistence only at 1 week; at 2-4 weeks it was
better on average, but not clearly. These weights saw Boston data from after
that season, so this is a sanity check of the weights, not a clean out-of-time
test. The paper's transfer experiment is the evidence.

**Before relying on it,** backtest it on your own data:

```python
weights = af.pretrained("boston")
bt = af.backtest(city, models=["gnn_st_noadapt"], init_from=weights, zero_shot=True)
```

`forecast(..., init_from=weights, zero_shot=True)` then carries that verdict
over.
