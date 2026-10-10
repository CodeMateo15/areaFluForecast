# Short history? Start from another city

Learned models need history. With only two or three years of data, a model
trained on a city with more history can stand in.

**Why this can work.** The transferable version of the graph model,
`gnn_st_noadapt`, has no city-specific learned links. Its weights don't depend
on how many areas a city has, so it can be applied anywhere.

In the paper, a model trained on Boston and applied to Columbus with no
Columbus training matched a model trained on Columbus itself, and beat
persistence. The reverse direction, from Columbus (a third of the history) to
Boston, was worse at 2 and 4 weeks.

## Backtest it first

```python
bt = af.backtest(my_city, models=["gnn_st_noadapt"], seasons=[2025],
                 init_from="boston_gnn_st_noadapt_h1.pt", zero_shot=True, horizons=(1,))
bt.comparison
```

The saved model appears in the report as **`pretrained`**, alongside a model
trained only on your data and persistence. `zero_shot=True` applies it with no
local training. `zero_shot=False` *fine-tunes* it instead: it starts from the
saved weights and keeps training on your weeks.

## Then forecast with it

```python
fc = af.forecast(my_city, models=[], horizons=(1,), init_from="boston_gnn_st_noadapt_h1.pt",
                 zero_shot=True, backtest=bt)
```

The forecast's trust label comes from the `pretrained` row of that backtest.

## Where the model comes from

- **Your own:** train in a city with more history using `save_models=True`:

    ```python
    source = af.resolve("gnn_st_noadapt")
    af.forecast(big_city, models=[source], save_models=True)   # writes models/*.pt
    ```

- **Pretrained Boston weights:** one file per horizon, 1-4 weeks, published
  with the first release.

**Inputs must match.** A saved model expects the same inputs it was trained
with. The Boston weights use weather and no static characteristics, so call
`city.fetch_weather()` first. If the inputs don't match, the package stops and
says which weights couldn't be loaded, instead of silently running a partly
random model.

[Example 3](examples/03_short_history_transfer.ipynb) walks through the whole
thing on a made-up city with 2.3 years of data.
