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

The package ships with models trained on Boston, one per horizon:

```python
weights = af.pretrained("boston")
bt = af.backtest(my_city, models=["gnn_st_noadapt"], seasons=[2025],
                 init_from=weights, zero_shot=True)
bt.comparison
```

The saved model appears in the report as **`pretrained`**, alongside a model
trained only on your data and persistence. `zero_shot=True` applies it with no
local training. `zero_shot=False` *fine-tunes* it instead: it starts from the
saved weights and keeps training on your weeks.

## Then forecast with it

```python
fc = af.forecast(my_city, models=[], init_from=weights, zero_shot=True, backtest=bt)
```

The forecast's trust label comes from the `pretrained` row of that backtest.
That verdict only carries over to the same use of the same file. If the backtest
applied the model untrained and the forecast fine-tunes it (or the other way
round), or the files differ, the label reads "backtested with other settings".
`forecast()` defaults to `zero_shot=False`, so pass `zero_shot=True` to match a
zero-shot backtest.

## One file per horizon

A saved model forecasts the one horizon it was trained for. `af.pretrained`
already returns one per horizon. For your own saved models, give one file per
horizon:

```python
weights = {h: f"runs/big_city/forecast_.../models/gnn_st_noadapt_h{h}.pt" for h in (1, 2, 3, 4)}
fc = af.forecast(my_city, models=[], init_from=weights, zero_shot=True)
```

`af.Checkpoint.load(path)` opens a single file, so you can see which horizon,
inputs and city it came from.

A single file with more than one horizon is an error: a 1-week model used 4
weeks ahead would otherwise run without complaint and be wrong.

## Where the model comes from

- **Your own:** train in a city with more history using `save_models=True`:

    ```python
    source = af.resolve("gnn_st_noadapt")
    af.forecast(big_city, models=[source], save_models=True)   # writes models/*.pt
    ```

- **Bundled Boston models:** `af.pretrained("boston")`. These are the
  transferable graph model, trained on every Boston week through May 2026,
  with weather and no static characteristics. Applied untrained to Columbus's
  2024-25 season, they were clearly better than persistence at 1 week, and
  better on average but not clearly at 2-4 weeks. They saw Boston weeks from
  after that season, so treat this as a sanity check and backtest them on your
  data.

Applying them is inference only, but it still runs 100 dropout samples per
ensemble member on the validation weeks to calibrate the bands: about half a
minute per horizon for 15 areas, and over a minute for 60.

**Inputs must match.** A saved model expects the same inputs it was trained
with. The Boston weights use weather and no static characteristics, so call
`city.fetch_weather()` first. The package compares the saved model's input
features with your city's. With `zero_shot=True` a mismatch stops the run and
says what differs, instead of silently running a partly random model. When
fine-tuning, it warns and trains the layers that didn't fit from scratch.

**Only open files you trust.** Checkpoints are opened in PyTorch's weights-only
mode, which cannot run code.

[Example 3](examples/03_short_history_transfer.ipynb) walks through the whole
thing on a made-up city with 2.3 years of data.
