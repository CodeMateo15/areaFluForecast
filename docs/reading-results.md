# Reading the results

`bt.report()` answers four questions. Here's how to read each one, and then how
to decide which forecast to use.

## 1. Did the model beat persistence?

| column | the question it answers |
|---|---|
| **WIS** (weighted interval score) | How good was the *whole* forecast, the uncertainty band as well as the single number? Lower is better, in rate units. CDC FluSight ranks forecasts by it. |
| **relative WIS** | The model's WIS divided by persistence's, on the same weeks. Below 1 means it beat "next week looks like this week"; 0.8 means 20% lower error. |
| **p (Holm)** | Could the gap just be week-to-week noise? It's tested over **weeks**, keeping all areas together, because neighbouring areas move together and treating each area-week as separate would make every gap look real. It's corrected for the number of comparisons. |
| **verdict** | "better than persistence" means relative WIS below 1 **and** p below 0.05. |

## 2. Are the bands honest?

**Coverage** is how often the truth fell inside the 95% band. It should be near
95%:
- **Much lower** means the model is overconfident.
- **Much higher** means the bands are wider than they need to be.

The bands are calibrated on validation weeks before the season, never on the
weeks being scored.

## 3. Is the forecast late?

The **timing** section of the report gives one line per model and horizon:

```text
gnn_st, 1 wk ahead: 1 week(s) behind the curve (no earlier than persistence);
  67% of its squared error is timing; peaks a median 1 week(s) late
persistence, 1 wk ahead: 1 week(s) behind the curve (exactly 1 by construction)
```

- **Weeks behind the curve:** how far the forecast would have to be slid
  earlier to fit the observed curve best. Persistence is exactly *h* weeks
  behind at *h* weeks ahead, by definition.
- **Share of squared error that is timing:** how much of the error would
  disappear with that slide.
- **Peak weeks late:** forecast peak week minus observed peak week, the median
  over areas.

A model as far behind as persistence gets its advantage from the **size** of
the curve, not from seeing turns earlier. In the research behind this package,
that held for every model, tuned or not. See
[Limits](limits.md#the-forecasts-run-late) for why.

## 4. Where in the season?

`bt.predictions` holds every forecast, so you can plot errors week by week.
[Example 4](examples/04_reading_the_results.ipynb) shows that, along with a
calibration (PIT) histogram and forecast-versus-observed plots for one area.

## Which forecast should we use?

A simple rule that's easy to defend: for each horizon, use the model with the
lowest WIS **among those clearly better than persistence**. If none is clearly
better, report persistence, or report the forecasts with a caveat.

```python
def choose(comparison):
    rows = []
    for h, part in comparison.groupby("horizon"):
        good = part[part.verdict == "better than persistence"].nsmallest(1, "wis")
        rows.append({"horizon": h, "use": good.model.iloc[0] if len(good) else "persistence"})
    return pd.DataFrame(rows)

choose(bt.comparison)
```

"None of the models helped here" is a legitimate result, and the backtest
exists to tell you that before anyone acts on a forecast.
