# areaFluForecast

**Neighborhood-scale influenza forecasts, 1 to 4 weeks ahead, for public health
departments.**

You supply weekly flu rates for your areas (neighborhoods, ZIP-code groups,
districts) and a map of which areas border which. The package does three
things:

1. **Trains three models** on your data:
    - **`gnn_st`**, a graph neural network that uses the map and which areas'
      curves move together. This is the model from the accompanying paper.
    - **`lstm`**, a recurrent network that sees every area but no map.
    - **`colagnn`**, Cola-GNN (Deng et al. 2020), a published graph model for flu,
      reimplemented from the paper.
2. **Checks each model against persistence** ("next week looks like this
   week") on your own past seasons, before you rely on any of them.
3. **Forecasts** the weeks after your last observation. Each forecast comes
   with a median, a 95% band and the 23 quantiles CDC FluSight uses. A trust
   label says whether that model actually beat persistence here.

```python
import areafluforecast as af

city = af.City.from_files("rates.csv", areas="areas.geojson", id_col="NAME")
city.fetch_weather()            # optional, free, no API key
bt = af.backtest(city)          # would it have worked here?
fc = af.forecast(city)          # the next 1-4 weeks, with a trust check
fc.to_csv("forecast.csv")
```

!!! note "What the evidence says"
    - **Boston:** on seasons the models never saw, the graph model's error was
      18–28% below persistence. It tied a carefully tuned LSTM at 1 week and led
      it on average at 2–4 weeks.
    - **Chicago:** on public ZIP-level data, all three models clearly beat
      persistence ([example 5](examples/05_real_city_chicago.ipynb)).
    - **Columbus:** in its one unseen season the graph model's error was 12–23%
      below persistence on average, but with a single season that lead was not
      statistically clear.
    - **Everywhere:** the forecasts run about as far behind the epidemic curve as
      persistence does ([why](limits.md#the-forecasts-run-late)).

    Your city could look like any of these. The backtest is how you find out.

## Where to start

<div class="grid cards" markdown>

- **[Installation](installation.md):** one `pip install`, CPU only.
- **[Preparing your data](your-data.md):** the two files you need, and the
  checks the package runs on them.
- **[Backtest and forecast](workflow.md):** the two calls, what they save, how
  long they take.
- **[Reading the results](reading-results.md):** what WIS, relative WIS and
  the timing lines mean, and which forecast to use.
- **[Examples](examples/index.md):** five notebooks with their outputs,
  including a real city built from open data.

</div>

## Your data stays with you

The package never uploads your data. Its only network call is the optional
weather download, which sends area centre points to Open-Meteo.
