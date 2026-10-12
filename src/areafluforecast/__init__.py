"""areaFluForecast: neighborhood-scale influenza forecasts for health departments.

    import areafluforecast as af

    city = af.City.from_files("rates.csv", areas="areas.geojson", id_col="name")
    city.fetch_weather()                    # optional
    bt = af.backtest(city)                  # would it have worked here?
    fc = af.forecast(city)                  # next 1-4 weeks, with a trust check
    fc.to_csv("forecast.csv")

The models are the graph neural network evaluated in the accompanying paper, a
multivariate LSTM and Cola-GNN, each compared with persistence ("next week
looks like this week") on your own past seasons before you rely on it.
"""

from .backtest import BacktestResult, backtest, latest_backtest
from .checkpoint import load_checkpoint, save_checkpoint
from .city import City, DataError, rates_from_counts, read_rates_csv
from .configs import PRESETS, ColaGNNConfig, GNNConfig, LSTMConfig, resolve
from .datasets import load_city, load_sample, save_city
from .forecast import ForecastResult, TrustWarning, forecast

__version__ = "0.2.0"

__all__ = ["City", "DataError", "backtest", "forecast", "load_sample", "load_city", "save_city", "BacktestResult",
           "ForecastResult", "TrustWarning", "GNNConfig", "LSTMConfig", "ColaGNNConfig",
           "PRESETS", "resolve", "latest_backtest", "save_checkpoint", "load_checkpoint",
           "rates_from_counts", "read_rates_csv", "__version__"]
