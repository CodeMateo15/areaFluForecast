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

Everything listed in `__all__` is the public API, covered by the versioning
policy; the submodules adjacency, scoring, timing and weather are public too.
Anything else may change without notice.
"""

import warnings as _warnings

from . import adjacency, scoring, timing, weather
from .backtest import BacktestResult, backtest, combine, latest_backtest
from .checkpoint import Checkpoint, pretrained
from .city import City, rates_from_counts, read_rates_csv
from .configs import (
    PRESETS,
    ArimaConfig,
    BaselineConfig,
    ColaGNNConfig,
    GNNConfig,
    LSTMConfig,
    ModelConfig,
    resolve,
)
from .datasets import load_city, load_sample, save_city
from .errors import DataError, DataWarning, TrustWarning
from .drivers import explain
from .forecast import ForecastResult, forecast

__version__ = "1.2.0"

__all__ = ["City", "load_sample", "load_city", "save_city", "rates_from_counts", "read_rates_csv",
           "backtest", "BacktestResult", "latest_backtest", "combine",
           "forecast", "ForecastResult", "explain",
           "Checkpoint", "pretrained",
           "GNNConfig", "LSTMConfig", "ColaGNNConfig", "BaselineConfig", "ArimaConfig", "ModelConfig",
           "PRESETS", "resolve",
           "DataError", "DataWarning", "TrustWarning",
           "adjacency", "scoring", "timing", "weather", "__version__"]

# Kept from 0.1 so existing code keeps working; they warn, and go in 2.0.
_DEPRECATED = {
    "save_checkpoint": "use Checkpoint(...).save(path), or forecast(..., save_models=True)",
    "load_checkpoint": "use Checkpoint.load(path)",
}


def __getattr__(name: str):
    if name in _DEPRECATED:
        _warnings.warn(f"areafluforecast.{name} is deprecated and will be removed in 2.0: "
                       f"{_DEPRECATED[name]}.", DeprecationWarning, stacklevel=2)
        from . import checkpoint
        return getattr(checkpoint, name)
    raise AttributeError(f"module 'areafluforecast' has no attribute {name!r}")
