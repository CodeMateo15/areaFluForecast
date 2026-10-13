"""The seasonal ARIMA preset: order search, fallback, and a one-order run on the sample."""

import numpy as np
import pandas as pd

import areafluforecast as af
from areafluforecast.arima import forecast_bound, safe_forecast, seasonal_exog, select_order
from areafluforecast.configs import from_json, is_learned, to_json
from areafluforecast.engine import run_model
from areafluforecast.intervals import QUANTILE_COLUMNS


def _ar1(n=200, phi=0.8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return pd.Series(x + 20, index=pd.date_range("2020-01-05", periods=n, freq="7D"))


def test_preset_round_trips_and_is_not_learned():
    config = af.resolve("arima")
    assert config.harmonics == 3 and not is_learned(config)
    assert from_json(to_json(config)) == config


def test_harmonics_shape():
    index = pd.date_range("2024-01-07", periods=5, freq="7D")
    assert seasonal_exog(index, 0) is None
    assert seasonal_exog(index, 3).shape == (5, 6)


def test_select_order_finds_an_ar_term():
    series = _ar1()
    val = series.index[150:180]
    order = select_order(series, series.index[140], val, 1, 1, harmonics=0,
                         max_p=2, max_d=1, max_q=2)
    assert order[0] >= 1


def test_failed_fit_falls_back_to_last_value(monkeypatch):
    import areafluforecast.arima as arima

    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(arima, "_fit", broken)
    series = pd.Series([1.0, 2.0, 5.0, np.nan], index=pd.date_range("2024-01-07", periods=4, freq="7D"))
    values, fallback = safe_forecast(series, (1, 0, 0), 2, 0)
    assert fallback and np.allclose(values, 5.0)
    assert forecast_bound(series) == 50.0


def test_one_order_run_on_sample():
    city = af.load_sample("boston")
    config = af.resolve("arima", {"max_p": 1, "max_d": 1, "max_q": 1})
    out = run_model(city, config, horizon=1, test_start="2025-12-01", test_end="2026-01-31")
    q = out.test[list(QUANTILE_COLUMNS)].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-9).all()
    assert (out.test["predicted"] >= 0).all()
    assert set(out.meta["selected_orders"].values()) == {"(0, 0, 0)"}
