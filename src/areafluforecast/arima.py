"""Seasonal ARIMA per area: order search, walk-forward forecasts, divergence guard.

A port of the research `run_arima.py`: one univariate model per area, the order
chosen by walk-forward validation RMSE at the forecast horizon, then the fitted
state rolled forward one observed week at a time (no refit). `harmonics` sin/cos
pairs on the day-of-year angle enter as exogenous regressors; they are a
function of the calendar only, so their future values are known at the origin.

Autoregressive models like this stay competitive with far more complex ones at
1-4 weeks ahead (Kandula & Shaman 2019, Epidemics 27:41), which is why the
paper uses one as its statistical baseline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Forecasts above this multiple of the observed historical maximum are treated
# as a diverged fit rather than a prediction. `enforce_stationarity=False` lets
# statsmodels return explosive AR roots, and an explosive forecast is perfectly
# finite, so an isfinite() check alone does not catch it.
DIVERGENCE_FACTOR = 10.0
DIVERGENCE_FLOOR = 10.0

# Parameter standard errors are never used (only point forecasts are), and
# computing them can hang LAPACK on near-singular fits. Forecasts are unchanged.
FIT_COV = "none"


def _arima():
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("The arima model needs statsmodels: pip install statsmodels") from exc
    return ARIMA


def seasonal_exog(index: pd.DatetimeIndex, harmonics: int) -> np.ndarray | None:
    """Day-of-year harmonics for `index`, without the intercept ARIMA already has."""
    if harmonics <= 0:
        return None
    angle = 2.0 * np.pi * index.dayofyear.to_numpy(dtype=np.float64) / 365.2425
    return np.column_stack([f(k * angle) for k in range(1, harmonics + 1)
                            for f in (np.sin, np.cos)])


def future_index(end: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([end + pd.Timedelta(weeks=k) for k in range(1, steps + 1)])


def forecast_bound(history: pd.Series) -> float:
    """Largest forecast value considered a prediction rather than a divergence."""
    observed = history.dropna()
    peak = float(observed.max()) if len(observed) else 0.0
    return max(peak * DIVERGENCE_FACTOR, DIVERGENCE_FLOOR)


def _check(values: np.ndarray, steps: int, bound: float) -> np.ndarray:
    if values.size != steps or not np.all(np.isfinite(values)):
        raise ValueError("non-finite ARIMA forecast")
    if np.any(np.abs(values) > bound):
        raise ValueError(f"diverged ARIMA forecast (|{np.max(np.abs(values)):.3g}| > {bound:.3g})")
    return values


def _fit(history: pd.Series, order, harmonics: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _arima()(history, exog=seasonal_exog(history.index, harmonics), order=order,
                        enforce_stationarity=False, enforce_invertibility=False).fit(
                            cov_type=FIT_COV)


def safe_forecast(history: pd.Series, order, steps: int, harmonics: int) -> tuple[np.ndarray, bool]:
    """Forecast, falling back to the last observed value if the fit fails."""
    try:
        fitted = _fit(history, order, harmonics)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = np.asarray(fitted.forecast(
                steps=steps, exog=seasonal_exog(future_index(history.index[-1], steps), harmonics)),
                dtype=float)
        return _check(values, steps, forecast_bound(history)), False
    except Exception:
        observed = history.dropna()
        fallback = float(observed.iloc[-1]) if len(observed) else 0.0
        return np.full(steps, fallback), True


def walk_forward_forecasts(series: pd.Series, order, origins: pd.DatetimeIndex, steps: int,
                           harmonics: int, initial_end: pd.Timestamp | None = None,
                           ) -> tuple[list[np.ndarray], int]:
    """Fit once, then update the model state with each newly observed week.

    If anything on that path fails, every origin is redone with a full refit
    (`safe_forecast`); the second value counts the origins that fell back to
    the last observed value.
    """
    forecasts: list[np.ndarray] = []
    fallbacks = 0
    first_end = initial_end if initial_end is not None else origins[0]
    bound = forecast_bound(series.loc[:first_end])
    try:
        result = _fit(series.loc[:first_end], order, harmonics)
        last_end = first_end
        for origin in origins:
            new_values = series.loc[(series.index > last_end) & (series.index <= origin)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if len(new_values):
                    result = result.append(new_values,
                                           exog=seasonal_exog(new_values.index, harmonics),
                                           refit=False)
                values = np.asarray(result.forecast(
                    steps=steps, exog=seasonal_exog(future_index(origin, steps), harmonics)),
                    dtype=float)
            forecasts.append(_check(values, steps, bound))
            last_end = origin
    except Exception:
        forecasts = []
        for origin in origins:
            values, fallback = safe_forecast(series.loc[:origin], order, steps, harmonics)
            forecasts.append(values)
            fallbacks += int(fallback)
    return forecasts, fallbacks


def select_order(series: pd.Series, train_last: pd.Timestamp, val_dates: pd.DatetimeIndex,
                 horizon: int, steps: int, *, harmonics: int, max_p: int, max_d: int,
                 max_q: int) -> tuple[int, int, int]:
    """Grid search (p, d, q) by walk-forward RMSE on the validation origins,
    scored at `horizon`. An order that needs the fallback on validation is
    disqualified: its RMSE would be the fallback's, not the order's."""
    best_order, best_rmse = (1, 0, 0), np.inf
    actual = np.asarray([series.loc[d + pd.Timedelta(weeks=horizon)] for d in val_dates],
                        dtype=float)
    for p in range(max_p):
        for d in range(max_d):
            for q in range(max_q):
                order = (p, d, q)
                preds, fallbacks = walk_forward_forecasts(series, order, val_dates, steps,
                                                          harmonics, initial_end=train_last)
                if fallbacks:
                    continue
                predicted = np.asarray([f[horizon - 1] for f in preds], dtype=float)
                squared = (predicted - actual) ** 2
                if not np.any(np.isfinite(squared)):
                    continue
                rmse = float(np.sqrt(np.nanmean(squared)))
                if rmse < best_rmse:
                    best_order, best_rmse = order, rmse
    return best_order
