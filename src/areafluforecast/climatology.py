"""A per-area seasonal baseline, fitted on training weeks only.

At longer horizons the last observed week is a poor starting point, while a
seasonal average does not decay with the horizon. The models predict a residual
from a learned blend of the two (see `models.SpatioTemporalGNN`).

Fitted in log1p space: rates are positive, right-skewed and span an order of
magnitude between the off-season floor and the peak, and a harmonic regression
on the raw rate can go negative in the trough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_HARMONICS = 3


def _design(index: pd.DatetimeIndex, n_harmonics: int) -> np.ndarray:
    """Intercept plus sin/cos pairs on the day-of-year angle (no week-53 seam)."""
    angle = 2.0 * np.pi * index.dayofyear.to_numpy(dtype=np.float64) / 365.2425
    columns = [np.ones(len(index))]
    for k in range(1, n_harmonics + 1):
        columns.append(np.sin(k * angle))
        columns.append(np.cos(k * angle))
    return np.column_stack(columns)


@dataclass(frozen=True)
class Climatology:
    """Harmonic seasonal baseline, one coefficient vector per area."""

    coefficients: np.ndarray      # (n_terms, n_nodes)
    n_harmonics: int
    n_train_rows: int
    fallback_nodes: tuple[int, ...]

    def level(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Expected rate per area for each week in `index`, in rate units."""
        predicted = _design(index, self.n_harmonics) @ self.coefficients
        return np.maximum(np.expm1(predicted), 0.0)

    def to_json(self) -> dict:
        return {"n_harmonics": self.n_harmonics, "n_train_rows": self.n_train_rows,
                "coefficients": self.coefficients.tolist(),
                "fallback_nodes": list(self.fallback_nodes)}


def fit_climatology(rates: pd.DataFrame, *, end: int | None = None,
                    n_harmonics: int = DEFAULT_HARMONICS) -> Climatology:
    """Fit the seasonal baseline on `rates` rows before position `end`.

    Missing weeks are dropped per area rather than imputed. An area with too few
    observations to identify the harmonics falls back to its own mean.
    """
    frame = rates if end is None else rates.iloc[:end]
    values = frame.to_numpy(dtype=np.float64)
    design = _design(frame.index, n_harmonics)
    n_terms = design.shape[1]

    coefficients = np.zeros((n_terms, values.shape[1]), dtype=np.float64)
    fallback: list[int] = []
    for node in range(values.shape[1]):
        observed = np.isfinite(values[:, node])
        if int(observed.sum()) < 2 * n_terms:
            fallback.append(node)
            mean = np.nanmean(values[:, node]) if observed.any() else 0.0
            coefficients[0, node] = np.log1p(max(mean, 0.0))
            continue
        target = np.log1p(np.maximum(values[observed, node], 0.0))
        beta, *_ = np.linalg.lstsq(design[observed], target, rcond=None)
        coefficients[:, node] = beta

    return Climatology(coefficients=coefficients, n_harmonics=n_harmonics,
                       n_train_rows=int(len(frame)), fallback_nodes=tuple(fallback))
