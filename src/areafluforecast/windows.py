"""Sample origins and the train / validation / test split.

Integer positions into the weekly index are the primitive: a position can always
produce its date, while a date cannot produce its position without a lookup.

Validation is always "pre_test": train and validation only use origins whose
furthest target lands strictly before the first scored week, and validation is
the last `val_fraction` of those. Nothing after the scored period can leak back
into model selection, whether the scored period is a past season (backtest) or
the weeks after the last observation (forecast).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import DataError


@dataclass(frozen=True)
class Window:
    """Sampling geometry and the scored window."""

    lookback: int
    horizons: tuple[int, ...]
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    val_fraction: float = 0.15

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def min_horizon(self) -> int:
        return min(self.horizons)

    def to_json(self) -> dict:
        return {"lookback": self.lookback, "horizons": list(self.horizons),
                "test_start": str(pd.Timestamp(self.test_start).date()),
                "test_end": str(pd.Timestamp(self.test_end).date()),
                "val_fraction": self.val_fraction}


@dataclass(frozen=True)
class Split:
    """Train / validation / test origins as integer positions into `index`."""

    index: pd.DatetimeIndex
    train: list[int]
    val: list[int]
    test: list[int]
    window: Window

    def to_json(self) -> dict:
        return {"n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test)}


def valid_origins(index: pd.DatetimeIndex, window: Window) -> list[int]:
    """Positions whose full lookback window and every horizon target are
    consecutive weeks, so no sample straddles a gap in the calendar."""
    max_h = window.max_horizon
    span_len = window.lookback + max_h
    week = np.timedelta64(7, "D")
    positions: list[int] = []
    for t in range(window.lookback, len(index) - max_h):
        span = index[t - window.lookback + 1: t + max_h + 1]
        if len(span) == span_len and np.all(np.diff(span.values).astype("timedelta64[D]") == week):
            positions.append(t)
    return positions


def split_origins(index: pd.DatetimeIndex, origins: list[int], window: Window) -> Split:
    """Chronological split: test by target date, validation as the last slice of
    the pre-test origins, and a purge so no training target leaks forward."""
    first = window.min_horizon
    test = [t for t in origins
            if window.test_start <= index[t + first] <= window.test_end]
    if not test:
        raise DataError(
            f"No forecast origins score inside {pd.Timestamp(window.test_start).date()}.."
            f"{pd.Timestamp(window.test_end).date()}; the data runs "
            f"{index.min().date()}..{index.max().date()}.")
    test_set = set(test)
    test_target_start = index[test[0] + first]
    candidates = [t for t in origins
                  if t not in test_set and index[t + window.max_horizon] < test_target_start]

    n_val = max(1, int(window.val_fraction * len(candidates)))
    train, val = candidates[:-n_val], candidates[-n_val:]
    if not val:
        raise DataError("Not enough history before the scored window to hold out validation weeks.")
    val_target_start = index[val[0] + first]
    train = [t for t in train if index[t + window.max_horizon] < val_target_start]
    val = [t for t in val if index[t + window.max_horizon] < test_target_start]
    if not train or not val:
        raise DataError(
            f"Only {len(candidates)} usable origins before "
            f"{pd.Timestamp(window.test_start).date()}: too little history to train and "
            "validate. Supply more past weeks or score a later window.")
    return Split(index=index, train=train, val=val, test=test, window=window)


def normalization(frame: pd.DataFrame, split: Split) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std over rows up to the last training target, shaped (1, n)."""
    end = split.train[-1] + split.window.max_horizon + 1
    rows = frame.to_numpy(dtype=np.float32)[:end]
    mean = np.nanmean(rows, axis=0, keepdims=True)
    std = np.nanstd(rows, axis=0, keepdims=True) + 1e-8
    return mean, std


def train_end(split: Split) -> int:
    """Row position one past the last week a training target reveals.

    Normalisation, the seasonal baseline and the weather z-scores all stop here,
    so they share one leakage boundary.
    """
    return split.train[-1] + split.window.max_horizon + 1
