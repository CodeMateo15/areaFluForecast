"""Turning a city's weekly tables into model tensors.

One sample per forecast origin t: per-area features from the lookback window
ending at t, and targets at t + h. Copied from the research code
(influenza/samples.py), with the research-only data sources removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .city import City, scaled_static
from .climatology import fit_climatology
from .errors import DataError
from .graphs import Graph
from .weather import WEATHER_COLS, align_weather
from .windows import Split, Window, normalization, train_end

# Anchors carry no data: temporal features are zero and static features sit at
# the middle of the scaled range, so an anchor is a neutral boundary node whose
# only distinguishing property is which areas it borders.
ANCHOR_STATIC_VALUE = 0.5


@dataclass(frozen=True)
class FeatureSpec:
    use_weather: bool = True
    use_static: bool = False
    use_imputed_flag: bool = True


def impute_causal(frame: pd.DataFrame, *, ffill_limit: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing values from the past only (inputs must be finite; targets stay NaN).

    Interpolation would read the week after the origin, which is the target, so
    the last observation is carried forward (up to `ffill_limit` weeks), then the
    expanding median.
    """
    was_missing = frame.isna()
    filled = frame.ffill(limit=ffill_limit)
    filled = filled.fillna(frame.expanding(min_periods=1).median())
    return filled.fillna(0.0), was_missing


@dataclass
class Dataset:
    week_index: pd.DatetimeIndex
    rates: pd.DataFrame
    flu_features: pd.DataFrame
    flu_imputed: pd.DataFrame
    weather: dict[str, np.ndarray]
    static: np.ndarray | None
    static_cols: tuple[str, ...] = ()


def build_dataset(city: City, features: FeatureSpec, rates: pd.DataFrame | None = None) -> Dataset:
    rates = city.rates if rates is None else rates
    flu_features, flu_imputed = impute_causal(rates)
    weather = {}
    if features.use_weather:
        if city.weather is None:
            raise DataError(f"{city.label} has no weather; call city.fetch_weather() or turn "
                             "weather off in the model config.")
        weather = align_weather(city.weather, rates.index, city.areas)
    static, cols = (scaled_static(city) if features.use_static else (None, []))
    if features.use_static and static is None:
        raise DataError(f"{city.label} has no static characteristics; pass static= or turn "
                         "use_static off.")
    return Dataset(week_index=rates.index, rates=rates, flu_features=flu_features,
                   flu_imputed=flu_imputed, weather=weather, static=static,
                   static_cols=tuple(cols))


@dataclass
class Normalization:
    flu_mean: np.ndarray
    flu_std: np.ndarray

    def to_level(self, values: np.ndarray) -> np.ndarray:
        return values * self.flu_std.flatten()[None, :, None] + self.flu_mean.flatten()[None, :, None]


@dataclass
class Samples:
    X: np.ndarray            # (S, n_nodes, n_feat)
    g: np.ndarray            # (S, 0) -- no city-wide covariates
    y: np.ndarray            # (S, n_areas, n_horizons), NaN where unobserved
    anchors: np.ndarray      # (S, n_areas) normalised level at the origin
    clim: np.ndarray         # (S, n_areas, n_horizons) normalised seasonal baseline
    positions: list[int]
    feature_names: list[str]
    lookback: int
    temporal_channels: int

    @property
    def n_feat(self) -> int:
        return self.X.shape[2]

    @property
    def n_global(self) -> int:
        return self.g.shape[1]

    @property
    def n_static_feat(self) -> int:
        return self.n_feat - self.lookback * self.temporal_channels

    def subset(self, positions: list[int]) -> Samples:
        lookup = {p: r for r, p in enumerate(self.positions)}
        rows = [lookup[p] for p in positions]
        return Samples(X=self.X[rows], g=self.g[rows], y=self.y[rows],
                       anchors=self.anchors[rows], clim=self.clim[rows],
                       positions=list(positions), feature_names=self.feature_names,
                       lookback=self.lookback, temporal_channels=self.temporal_channels)


def _zscore(values: np.ndarray, *, end: int) -> np.ndarray:
    reference = values[:end]
    mean = np.nanmean(reference, axis=0, keepdims=True)
    std = np.nanstd(reference, axis=0, keepdims=True) + 1e-8
    return (values - mean) / std


def build_samples(dataset: Dataset, split: Split, features: FeatureSpec, window: Window,
                  graph: Graph, *, target: str = "blend") -> tuple[Samples, Normalization]:
    """Tensors for every origin in `split`, normalised on training rows only."""
    n_weeks = len(dataset.week_index)
    end = train_end(split)

    flu_mean, flu_std = normalization(dataset.rates, split)
    flu_features = (dataset.flu_features.to_numpy(dtype=np.float32) - flu_mean) / flu_std
    flu_targets = (dataset.rates.to_numpy(dtype=np.float32) - flu_mean) / flu_std
    weather_norm = ({col: _zscore(dataset.weather[col], end=end) for col in WEATHER_COLS}
                    if features.use_weather else {})
    imputed_mask = dataset.flu_imputed.to_numpy(dtype=np.float32)

    climatology = fit_climatology(dataset.rates, end=end)
    clim_level = (climatology.level(dataset.week_index) - flu_mean) / flu_std

    feature_names = _feature_names(features, window, dataset.static_cols)
    n_feat = len(feature_names)
    positions = sorted(set(split.train + split.val + split.test))
    n_samples, n_neigh = len(positions), graph.n_neigh
    use_static = features.use_static and dataset.static is not None

    X = np.zeros((n_samples, graph.n_nodes, n_feat), dtype=np.float32)
    g = np.zeros((n_samples, 0), dtype=np.float32)
    y = np.zeros((n_samples, n_neigh, len(window.horizons)), dtype=np.float32)
    anchors = np.zeros((n_samples, n_neigh), dtype=np.float32)
    clim = np.zeros((n_samples, n_neigh, len(window.horizons)), dtype=np.float32)

    for row, t in enumerate(positions):
        if t - window.lookback + 1 < 0 or t + window.max_horizon >= n_weeks:
            raise ValueError(f"Origin {t} lacks a full lookback or target span.")
        for node in range(n_neigh):
            values: list[float] = []
            for lag in range(window.lookback):
                week = t - lag
                values.append(flu_features[week, node])
                if features.use_imputed_flag:
                    values.append(imputed_mask[week, node])
            if features.use_weather:
                values.extend(weather_norm[col][t, node] for col in WEATHER_COLS)
            row_values = np.asarray(values, dtype=np.float32)
            if use_static:
                row_values = np.concatenate([row_values, dataset.static[node]])
            X[row, node] = row_values
        if graph.n_anchors:
            anchor_row = np.zeros(n_feat, dtype=np.float32)
            if use_static:
                anchor_row[n_feat - dataset.static.shape[1]:] = ANCHOR_STATIC_VALUE
            X[row, n_neigh:] = anchor_row
        anchors[row] = flu_features[t]
        for h_idx, horizon in enumerate(window.horizons):
            level = flu_targets[t + horizon]
            y[row, :, h_idx] = level - flu_features[t] if target == "delta" else level
            clim[row, :, h_idx] = clim_level[t + horizon, :n_neigh]

    temporal_channels = 1 + (1 if features.use_imputed_flag else 0)
    return (Samples(X=X, g=g, y=y, anchors=anchors, clim=clim, positions=positions,
                    feature_names=feature_names, lookback=window.lookback,
                    temporal_channels=temporal_channels),
            Normalization(flu_mean=flu_mean, flu_std=flu_std))


def _feature_names(features: FeatureSpec, window: Window, static_cols: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for lag in range(window.lookback):
        names.append(f"flu_lag{lag}")
        if features.use_imputed_flag:
            names.append(f"flu_imputed_lag{lag}")
    if features.use_weather:
        names.extend(WEATHER_COLS)
    if features.use_static:
        names.extend(static_cols)
    return names


def sequence_arrays(features: pd.DataFrame, targets: pd.DataFrame, positions: list[int],
                    mean, std, window: Window, clim_rate: np.ndarray | None = None):
    """Sliding windows for the LSTM and Cola-GNN: x (S, L, N), y (S, H, N), clim (S, H, N)."""
    x_source = (features.to_numpy(dtype=np.float32) - mean) / std
    y_source = (targets.to_numpy(dtype=np.float32) - mean) / std
    x = np.stack([x_source[t - window.lookback + 1: t + 1] for t in positions]).astype(np.float32)
    y = np.stack([np.stack([y_source[t + h] for h in window.horizons])
                  for t in positions]).astype(np.float32)
    if clim_rate is None:
        clim = np.zeros_like(y)
    else:
        c_source = (clim_rate.astype(np.float32) - mean) / std
        clim = np.stack([np.stack([c_source[t + h] for h in window.horizons])
                         for t in positions]).astype(np.float32)
    return x, y, clim
