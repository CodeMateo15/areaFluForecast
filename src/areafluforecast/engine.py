"""Fit one model for one horizon on one split, and forecast the scored origins.

Everything above this (backtest, forecast, tune) is a loop over `run_model`.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .city import City
from .climatology import fit_climatology
from .configs import BaselineConfig, ColaGNNConfig, GNNConfig, LSTMConfig, ModelConfig, to_json
from .graphs import GraphSpec, build_graph
from .intervals import QUANTILE_COLUMNS, attach_intervals, fit_intervals
from .models import ColaGNN, MultivariateLSTM, SpatioTemporalGNN, blend_init
from .samples import FeatureSpec, build_dataset, build_samples, impute_causal, sequence_arrays
from .training import (assert_mc_dropout_valid, calibrate_intervals, mc_dropout_stgnn,
                       seed_everything, train_sequence_model, train_stgnn)
from .windows import Split, Window, normalization, split_origins, train_end, valid_origins

# Called once per finished unit of work (one ensemble member, or one baseline).
UnitCallback = Callable[[str], None] | None


@dataclass
class FitOutput:
    """Forecasts for the scored origins, plus what is needed to reuse the model."""

    test: pd.DataFrame          # origin_date, target_date, horizon, area, actual, predicted, lower, upper, q...
    val: pd.DataFrame           # validation-split point forecasts
    config: ModelConfig
    members: list[dict] = field(default_factory=list)      # state_dicts
    meta: dict = field(default_factory=dict)
    standard: pd.DataFrame | None = None   # with config.refit: the pre-refit forecasts


def units(config: ModelConfig) -> int:
    """How many progress units fitting this config takes."""
    return max(1, getattr(config, "n_seeds", 1)) * (2 if getattr(config, "refit", False) else 1)


def make_window(config: ModelConfig, horizon: int, test_start, test_end) -> Window:
    lookback = getattr(config, "lookback", 16)
    return Window(lookback=lookback, horizons=(int(horizon),),
                  test_start=pd.Timestamp(test_start), test_end=pd.Timestamp(test_end))


def make_split(index: pd.DatetimeIndex, window: Window, *, only_origin: int | None = None) -> Split:
    split = split_origins(index, valid_origins(index, window), window)
    if only_origin is not None:
        if only_origin not in split.test:
            raise ValueError("The forecast origin is not a valid origin; is the last week observed "
                             "with a full lookback behind it?")
        split = replace(split, test=[only_origin])
    return split


def run_model(city: City, config: ModelConfig, *, horizon: int, test_start, test_end,
              rates: pd.DataFrame | None = None, only_origin: int | None = None,
              init_members: list[dict] | None = None, zero_shot: bool = False,
              init_features: list[str] | None = None,
              on_unit: UnitCallback = None, device: str | None = None) -> FitOutput:
    """Train `config` to forecast `horizon` weeks ahead and score [test_start, test_end].

    `rates` overrides the city's rates (forecasting passes them extended with
    empty future weeks). `only_origin` keeps a single scored origin.
    `init_members` are saved weights to start from; `init_features` the input
    feature names they were trained on, checked against this city's.
    """
    rates = city.rates if rates is None else rates
    window = make_window(config, horizon, test_start, test_end)
    split = make_split(rates.index, window, only_origin=only_origin)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(config, GNNConfig):
        out = _run_gnn(city, config, rates, window, split, torch_device, init_members,
                       zero_shot, init_features, on_unit)
    elif isinstance(config, (LSTMConfig, ColaGNNConfig)):
        out = _run_sequence(city, config, rates, window, split, torch_device, on_unit)
    elif isinstance(config, BaselineConfig):
        out = _run_baseline(city, config, rates, window, split, on_unit)
    else:
        raise TypeError(f"Unknown config {type(config).__name__}")
    out.meta.update({"horizon": int(horizon), "window": window.to_json(),
                     "split": split.to_json(), "config": to_json(config)})
    return out


# --- helpers -----------------------------------------------------------------

def _frame(index, positions, horizons, areas, predicted, actual, *, lower=None, upper=None,
           quantiles=None) -> pd.DataFrame:
    """Long table from (S, N, H) arrays (quantiles: (S, N, H, 23))."""
    rows = []
    for s, position in enumerate(positions):
        for h_idx, horizon in enumerate(horizons):
            for node, area in enumerate(areas):
                row = {"origin_date": index[position],
                       "target_date": index[position + horizon],
                       "horizon": int(horizon), "area": area,
                       "actual": float(actual[s, node, h_idx]),
                       "predicted": max(0.0, float(predicted[s, node, h_idx]))}
                if lower is not None:
                    row["lower"] = float(lower[s, node, h_idx])
                    row["upper"] = float(upper[s, node, h_idx])
                if quantiles is not None:
                    row.update(zip(QUANTILE_COLUMNS, map(float, quantiles[s, node, h_idx])))
                rows.append(row)
    return pd.DataFrame(rows)


def _features(city: City, config: GNNConfig) -> FeatureSpec:
    use_weather = city.has_weather if config.use_weather is None else config.use_weather
    return FeatureSpec(use_weather=use_weather, use_static=config.use_static,
                       use_imputed_flag=config.use_imputed_flag)


def gnn_graph_spec(config: GNNConfig) -> GraphSpec:
    if not config.graph:
        return GraphSpec(anchors=False, geo=False, corr=False, demo=False)
    return GraphSpec(anchors=config.anchors, geo=True, geo_max_hop=config.geo_max_hop,
                     corr=config.corr, corr_binary=config.corr_binary,
                     corr_threshold=config.corr_threshold, demo=config.similarity_edges)


# --- graph model ---------------------------------------------------------------

def _run_gnn(city, config: GNNConfig, rates, window, split, device, init_members, zero_shot,
             init_features, on_unit) -> FitOutput:
    features = _features(city, config)
    dataset = build_dataset(city, features, rates)
    history = dataset.rates.loc[dataset.rates.index < window.test_start]
    static_for_graph = dataset.static
    if config.similarity_edges and static_for_graph is None:
        from .city import scaled_static
        static_for_graph = scaled_static(city)[0]
    graph = build_graph(gnn_graph_spec(config), city=city, flu_history=history,
                        static=static_for_graph)
    samples, norm = build_samples(dataset, split, features, window, graph, target="blend")
    train_set, val_set = samples.subset(split.train), samples.subset(split.val)
    test_set = samples.subset(split.test)
    relations = graph.relation_tensors()
    if init_members and init_features is not None and list(init_features) != samples.feature_names:
        theirs, ours = list(init_features), samples.feature_names
        detail = (f"only in the saved model: {sorted(set(theirs) - set(ours))}; only here: "
                  f"{sorted(set(ours) - set(theirs))}" if set(theirs) != set(ours)
                  else "the same features in a different order")
        message = (f"The saved model was trained on different input features ({detail}). It "
                   "needs the same inputs: weather on or off, the same weather and static columns.")
        if zero_shot:
            raise ValueError(message)
        warnings.warn(message + " Inputs that differ start from untrained weights.", stacklevel=4)

    def build() -> SpatioTemporalGNN:
        return SpatioTemporalGNN(
            lookback=samples.lookback, temporal_channels=samples.temporal_channels,
            n_static_feat=samples.n_static_feat, n_global=samples.n_global,
            n_nodes=graph.n_nodes, n_horizons=len(window.horizons),
            relation_names=[name for name, _, _ in relations], hidden=config.channels,
            dropout=config.dropout, dilations=config.dilations,
            adaptive_dim=config.adaptive_dim if config.adaptive else 0,
            dropedge=config.drop_edge, blend_init=blend_init(window.horizons),
            n_blocks=config.n_blocks).to(device)

    seeds = [config.seed + i for i in range(max(1, config.n_seeds))]
    per_member = max(1, config.mc_samples // len(seeds))
    test_draws, val_draws, members, results = [], [], [], []
    val_actual = test_actual = None
    for member, seed in enumerate(seeds):
        seed_everything(seed)
        model = build()
        if init_members:
            source = init_members[member % len(init_members)]
            own = model.state_dict()
            loadable = {k: v for k, v in source.items()
                        if k in own and tuple(v.shape) == tuple(own[k].shape)}
            missing = sorted(set(own) - set(loadable))
            if missing and (zero_shot or member == 0):
                message = (f"The pretrained model does not fit this city's inputs: {missing} "
                           "could not be loaded. A pretrained model needs the same input features "
                           "(weather on or off, same static columns) and adaptive=False.")
                if zero_shot:
                    raise ValueError(message)
                warnings.warn(message + " Those layers start from random weights and are "
                              "trained here.", stacklevel=4)
            model.load_state_dict(loadable, strict=False)
        assert_mc_dropout_valid(model)
        if not zero_shot:
            results.append(train_stgnn(model, train_set, val_set, relations=relations,
                                       config=config, device=device, n_neigh=graph.n_neigh))
        torch.manual_seed(config.mc_seed + member)
        t_draws, test_actual = mc_dropout_stgnn(model, test_set, relations=relations, norm=norm,
                                                n_neigh=graph.n_neigh, device=device, n_mc=per_member)
        v_draws, val_actual = mc_dropout_stgnn(model, val_set, relations=relations, norm=norm,
                                               n_neigh=graph.n_neigh, device=device, n_mc=per_member)
        test_draws.append(t_draws)
        val_draws.append(v_draws)
        members.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if on_unit:
            on_unit(f"{config.kind} seed {seed}")

    test_all = np.concatenate(test_draws, axis=0)
    val_all = np.concatenate(val_draws, axis=0)
    mean, lower, upper, quantiles, interval_model = calibrate_intervals(
        test_all, val_all, val_actual, window.horizons, two_sided=config.two_sided_intervals)
    test = _frame(split.index, split.test, window.horizons, city.areas, mean, test_actual,
                  lower=lower, upper=upper, quantiles=quantiles)
    val = _frame(split.index, split.val, window.horizons, city.areas, val_all.mean(axis=0), val_actual)
    standard = None
    if config.refit and not zero_shot and results:
        standard = test
        both = samples.subset(list(split.train) + list(split.val))
        refit_draws, refit_members = [], []
        for member, (seed, result) in enumerate(zip(seeds, results)):
            seed_everything(seed)
            model = build()
            train_stgnn(model, both, None, relations=relations,
                        config=replace(config, epochs=max(1, result.best_epoch)),
                        device=device, n_neigh=graph.n_neigh)
            torch.manual_seed(config.mc_seed + member)
            draws, _ = mc_dropout_stgnn(model, test_set, relations=relations, norm=norm,
                                        n_neigh=graph.n_neigh, device=device, n_mc=per_member)
            refit_draws.append(draws)
            refit_members.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if on_unit:
                on_unit(f"{config.kind} refit seed {seed}")
        draws = np.concatenate(refit_draws, axis=0)
        mean, epistemic = draws.mean(axis=0), draws.var(axis=0)
        grid = np.broadcast_to(np.asarray(window.horizons), mean.shape)
        lower, upper = interval_model.bounds(mean.ravel(), grid.ravel(), epistemic=epistemic.ravel())
        quantiles = interval_model.quantiles(mean.ravel(), grid.ravel(), epistemic=epistemic.ravel())
        test = _frame(split.index, split.test, window.horizons, city.areas, mean, test_actual,
                      lower=lower.reshape(mean.shape), upper=upper.reshape(mean.shape),
                      quantiles=quantiles.reshape(*mean.shape, -1))
        members = refit_members
    meta = {"graph": graph.summary(), "n_features": samples.n_feat,
            "feature_names": samples.feature_names, "node_names": graph.node_names,
            "intervals": interval_model.to_json(),
            "best_epochs": [r.best_epoch for r in results],
            "blend_weights": _first_model_blend(members, build)}
    return FitOutput(test=test, val=val, config=config, members=members, meta=meta,
                     standard=standard)


def _first_model_blend(members, build) -> list[float] | None:
    try:
        model = build()
        model.load_state_dict(members[0])
        return model.blend_weights()
    except Exception:  # informational only
        return None


# --- LSTM and Cola-GNN -----------------------------------------------------------

def _run_sequence(city, config, rates, window, split, device, on_unit) -> FitOutput:
    features, _ = impute_causal(rates)
    mean, std = normalization(rates, split)
    is_lstm = isinstance(config, LSTMConfig)
    clim_rate = None
    if is_lstm and config.target == "blend":
        clim_rate = fit_climatology(rates, end=train_end(split)).level(rates.index)
    train = sequence_arrays(features, rates, split.train, mean, std, window, clim_rate)
    val = sequence_arrays(features, rates, split.val, mean, std, window, clim_rate)
    x_test, _, c_test = sequence_arrays(features, rates, split.test, mean, std, window, clim_rate)
    adjacency = None
    if not is_lstm:
        spec = GraphSpec(anchors=False, geo=True, geo_max_hop=1, corr=False, demo=False)
        adjacency = build_graph(spec, city=city, flu_history=rates.iloc[:0]).border_adjacency()

    seeds = [config.seed + i for i in range(max(1, config.n_seeds))]
    test_draws, val_draws, members, results = [], [], [], []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if is_lstm:
            model = MultivariateLSTM(city.n_areas, len(window.horizons),
                                     hidden_size=config.hidden_size, dropout=config.dropout,
                                     blend_init=(blend_init(window.horizons)
                                                 if config.target == "blend" else None)).to(device)
        else:
            model = ColaGNN(city.n_areas, window.lookback, len(window.horizons), adjacency,
                            hidden=config.hidden, conv_channels=config.conv_channels,
                            gnn_hidden=config.hidden, dropout=config.dropout).to(device)
        results.append(train_sequence_model(model, train, val, config=config, device=device,
                                            seed=seed, use_clim=is_lstm))
        model.eval()
        with torch.no_grad():
            def predict(x, c):
                x_t, c_t = torch.from_numpy(x).to(device), torch.from_numpy(c).to(device)
                return model(x_t, c_t).cpu().numpy() if is_lstm else model(x_t).cpu().numpy()
            test_draws.append(predict(x_test, c_test) * std[:, None, :] + mean[:, None, :])
            val_draws.append(predict(val[0], val[2]) * std[:, None, :] + mean[:, None, :])
        members.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if on_unit:
            on_unit(f"{config.kind} seed {seed}")

    # (S, H, N) -> (S, N, H) for the shared frame builder.
    predicted = np.maximum(0.0, np.mean(test_draws, axis=0)).transpose(0, 2, 1)
    val_predicted = np.maximum(0.0, np.mean(val_draws, axis=0)).transpose(0, 2, 1)
    values = rates.to_numpy(dtype=float)
    actual_test = np.stack([[values[t + h] for h in window.horizons] for t in split.test]).transpose(0, 2, 1)
    actual_val = np.stack([[values[t + h] for h in window.horizons] for t in split.val]).transpose(0, 2, 1)
    test = _frame(split.index, split.test, window.horizons, city.areas, predicted, actual_test)
    val = _frame(split.index, split.val, window.horizons, city.areas, val_predicted, actual_val)
    interval_model = fit_intervals(val["predicted"], val["actual"], val["horizon"],
                                   two_sided=config.two_sided_intervals)
    test = attach_intervals(test, interval_model)
    meta = {"intervals": interval_model.to_json(), "best_epochs": [r.best_epoch for r in results]}
    return FitOutput(test=test, val=val, config=config, members=members, meta=meta)


# --- baselines -------------------------------------------------------------------

LAST_OBSERVED_WEEKS = 4
NEAREST_TOLERANCE_WEEKS = 2


def _run_baseline(city, config: BaselineConfig, rates, window, split, on_unit) -> FitOutput:
    """Persistence ("next week looks like this week") or the same week last year.

    Persistence uses the origin week's value; if that week is missing, the most
    recent observed week up to four weeks back; else the training median. The
    seasonal baseline uses the week 52 weeks before the target (or within two
    weeks of it). Both only read weeks at or before the origin.
    """
    values = rates.to_numpy(dtype=float)
    medians = np.nanmedian(values[:train_end(split)], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    def predict(positions):
        out = np.zeros((len(positions), city.n_areas, len(window.horizons)))
        for s, t in enumerate(positions):
            for h_idx, horizon in enumerate(window.horizons):
                for node in range(city.n_areas):
                    out[s, node, h_idx] = _baseline_value(values[:, node], t, horizon,
                                                          config.kind, medians[node])
        return out

    def actual(positions):
        return np.stack([[values[t + h] for h in window.horizons]
                         for t in positions]).transpose(0, 2, 1)

    test = _frame(split.index, split.test, window.horizons, city.areas, predict(split.test),
                  actual(split.test))
    val = _frame(split.index, split.val, window.horizons, city.areas, predict(split.val),
                 actual(split.val))
    interval_model = fit_intervals(val["predicted"], val["actual"], val["horizon"],
                                   two_sided=config.two_sided_intervals)
    test = attach_intervals(test, interval_model)
    if on_unit:
        on_unit(config.kind)
    return FitOutput(test=test, val=val, config=config, meta={"intervals": interval_model.to_json()})


def _baseline_value(series: np.ndarray, origin: int, horizon: int, kind: str, median: float) -> float:
    if kind == "persistence":
        for back in range(0, LAST_OBSERVED_WEEKS + 1):
            if origin - back >= 0 and np.isfinite(series[origin - back]):
                return float(series[origin - back])
        return float(median)
    if kind == "seasonal_naive":
        source = origin + horizon - 52
        for offset in [0] + [o for k in range(1, NEAREST_TOLERANCE_WEEKS + 1) for o in (-k, k)]:
            week = source + offset
            if 0 <= week <= origin and np.isfinite(series[week]):
                return float(series[week])
        return float(median)
    raise ValueError(f"Unknown baseline {kind!r}")
