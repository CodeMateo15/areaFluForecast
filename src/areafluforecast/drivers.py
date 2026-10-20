"""What drove a frozen graph model's forecast, in the forecast's own units.

For one area, horizon and origin, the forecast is split into the inputs that moved
it away from a typical week:

    level      this week's reading (the origin week's rate, which also sets the
               persistence half of the model's baseline)
    trend      the three weeks before it
    history    the twelve weeks before those
    weather    the origin week's weather in the area
    calendar   the seasonal curve at the target week (the model's other baseline)
    <area>     each other area's inputs: what spills over through the graph

Values are Owen values (Owen 1977): Shapley values (Shapley 1953) over the first
five players plus one "neighbors" group, the group's share split again between
its areas. An absent input is set to its mean over the training origins within
`SAME_TIME_WEEKS` of the same time of year, so the values add up exactly to
(forecast - the forecast for a usual week at this time of year), and `calendar`
only carries how far the seasonal curve's target week sits from those weeks'. They are
estimated by sampling orderings that keep the neighbors together (the reverse of
each ordering too, which lowers the noise); `se` is the sampling error.

The model is the frozen ensemble's deterministic (eval-mode) mean, not the
MC-Dropout mean the published median is; `explain` in scripts/live_forecast.py
reports the small difference as its own row. As with any attribution, the
values describe what the model responds to, not what causes flu to spread, and
inputs set to their typical value while correlated ones are not form weeks the
model never saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .city import City
from .configs import GNNConfig, resolve
from .engine import gnn_inputs, make_split, make_window
from .errors import DataError
from .forecast import load_frozen

METHOD = 1                             # bump when the attribution changes: cached weeks redo
OWN = ("level", "trend", "history", "weather", "calendar")
LEVEL_LAGS, TREND_LAGS = 1, 4          # lag 0 | lags 1-3 | lags 4 and up
CHUNK = 1024                           # graphs per forward pass
SAME_TIME_WEEKS = 4                    # "a usual week at this time of year": +-4 weeks


@dataclass
class _Ensemble:
    """One horizon's frozen members and the tensors for every scored origin."""

    members: list[torch.nn.Module]
    relations: list[tuple]
    X: torch.Tensor           # (S, N, F) per scored origin
    anchors: torch.Tensor     # (S, n)
    clim: torch.Tensor        # (S, n, 1)
    train: tuple[np.ndarray, np.ndarray, np.ndarray]   # X, anchors, clim of training origins
    train_dates: pd.DatetimeIndex
    dates: pd.DatetimeIndex   # the scored origins
    positions: list[int]
    mean: np.ndarray          # (n,) rate scaling
    std: np.ndarray
    lookback: int
    channels: int
    n_areas: int
    blend_share: float

    def typical(self, row: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean inputs of the training origins near this origin's time of year."""
        day = self.dates[row].dayofyear
        gap = np.abs(self.train_dates.dayofyear.to_numpy() - day)
        near = np.minimum(gap, 365 - gap) <= 7 * SAME_TIME_WEEKS
        return tuple(torch.from_numpy(np.ascontiguousarray(a[near].mean(axis=0)))
                     for a in self.train)


def _batched(relations: list[tuple], copies: int, n_nodes: int) -> list[tuple]:
    """The edge sets of `copies` disjoint copies of the graph."""
    out = []
    for name, index, weight in relations:
        offsets = torch.arange(copies, dtype=index.dtype).repeat_interleave(index.size(1)) * n_nodes
        out.append((name, index.repeat(1, copies) + offsets, weight.repeat(copies)))
    return out


def _mix(mixer, h: torch.Tensor, relations: list[tuple], copies: int, n_nodes: int) -> torch.Tensor:
    """MultiRelationalMixer.forward in eval mode, for a batch of disjoint graphs: the
    learned adjacency is applied per copy instead of across the whole batch."""
    by_name = {name: (index, weight) for name, index, weight in relations}
    parts = [conv(h, *by_name[name]) for name, conv in zip(mixer.relation_names, mixer.convs)]
    adjacency = mixer.adaptive_adjacency()
    if adjacency is not None:
        per_copy = adjacency @ h.view(copies, n_nodes, -1)
        parts.append(mixer.adaptive_project(per_copy.reshape(copies * n_nodes, -1)))
    weights = F.softmax(mixer.gate, dim=0)
    mixed = sum(w * part for w, part in zip(weights, parts))
    return mixer.norm(h + F.relu(mixed))


def batched_forward(model, X: torch.Tensor, relations: list[tuple], anchor: torch.Tensor,
                    clim: torch.Tensor) -> torch.Tensor:
    """SpatioTemporalGNN.forward (eval mode, no city-wide inputs) for B graphs at once:
    X (B, N, F), anchor (B, n), clim (B, n, H) -> (B, n, H) normalised forecasts."""
    copies, n_nodes, n_feat = X.shape
    encoded = model._encode(X.reshape(copies * n_nodes, n_feat)).view(copies, n_nodes, -1)
    return _from_encoded(model, encoded, relations, anchor, clim)


def _from_encoded(model, encoded: torch.Tensor, relations: list[tuple], anchor: torch.Tensor,
                  clim: torch.Tensor) -> torch.Tensor:
    """The rest of the forward pass from each node's encoded history (B, N, C). Each
    node's encoding depends on its own inputs only, which is what makes explaining
    cheap: a node takes few distinct inputs across coalitions."""
    copies, n_nodes, _ = encoded.shape
    big = _batched(relations, copies, n_nodes)
    h = encoded.reshape(copies * n_nodes, -1)
    for mixer in model.mixers:
        h = _mix(mixer, h, big, copies, n_nodes)
    residual = model.output(F.relu(model.head_fuse(h))).view(copies, n_nodes, -1)
    share = torch.sigmoid(model.blend_logit)
    return share * anchor.unsqueeze(-1) + (1.0 - share) * clim + residual[:, :anchor.size(1)]


def _players(target: int, n_areas: int, n_nodes: int, n_feat: int, lookback: int,
             channels: int) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Player names and which inputs each one sets: masks over X (P, N, F), the
    anchors (P, n) and the seasonal curve (P, n)."""
    names = list(OWN) + [f"area:{j}" for j in range(n_areas) if j != target]
    on_x = torch.zeros(len(names), n_nodes, n_feat, dtype=torch.bool)
    on_anchor = torch.zeros(len(names), n_areas, dtype=torch.bool)
    on_clim = torch.zeros(len(names), n_areas, dtype=torch.bool)
    span = lookback * channels
    on_x[0, target, :LEVEL_LAGS * channels] = True
    on_anchor[0, target] = True
    on_x[1, target, LEVEL_LAGS * channels:TREND_LAGS * channels] = True
    on_x[2, target, TREND_LAGS * channels:span] = True
    on_x[3, target, span:] = True                  # weather (and static inputs, if any)
    on_clim[4, target] = True
    for p, j in enumerate([j for j in range(n_areas) if j != target], start=len(OWN)):
        on_x[p, j] = True
        on_anchor[p, j] = on_clim[p, j] = True
    return names, on_x, on_anchor, on_clim


def _orderings(n_own: int, n_group: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Player orderings that keep the group (players n_own..) together, in pairs:
    each ordering followed by its reverse."""
    out = []
    for _ in range(max(1, n_perm // 2)):
        groups = rng.permutation(n_own + 1)
        inner = rng.permutation(np.arange(n_own, n_own + n_group))
        order = np.concatenate([inner if g == n_own else [g] for g in groups])
        out.extend([order, order[::-1]])
    return np.asarray(out)


N_OWN_X = 4                             # own players that set X: level, trend, history, weather


@torch.no_grad()
def _values(ens: _Ensemble, row: int, target: int, coalitions: torch.Tensor, masks) -> np.ndarray:
    """Ensemble-mean normalised forecast for `target` with each coalition's inputs set
    to this origin's values and the rest typical: (K,).

    Node rows are encoded once: the target's 2^4 mixes of its own players, and the
    actual and typical row of every other node. A coalition then only picks rows."""
    on_x, on_anchor, on_clim = masks
    x, a, c = ens.X[row], ens.anchors[row], ens.clim[row]
    tx, ta, tc = ens.typical(row)
    n_nodes = x.size(0)
    mixes = torch.tensor([[(m >> b) & 1 for b in range(N_OWN_X)] for m in range(2 ** N_OWN_X)],
                         dtype=torch.float32)                         # (16, 4)
    own = (mixes @ on_x[:N_OWN_X, target].float()) > 0                # (16, F)
    rows = torch.cat([torch.where(own, x[target], tx[target]), x, tx])  # (16 + 2N, F)
    # Each node's row: the target picks its mix, node j its actual (16 + j) or typical
    # (16 + N + j) row by whether its player is in.
    owner = torch.full((n_nodes,), -1, dtype=torch.long)
    for p in range(len(OWN), on_x.size(0)):
        owner[on_x[p].any(dim=1)] = p
    bits = 2 ** torch.arange(N_OWN_X)
    out = np.zeros(len(coalitions))
    encoded = [m._encode(rows) for m in ens.members]
    for start in range(0, len(coalitions), CHUNK):
        k = coalitions[start:start + CHUNK]
        index = torch.empty(len(k), n_nodes, dtype=torch.long)
        for node in range(n_nodes):
            if node == target:
                index[:, node] = (k[:, :N_OWN_X].long() * bits).sum(dim=1)
            elif owner[node] >= 0:
                index[:, node] = torch.where(k[:, owner[node]], 16 + node, 16 + n_nodes + node)
            else:                                   # a node no player sets: always typical
                index[:, node] = 16 + n_nodes + node
        kf = k.float()
        anchor = torch.where((kf @ on_anchor.float()) > 0, a, ta)
        clim = torch.where(((kf @ on_clim.float()) > 0).unsqueeze(-1), c, tc)
        total = sum(_from_encoded(m, enc[index], ens.relations, anchor, clim)[:, target, 0]
                    for m, enc in zip(ens.members, encoded))
        out[start:start + len(k)] = (total / len(ens.members)).numpy()
    return out


def _explain_one(ens: _Ensemble, row: int, target: int, n_perm: int,
                 rng: np.random.Generator) -> tuple[list[str], np.ndarray, np.ndarray, float, float]:
    """(player names, values, standard errors, base, output) in normalised units."""
    n_nodes, n_feat = ens.X.shape[1:]
    names, *masks = _players(target, ens.n_areas, n_nodes, n_feat, ens.lookback, ens.channels)
    n_players = len(names)
    orders = _orderings(len(OWN), n_players - len(OWN), n_perm, rng)
    # Coalition k of ordering o holds its first k players.
    position = np.empty_like(orders)
    np.put_along_axis(position, orders, np.arange(n_players)[None, :].repeat(len(orders), 0), 1)
    steps = np.arange(n_players + 1)
    coalitions = torch.from_numpy(position[:, None, :] < steps[None, :, None])
    values = _values(ens, row, target, coalitions.reshape(-1, n_players), masks)
    values = values.reshape(len(orders), n_players + 1)
    gains = np.diff(values, axis=1)                    # (O, P) by step
    by_player = np.take_along_axis(gains, position, axis=1)
    pairs = by_player.reshape(-1, 2, n_players).mean(axis=1)
    se = pairs.std(axis=0, ddof=1) / np.sqrt(len(pairs)) if len(pairs) > 1 else np.zeros(n_players)
    return names, pairs.mean(axis=0), se, float(values[0, 0]), float(values[0, -1])


def _ensemble(city: City, checkpoint, config: GNNConfig, rates: pd.DataFrame, horizon: int,
              future: pd.DatetimeIndex) -> _Ensemble:
    anchor = pd.Timestamp(checkpoint.meta["anchor"])
    window = make_window(config, horizon, anchor, future[-1])
    split = make_split(rates.index, window)
    inputs = gnn_inputs(city, config, rates, window, split, torch.device("cpu"))
    samples = inputs.samples
    if samples.n_global:
        raise DataError("explain() covers graph models without city-wide inputs.")
    if checkpoint.feature_names is not None and list(checkpoint.feature_names) != samples.feature_names:
        raise DataError(f"{checkpoint.path} was trained on other input features.")
    members = []
    for state in checkpoint.members:
        model = inputs.build()
        model.load_state_dict(state)
        members.append(model.eval())
    train = samples.subset(split.train)
    scored = samples.subset(split.test)
    as_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return _Ensemble(
        members=members, relations=inputs.relations, X=as_tensor(scored.X),
        anchors=as_tensor(scored.anchors), clim=as_tensor(scored.clim),
        train=(train.X, train.anchors, train.clim), train_dates=rates.index[split.train],
        dates=rates.index[split.test], positions=list(split.test), mean=inputs.norm.flu_mean.flatten(),
        std=inputs.norm.flu_std.flatten(), lookback=samples.lookback,
        channels=samples.temporal_channels, n_areas=city.n_areas,
        blend_share=float(members[0].blend_weights()[0]))


def explain(city: City, models_dir: str | Path, *, origins=None, horizons=(1, 2, 3, 4),
            n_perm: int | dict = 256, seed: int = 0, model: str = "gnn_st",
            config: dict | None = None) -> pd.DataFrame:
    """What drove the frozen `model` in `models_dir` at each origin in `origins` (dates;
    default the newest week), one row per (origin, horizon, area, driver).

    `n_perm` is the number of sampled orderings, or {origin: number}; `config` the
    settings overrides the model was saved with, as forecast(configs=) takes them. Columns:
    origin_date, target_date, horizon, area, driver (level, trend, history, weather,
    calendar, or neighbor with its area in source_area), value and se in rate units,
    base (the forecast for a usual week at this time of year), output (the model's forecast; base plus
    the values), blend_share (the model's weight on this week's level against the
    seasonal curve) and actual (the target week, when observed).
    """
    config = resolve(model, config)
    if not isinstance(config, GNNConfig):
        raise ValueError(f"explain() covers the graph models; {model!r} is not one.")
    horizons = tuple(sorted(int(h) for h in horizons))
    frozen = load_frozen(models_dir, city, {model: config}, horizons)
    last = city.rates.index.max()
    future = pd.date_range(last + pd.Timedelta(days=7), periods=max(horizons), freq="7D")
    rates = pd.concat([city.rates, pd.DataFrame(np.nan, index=future, columns=city.areas)])
    rates.index.name = city.rates.index.name
    origins = [last] if origins is None else [pd.Timestamp(o) for o in origins]
    actual = rates.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for horizon in horizons:
        ens = _ensemble(city, frozen[(model, horizon)], config, rates, horizon, future)
        lookup = {rates.index[p]: r for r, p in enumerate(ens.positions)}
        for origin in origins:
            if origin not in lookup:
                raise DataError(f"{origin.date()} is not a scored origin of the {horizon}-week "
                                f"model (it scores {min(lookup).date()} to {max(lookup).date()}).")
            row = lookup[origin]
            position = ens.positions[row]
            perms = n_perm.get(origin, 256) if isinstance(n_perm, dict) else n_perm
            for target, area in enumerate(city.areas):
                names, values, se, base, output = _explain_one(ens, row, target, perms, rng)
                scale, shift = ens.std[target], ens.mean[target]
                common = {"origin_date": origin, "target_date": rates.index[position + horizon],
                          "horizon": horizon, "area": area, "base": base * scale + shift,
                          "output": output * scale + shift, "blend_share": ens.blend_share,
                          "actual": actual[position + horizon, target]}
                for name, value, error in zip(names, values, se):
                    source = city.areas[int(name.split(":")[1])] if name.startswith("area:") else ""
                    rows.append({**common, "driver": "neighbor" if source else name,
                                 "source_area": source, "value": value * scale,
                                 "se": error * scale})
    columns = ["origin_date", "target_date", "horizon", "area", "driver", "source_area", "value",
               "se", "base", "output", "blend_share", "actual"]
    return pd.DataFrame(rows, columns=columns)
