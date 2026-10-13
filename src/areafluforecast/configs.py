"""Model configurations, presets and the tuning search spaces.

Defaults reproduce the research runs. The search spaces and the way the 12
candidate configurations are drawn are the research `tune.py`'s exactly, so
`tune()` here proposes the same candidates the paper's models were chosen from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import numpy as np


@dataclass(frozen=True)
class GNNConfig:
    """The graph model ("gnn_st").

    Experimental, outside the versioning policy (may change or go in a minor
    release): `refit`, `drop_edge`, `corr_binary` and `two_sided_intervals`.
    """

    kind: str = "gnn_st"
    lookback: int = 16
    lr: float = 3e-3
    weight_decay: float = 3e-3
    epochs: int = 120
    batch_size: int = 16
    dropout: float = 0.25
    channels: int = 24
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    adaptive: bool = True
    adaptive_dim: int = 8
    drop_edge: float = 0.0
    n_blocks: int = 2
    huber_delta: float = 1.0
    pearson_weight: float = 0.3
    grad_clip: float = 1.0
    n_seeds: int = 10
    seed: int = 42
    mc_samples: int = 100
    mc_seed: int = 123
    # Inputs. None = use weather when the city has it.
    use_weather: bool | None = None
    use_static: bool = False
    use_imputed_flag: bool = True
    # Graph. graph=False removes every edge and the learned adjacency: each area
    # is forecast from its own history with shared weights (the "no graph" check).
    graph: bool = True
    anchors: bool = True
    geo_max_hop: int = 3
    corr: bool = True
    corr_binary: bool = False
    corr_threshold: float = 0.85
    similarity_edges: bool = False
    two_sided_intervals: bool = False
    # After the usual fit, retrain each seed from scratch on training + validation
    # weeks for its best epoch count, so the newest held-out weeks are learned
    # from too. The ranges stay sized on the first fit's validation errors (the
    # refit has seen those weeks). Off by default: the research runs never refit.
    refit: bool = False


@dataclass(frozen=True)
class LSTMConfig:
    kind: str = "lstm"
    lookback: int = 8
    hidden_size: int = 64
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 120
    batch_size: int = 32
    target: str = "level"          # or "blend": residual from the seasonal blend
    n_seeds: int = 10
    seed: int = 42
    two_sided_intervals: bool = False


@dataclass(frozen=True)
class ColaGNNConfig:
    kind: str = "colagnn"
    lookback: int = 20
    hidden: int = 32
    conv_channels: int = 16
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 150
    batch_size: int = 32
    n_seeds: int = 10
    seed: int = 42
    two_sided_intervals: bool = False


@dataclass(frozen=True)
class BaselineConfig:
    kind: str = "persistence"      # or "seasonal_naive"
    two_sided_intervals: bool = False


@dataclass(frozen=True)
class ArimaConfig:
    """One seasonal ARIMA per area (the paper's "Seasonal ARIMA" row).

    The (p, d, q) order is chosen per area by walk-forward RMSE on the
    validation origins, at the horizon being forecast. `harmonics` sin/cos pairs
    of the day of year enter as regressors for the annual cycle (0 = plain
    ARIMA). The grid is p < max_p, d < max_d, q < max_q. Needs statsmodels
    (`pip install areafluforecast[arima]`). Not trained by gradient descent, so
    it has no seeds and is never tuned.
    """

    kind: str = "arima"
    harmonics: int = 3
    max_p: int = 4
    max_d: int = 2
    max_q: int = 4
    two_sided_intervals: bool = False


ModelConfig = GNNConfig | LSTMConfig | ColaGNNConfig | BaselineConfig | ArimaConfig

PRESETS: dict[str, ModelConfig] = {
    # The package default: the paper's graph model without the static
    # demographic block, which the paper's ablations found inside noise at 1-2
    # weeks and harmful at 4. Weather is used when the city has it.
    "gnn_st": GNNConfig(),
    # The exact paper configuration (needs static characteristics; anchors and
    # weather when the city has them).
    "gnn_st_paper": GNNConfig(kind="gnn_st_paper", use_weather=True, use_static=True,
                              similarity_edges=True),
    # Same model with the graph removed, to see what the graph adds on your data.
    "gnn_st_nograph": GNNConfig(kind="gnn_st_nograph", graph=False, adaptive=False,
                                anchors=False),
    # For cross-city transfer: the learned adjacency is sized to one city's areas.
    "gnn_st_noadapt": GNNConfig(kind="gnn_st_noadapt", adaptive=False),
    "lstm": LSTMConfig(),
    "colagnn": ColaGNNConfig(),
    "persistence": BaselineConfig(kind="persistence"),
    "seasonal_naive": BaselineConfig(kind="seasonal_naive"),
    "arima": ArimaConfig(),
}
DEFAULT_MODELS = ("gnn_st", "lstm", "colagnn", "arima")
BASELINES = ("persistence", "seasonal_naive")


def resolve(model: str | ModelConfig, overrides: dict | None = None) -> ModelConfig:
    """A preset name or a config object, with optional field overrides."""
    if isinstance(model, str):
        if model not in PRESETS:
            raise ValueError(f"Unknown model {model!r}. Choose from {sorted(PRESETS)} or pass a config.")
        config = PRESETS[model]
    else:
        config = model
    if overrides:
        valid = {f.name for f in fields(config)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"{config.kind} has no settings {sorted(unknown)}.")
        config = replace(config, **overrides)
    return config


# GNNConfig's default mc_seed sits this far above its default seed.
MC_SEED_OFFSET = 81


def with_seed(config: ModelConfig, seed: int | None) -> ModelConfig:
    """`config` with every random seed derived from `seed` (unchanged when None)."""
    if seed is None or not is_learned(config):
        return config
    changes = {"seed": int(seed)}
    if isinstance(config, GNNConfig):
        changes["mc_seed"] = int(seed) + MC_SEED_OFFSET
    return replace(config, **changes)


def is_learned(config: ModelConfig) -> bool:
    return not isinstance(config, (BaselineConfig, ArimaConfig))


def to_json(config: ModelConfig) -> dict:
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, tuple):
            payload[key] = list(value)
    payload["type"] = type(config).__name__
    return payload


def from_json(payload: dict) -> ModelConfig:
    payload = dict(payload)
    cls = {c.__name__: c for c in (GNNConfig, LSTMConfig, ColaGNNConfig, BaselineConfig,
                                              ArimaConfig)}[
        payload.pop("type")]
    valid = {f.name for f in fields(cls)}
    clean = {k: (tuple(v) if isinstance(v, list) else v) for k, v in payload.items() if k in valid}
    return cls(**clean)


# --- research CLI flags, for tuning and parity -------------------------------

_FLAG_FIELDS = {
    "--lr": ("lr", float), "--weight-decay": ("weight_decay", float),
    "--dropout": ("dropout", float), "--channels": ("channels", int),
    "--epochs": ("epochs", int), "--hidden-size": ("hidden_size", int),
    "--hidden": ("hidden", int), "--lookback": ("lookback", int),
    "--target": ("target", str), "--n-seeds": ("n_seeds", int),
    "--batch-size": ("batch_size", int), "--conv-channels": ("conv_channels", int),
}


def with_flags(config: ModelConfig, flags: list[str]) -> ModelConfig:
    """Apply research-style flags (["--lr", "0.003", ...]) to a config."""
    changes = {}
    for flag, value in zip(flags[::2], flags[1::2]):
        if flag not in _FLAG_FIELDS:
            raise ValueError(f"Unknown flag {flag!r}.")
        name, cast = _FLAG_FIELDS[flag]
        changes[name] = cast(value)
    return replace(config, **changes)


_GNN_SPACE = {"--lr": [1e-3, 3e-3, 1e-2], "--weight-decay": [3e-4, 1e-3, 3e-3, 1e-2],
              "--dropout": [0.1, 0.25, 0.4], "--channels": [16, 24, 32],
              "--epochs": [80, 120, 200]}
SEARCH_SPACES = {
    "gnn_st": _GNN_SPACE,
    "gnn_st_nograph": _GNN_SPACE,
    "lstm": {"--lr": [3e-4, 1e-3, 3e-3], "--weight-decay": [1e-5, 1e-4, 1e-3],
             "--dropout": [0.0, 0.1, 0.25], "--hidden-size": [32, 64, 128],
             "--lookback": [8, 16, 26], "--epochs": [80, 120, 200],
             "--target": ["level", "blend"]},
    "colagnn": {"--lr": [3e-4, 1e-3, 3e-3], "--weight-decay": [0.0, 5e-4, 5e-3],
                "--dropout": [0.1, 0.2, 0.4], "--hidden": [16, 32, 64],
                "--lookback": [12, 20, 26], "--epochs": [100, 150, 200]},
}
SEARCH_SPACES["gnn_st_paper"] = _GNN_SPACE
SEARCH_SPACES["gnn_st_noadapt"] = _GNN_SPACE
K_CONFIGS = 12


def _stable_seed(arm: str) -> int:
    return int.from_bytes(arm.encode(), "little") % (2 ** 32)


def candidate_flags(arm: str, k: int = K_CONFIGS) -> list[list[str]]:
    """K flag lists: [] (the default) then K-1 random draws with a fixed seed.

    The research drew gnn_st's candidates under the name "gnn_st"; the paper
    variants reuse that stream so they get the same candidates.
    """
    space = SEARCH_SPACES[arm]
    stream = "gnn_st" if arm.startswith("gnn_st") and arm != "gnn_st_nograph" else arm
    rng = np.random.default_rng(_stable_seed(stream))
    configs, seen = [[]], {()}
    while len(configs) < k:
        flags = []
        for flag, values in space.items():
            flags += [flag, str(values[rng.integers(len(values))])]
        if tuple(flags) not in seen:
            seen.add(tuple(flags))
            configs.append(flags)
    return configs
