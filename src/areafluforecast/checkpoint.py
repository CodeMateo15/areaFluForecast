"""Saving trained models and loading them as a starting point elsewhere.

A checkpoint holds every ensemble member's weights plus the configuration and
the city it was trained on. Applying it to another city (with or without further
training) re-derives that city's normalisation, seasonal baseline and interval
calibration from the city's own data, as the research transfer experiments did.

A checkpoint is trained for one horizon. To start a 1-4 week forecast from saved
models, pass one file per horizon: ``init_from={1: "h1.pt", 2: "h2.pt", ...}``.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

from .configs import ModelConfig, from_json, to_json

FORMAT = "areafluforecast-checkpoint-1"
FORMAT_PREFIX = "areafluforecast-checkpoint-"


def save_checkpoint(path: str | Path, output, *, city_name: str, areas: list[str]) -> Path:
    from . import __version__

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "config": to_json(output.config),
                "member_state_dicts": output.members, "city": city_name, "areas": areas,
                "meta": output.meta, "package_version": __version__,
                "torch_version": str(torch.__version__)}, path)
    return path


def load_checkpoint(path: str | Path, *, trust_research_file: bool = False
                    ) -> tuple[ModelConfig, list[dict], dict]:
    """(config, member state_dicts, everything else) from a package or research checkpoint.

    Files are opened with ``torch.load(weights_only=True)``, which cannot run code.
    A research ``run_gnn.py`` checkpoint that needs full unpickling is opened only
    with ``trust_research_file=True``; do that only for files you made yourself.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if not trust_research_file:
            raise ValueError(
                f"{path} holds more than weights and settings, so it was not opened: loading "
                "it could run code. If it is a research checkpoint you made yourself, pass "
                "trust_research_file=True to load_checkpoint.") from error
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not an areaFluForecast checkpoint.")
    stamp = payload.get("format")
    if stamp == FORMAT:
        return from_json(payload["config"]), payload["member_state_dicts"], payload
    if isinstance(stamp, str) and stamp.startswith(FORMAT_PREFIX):
        raise ValueError(f"{path} was saved in format {stamp!r}, newer than this version of "
                         f"areaFluForecast reads ({FORMAT!r}). Upgrade the package.")
    if stamp is not None or not ("member_state_dicts" in payload or "model_state_dict" in payload):
        raise ValueError(f"{path} is not an areaFluForecast checkpoint.")
    # A research run_gnn.py checkpoint.
    members = payload.get("member_state_dicts") or [payload["model_state_dict"]]
    from .configs import GNNConfig
    train = (payload.get("experiment") or {}).get("train", {})
    config = GNNConfig(kind="gnn_st_noadapt", adaptive=bool(train.get("adaptive", False)),
                       channels=int(train.get("channels", 24)),
                       dropout=float(train.get("dropout", 0.25)))
    return config, members, payload


@dataclass
class InitWeights:
    """Saved weights to start the graph model from, matched to each horizon."""

    config: ModelConfig
    members: dict[int, list[dict]]
    feature_names: dict[int, list[str] | None]
    digests: dict[int, str]          # sha256 of each horizon's file, to match backtests


def _horizon_of(payload: dict) -> int | None:
    horizon = (payload.get("meta") or {}).get("horizon")
    return None if horizon is None else int(horizon)


def load_init(init_from, horizons) -> InitWeights:
    """Weights for every horizon in `horizons` from one checkpoint or a {horizon: path} map.

    A single file may only serve the horizon it was trained for: a 1-week model
    used 4 weeks ahead would be wrong without any error.
    """
    horizons = sorted(int(h) for h in horizons)
    if isinstance(init_from, dict):
        sources = {int(h): p for h, p in init_from.items()}
        missing = [h for h in horizons if h not in sources]
        if missing:
            raise ValueError(f"init_from has no checkpoint for {missing} weeks ahead; it covers "
                             f"{sorted(sources)}. Pass one file per horizon you forecast.")
    else:
        sources = {h: init_from for h in horizons}

    config, members, features, digests, loaded = None, {}, {}, {}, {}
    for horizon in horizons:
        path = sources[horizon]
        if path not in loaded:
            loaded[path] = load_checkpoint(path)
            digests[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        this_config, this_members, payload = loaded[path]
        trained_for = _horizon_of(payload)
        if trained_for is not None and trained_for != horizon:
            hint = ("pass init_from={1: 'h1.pt', 2: 'h2.pt', ...}, one file per horizon"
                    if not isinstance(init_from, dict) else "check the file for each horizon")
            raise ValueError(f"{path} was trained to forecast {trained_for} weeks ahead, not "
                             f"{horizon}. Use horizons=({trained_for},), or {hint}.")
        if trained_for is None and len(horizons) > 1 and not isinstance(init_from, dict):
            raise ValueError(f"{path} does not record which horizon it was trained for, so it "
                             "can only be used for one horizon at a time.")
        if config is None:
            config = this_config
        elif to_json(this_config) != to_json(config):
            raise ValueError("The checkpoints in init_from were saved with different model "
                             "settings; they must come from one training run.")
        members[horizon] = this_members
        features[horizon] = (payload.get("meta") or {}).get("feature_names")
    return InitWeights(config=config, members=members, feature_names=features,
                       digests={h: digests[sources[h]] for h in horizons})
