"""Saving trained models and loading them as a starting point elsewhere.

A checkpoint holds every ensemble member's weights plus the configuration and
the city it was trained on. Applying it to another city (with or without further
training) re-derives that city's normalisation, seasonal baseline and interval
calibration from the city's own data, as the research transfer experiments did.
Transferring a graph model between regions follows Panagopoulos et al. (2021,
AAAI 35:4838).

A checkpoint is trained for one horizon. To start a 1-4 week forecast from saved
models, pass one per horizon: ``init_from={1: "h1.pt", 2: "h2.pt", ...}``, or the
bundled Boston models, ``init_from=af.pretrained("boston")``.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .engine import FitOutput

from .configs import GNNConfig, ModelConfig, from_json, to_json

FORMAT = "areafluforecast-checkpoint-1"
FORMAT_PREFIX = "areafluforecast-checkpoint-"
PRETRAINED = ("boston",)


@dataclass(kw_only=True)
class Checkpoint:
    """A trained model saved for one horizon: every ensemble member's weights,
    the settings it was trained with, and where it came from."""

    config: ModelConfig
    members: list[dict]                      # one state_dict per ensemble member
    horizon: int | None                      # weeks ahead it forecasts (None: unknown)
    city: str | None = None
    areas: list[str] = field(default_factory=list)
    feature_names: list[str] | None = None   # the inputs it expects, in order
    package_version: str | None = None       # the areaFluForecast version that saved it
    meta: dict = field(default_factory=dict)
    path: Path | None = None
    digest: str | None = None                # sha256 of the file, to match backtests

    @classmethod
    def load(cls, path: str | Path, *, trust_research_file: bool = False) -> Checkpoint:
        """Open a checkpoint saved by this package (or by the research code).

        Files are opened with ``torch.load(weights_only=True)``, which cannot run
        code. A research ``run_gnn.py`` checkpoint that needs full unpickling is
        opened only with ``trust_research_file=True``; do that only for files you
        made yourself.
        """
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as error:
            if not trust_research_file:
                raise ValueError(
                    f"{path} holds more than weights and settings, so it was not opened: "
                    "loading it could run code. If it is a research checkpoint you made "
                    "yourself, pass trust_research_file=True.") from error
            payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not an areaFluForecast checkpoint.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stamp = payload.get("format")
        if stamp == FORMAT:
            meta = payload.get("meta") or {}
            horizon = meta.get("horizon")
            return cls(config=from_json(payload["config"]), members=payload["member_state_dicts"],
                       horizon=None if horizon is None else int(horizon),
                       city=payload.get("city"), areas=list(payload.get("areas") or []),
                       feature_names=meta.get("feature_names"),
                       package_version=payload.get("package_version"), meta=meta, path=path,
                       digest=digest)
        if isinstance(stamp, str) and stamp.startswith(FORMAT_PREFIX):
            raise ValueError(f"{path} was saved in format {stamp!r}, newer than this version of "
                             f"areaFluForecast reads ({FORMAT!r}). Upgrade the package.")
        if stamp is not None or not ("member_state_dicts" in payload
                                     or "model_state_dict" in payload):
            raise ValueError(f"{path} is not an areaFluForecast checkpoint.")
        # A research run_gnn.py checkpoint.
        members = payload.get("member_state_dicts") or [payload["model_state_dict"]]
        train = (payload.get("experiment") or {}).get("train", {})
        config = GNNConfig(kind="gnn_st_noadapt", adaptive=bool(train.get("adaptive", False)),
                           channels=int(train.get("channels", 24)),
                           dropout=float(train.get("dropout", 0.25)))
        return cls(config=config, members=members, horizon=None, meta=payload, path=path,
                   digest=digest)

    def save(self, path: str | Path) -> Path:
        """Write the checkpoint; `Checkpoint.load` reads it back."""
        from . import __version__

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "horizon": self.horizon, "feature_names": self.feature_names}
        torch.save({"format": FORMAT, "config": to_json(self.config),
                    "member_state_dicts": self.members, "city": self.city,
                    "areas": list(self.areas), "meta": meta,
                    "package_version": self.package_version or __version__,
                    "torch_version": str(torch.__version__)}, path)
        self.path = path
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def __repr__(self) -> str:
        where = f", city={self.city!r}" if self.city else ""
        return (f"Checkpoint({self.config.kind}, horizon={self.horizon}, "
                f"members={len(self.members)}{where})")


def save_checkpoint(path: str | Path, output: FitOutput, *, city_name: str, areas: list[str]) -> Path:
    """Save a fitted model (an engine FitOutput)."""
    return Checkpoint(config=output.config, members=output.members,
                      horizon=output.meta.get("horizon"), city=city_name, areas=list(areas),
                      feature_names=output.meta.get("feature_names"),
                      meta=output.meta).save(path)


def load_checkpoint(path: str | Path, *, trust_research_file: bool = False
                    ) -> tuple[ModelConfig, list[dict], dict]:
    """(config, member state_dicts, everything else), as in 0.1. Use `Checkpoint.load`."""
    checkpoint = Checkpoint.load(path, trust_research_file=trust_research_file)
    return checkpoint.config, checkpoint.members, {"meta": checkpoint.meta, "city": checkpoint.city,
                                                   "areas": checkpoint.areas}


def pretrained(name: str = "boston", horizons: Iterable[int] = (1, 2, 3, 4)
               ) -> dict[int, Checkpoint]:
    """The bundled pretrained models, one per horizon, ready for `init_from=`.

    "boston": the transferable graph model (gnn_st_noadapt) trained on every
    Boston week through May 2026, with weather and no static characteristics.
    Backtest it on your own data before relying on it:

        weights = af.pretrained("boston")
        bt = af.backtest(city, models=[], init_from=weights, zero_shot=True)
    """
    if name not in PRETRAINED:
        raise ValueError(f"Bundled pretrained models: {', '.join(PRETRAINED)}. Got {name!r}.")
    folder = resources.files("areafluforecast") / "pretrained" / name
    out = {}
    for horizon in sorted(int(h) for h in horizons):
        source = folder / f"gnn_st_noadapt_h{horizon}.pt"
        if not source.is_file():
            raise ValueError(f"No bundled {name} model for {horizon} weeks ahead; "
                             "they cover 1-4 weeks.")
        with resources.as_file(source) as path:
            out[horizon] = Checkpoint.load(path)
    return out


def describe_init(init_from: InitSource | None) -> str | dict | None:
    """init_from as JSON-friendly text, for the saved run's record."""
    if init_from is None:
        return None
    if isinstance(init_from, dict):
        return {str(h): describe_init(c) for h, c in init_from.items()}
    path = getattr(init_from, "path", None) if not isinstance(init_from, (str, Path)) else init_from
    return str(path) if path is not None else repr(init_from)


# What init_from accepts: one saved model, or one per horizon.
InitSource = str | Path | Checkpoint | dict[int, str | Path | Checkpoint]


@dataclass
class InitWeights:
    """Saved weights to start the graph model from, matched to each horizon."""

    config: ModelConfig
    members: dict[int, list[dict]]
    feature_names: dict[int, list[str] | None]
    digests: dict[int, str]          # sha256 of each horizon's file, to match backtests


def load_init(init_from: InitSource, horizons: Iterable[int]) -> InitWeights:
    """Weights for every horizon in `horizons` from one checkpoint or a {horizon: checkpoint}
    map, where each checkpoint is a path or a `Checkpoint`.

    A single checkpoint may only serve the horizon it was trained for: a 1-week
    model used 4 weeks ahead would be wrong without any error.
    """
    horizons = sorted(int(h) for h in horizons)
    is_map = isinstance(init_from, dict)
    if is_map:
        sources = {int(h): p for h, p in init_from.items()}
        missing = [h for h in horizons if h not in sources]
        if missing:
            raise ValueError(f"init_from has no checkpoint for {missing} weeks ahead; it covers "
                             f"{sorted(sources)}. Pass one per horizon you forecast.")
    else:
        sources = {h: init_from for h in horizons}

    config, loaded, out = None, {}, {}
    for horizon in horizons:
        source = sources[horizon]
        key = id(source) if isinstance(source, Checkpoint) else str(source)
        if key not in loaded:
            loaded[key] = source if isinstance(source, Checkpoint) else Checkpoint.load(source)
        checkpoint = loaded[key]
        name = checkpoint.path or checkpoint
        if checkpoint.horizon is not None and checkpoint.horizon != horizon:
            hint = ("pass init_from={1: 'h1.pt', 2: 'h2.pt', ...}, one per horizon"
                    if not is_map else "check the checkpoint for each horizon")
            raise ValueError(f"{name} was trained to forecast {checkpoint.horizon} weeks ahead, "
                             f"not {horizon}. Use horizons=({checkpoint.horizon},), or {hint}.")
        if checkpoint.horizon is None and len(horizons) > 1 and not is_map:
            raise ValueError(f"{name} does not record which horizon it was trained for, so it "
                             "can only be used for one horizon at a time.")
        if config is None:
            config = checkpoint.config
        elif to_json(checkpoint.config) != to_json(config):
            raise ValueError("The checkpoints in init_from were saved with different model "
                             "settings; they must come from one training run.")
        out[horizon] = checkpoint
    return InitWeights(config=config, members={h: c.members for h, c in out.items()},
                       feature_names={h: c.feature_names for h, c in out.items()},
                       digests={h: c.digest for h, c in out.items()})
