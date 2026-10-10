"""Saving trained models and loading them as a starting point elsewhere.

A checkpoint holds every ensemble member's weights plus the configuration and
the city it was trained on. Applying it to another city (with or without further
training) re-derives that city's normalisation, seasonal baseline and interval
calibration from the city's own data, as the research transfer experiments did.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .configs import ModelConfig, from_json, to_json

FORMAT = "areafluforecast-checkpoint-1"


def save_checkpoint(path: str | Path, output, *, city_name: str, areas: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "config": to_json(output.config),
                "member_state_dicts": output.members, "city": city_name, "areas": areas,
                "meta": output.meta}, path)
    return path


def load_checkpoint(path: str | Path) -> tuple[ModelConfig, list[dict], dict]:
    """(config, member state_dicts, everything else) from a package or research checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") == FORMAT:
        return from_json(payload["config"]), payload["member_state_dicts"], payload
    # A research run_gnn.py checkpoint.
    members = payload.get("member_state_dicts") or [payload["model_state_dict"]]
    from .configs import GNNConfig
    train = (payload.get("experiment") or {}).get("train", {})
    config = GNNConfig(kind="gnn_st_noadapt", adaptive=bool(train.get("adaptive", False)),
                       channels=int(train.get("channels", 24)),
                       dropout=float(train.get("dropout", 0.25)))
    return config, members, payload
