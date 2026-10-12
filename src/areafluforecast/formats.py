"""Version stamps for the folders the package saves (city, backtest, forecast).

Each saved JSON records its format and the package version that wrote it. A
reader accepts its own format and unstamped files from version 0.1/0.2, and
refuses a newer format with a clear message instead of misreading it.
"""

from __future__ import annotations

from pathlib import Path

FORMATS = {"city": "areafluforecast-city-1", "backtest": "areafluforecast-backtest-1",
           "forecast": "areafluforecast-forecast-1"}


def stamp(kind: str) -> dict:
    from . import __version__

    return {"format": FORMATS[kind], "package_version": __version__}


def check(payload: dict, kind: str, path: str | Path) -> dict:
    """`payload` without its format stamp, after checking this version can read it."""
    payload = dict(payload)
    found = payload.pop("format", None)
    if found is not None and found != FORMATS[kind]:
        prefix = FORMATS[kind].rsplit("-", 1)[0] + "-"
        if found.startswith(prefix):
            raise ValueError(f"{path} was saved in format {found!r}, newer than this version of "
                             f"areaFluForecast reads ({FORMATS[kind]!r}). Upgrade the package.")
        raise ValueError(f"{path} is a {found!r} file, not a saved {kind}.")
    return payload
