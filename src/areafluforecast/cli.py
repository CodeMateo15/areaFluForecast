"""Command line: areaflu check | backtest | forecast.

    areaflu check    --rates rates.csv --areas areas.geojson --id-col name
    areaflu backtest --rates rates.csv --areas areas.geojson --id-col name --weather
    areaflu forecast --rates rates.csv --areas areas.geojson --id-col name --weather
    areaflu backtest --sample boston --seasons 1 --quick       # try it on bundled data

The city can also come from a JSON file of `City.from_files` arguments (paths in
it are relative to the JSON file):
    areaflu forecast --city city.json

Start from the bundled Boston models (transfer to a city with a short history):
    areaflu backtest --city city.json --weather --pretrained boston --zero-shot
    areaflu forecast --city city.json --weather --pretrained boston --zero-shot

Exit codes: 0 success, 2 bad command-line arguments or settings, 3 a problem
with the input data or files (the message says what to fix).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from .configs import DEFAULT_MODELS

QUICK = {"n_seeds": 1, "epochs": 5, "mc_samples": 4}
EXIT_USAGE, EXIT_DATA = 2, 3
PATH_KEYS = ("rates", "areas", "edges", "weather", "static", "centroids")


class UsageError(Exception):
    """A command-line argument is wrong."""


def _city(args):
    from .city import City
    from .datasets import load_sample

    if args.sample:
        city = load_sample(args.sample)
    elif args.city:
        spec_path = Path(args.city)
        spec = json.loads(spec_path.read_text())
        for key in PATH_KEYS:
            if spec.get(key) is not None and not Path(spec[key]).is_absolute():
                spec[key] = str(spec_path.parent / spec[key])
        city = City.from_files(**spec)
    else:
        if not args.rates:
            raise UsageError("give the data: --rates (with --areas or --edges), --city, or --sample.")
        city = City.from_files(args.rates, areas=args.areas, edges=args.edges, id_col=args.id_col,
                               name=args.name, flu_months=_months(args.flu_months),
                               week_col=args.week_col, area_col=args.area_col,
                               rate_col=args.rate_col, anchors=args.anchors or (),
                               static=args.static, weather=args.weather_csv,
                               centroids=args.centroids)
    if getattr(args, "weather", False) and not city.has_weather:
        city.fetch_weather()
    return city


def _months(text: str):
    try:
        return tuple(int(m) for m in text.split(","))
    except ValueError:
        raise UsageError(f"--flu-months takes comma-separated month numbers, got {text!r}.") from None


def _configs(args) -> dict:
    overrides = {}
    learned = [m for m in args.models if m not in ("persistence", "seasonal_naive")]
    if args.pretrained or args.init_from:
        learned.append("pretrained")
    if args.quick:
        for model in learned:
            overrides[model] = {k: v for k, v in QUICK.items()
                                if not (k == "mc_samples" and model.startswith(("lstm", "colagnn")))}
    if args.seeds is not None:
        for model in learned:
            overrides.setdefault(model, {})["n_seeds"] = args.seeds
    return overrides


def _init_from(args):
    """--pretrained NAME, or --init-from PATH / --init-from H=PATH (repeatable)."""
    from .checkpoint import pretrained

    if args.pretrained and args.init_from:
        raise UsageError("use --pretrained or --init-from, not both.")
    if args.pretrained:
        return pretrained(args.pretrained, horizons=args.horizons)
    if not args.init_from:
        return None
    if len(args.init_from) == 1 and "=" not in args.init_from[0]:
        return args.init_from[0]
    mapping = {}
    for item in args.init_from:
        horizon, sep, path = item.partition("=")
        if not sep or not horizon.isdigit():
            raise UsageError(f"--init-from takes PATH, or H=PATH per horizon; got {item!r}.")
        mapping[int(horizon)] = path
    return mapping


def _parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(prog="areaflu", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"areaflu {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    helps = {"check": "read the data and summarise it",
             "backtest": "would the models have worked here? score past seasons",
             "forecast": "forecast the weeks after the last observation, with a trust check"}
    for name in ("check", "backtest", "forecast"):
        p = sub.add_parser(name, help=helps[name], description=helps[name].capitalize() + ".")
        data = p.add_argument_group("data")
        data.add_argument("--rates", help="CSV of weekly rates, long (week, area, rate) or wide")
        data.add_argument("--areas", help="GeoJSON boundary file; neighbours are derived from it")
        data.add_argument("--edges", help="CSV edge list (alternative to --areas)")
        data.add_argument("--id-col", help="GeoJSON property holding the area id")
        data.add_argument("--week-col", default="week", help="week column in a long rates CSV")
        data.add_argument("--area-col", default="area", help="area column in the CSVs")
        data.add_argument("--rate-col", default="rate", help="rate column in a long rates CSV")
        data.add_argument("--name", help="city name (default: the rates file's name)")
        data.add_argument("--flu-months", default="10,11,12,1,2,3",
                          help="months of flu season, comma-separated (default 10,11,12,1,2,3)")
        data.add_argument("--anchors", nargs="*", metavar="NAME",
                          help="boundary areas without data that border yours")
        data.add_argument("--static", help="CSV of numeric area characteristics (area column)")
        data.add_argument("--weather-csv", help="long CSV of weekly weather per area")
        data.add_argument("--centroids", help="CSV of area, lat, lon (to fetch weather "
                                              "without a boundary file)")
        data.add_argument("--city", help="JSON file of City.from_files arguments")
        data.add_argument("--sample", choices=["boston", "chicago"], help="use a bundled sample")
        if name == "check":
            continue
        data.add_argument("--weather", action="store_true", help="fetch weather from Open-Meteo")
        run = p.add_argument_group("models")
        run.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                         help="models to run (default: gnn_st lstm colagnn)")
        run.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4],
                         help="weeks ahead (default: 1 2 3 4)")
        run.add_argument("--seeds", type=int, help="ensemble size per model (default 10)")
        run.add_argument("--seed", type=int, help="derive every random seed from this number")
        run.add_argument("--tune", action="store_true",
                         help="choose each model's settings from 12 candidates (12x slower)")
        run.add_argument("--quick", action="store_true",
                         help="tiny models, to try the pipeline (results mean nothing)")
        run.add_argument("--pretrained", choices=["boston"],
                         help="also run a bundled pretrained model, as 'pretrained'")
        run.add_argument("--init-from", nargs="+", metavar="CHECKPOINT",
                         help="also run a saved model, as 'pretrained': PATH for one horizon, "
                              "or H=PATH for each")
        run.add_argument("--zero-shot", action="store_true",
                         help="apply the pretrained model without training it here")
        run.add_argument("--device", help="cpu (default) or e.g. cuda")
        out = p.add_argument_group("output")
        out.add_argument("--out", default="areaflu_runs",
                         help="folder for the saved run (default: areaflu_runs)")
        out.add_argument("--no-progress", action="store_true", help="no progress bar")
        if name == "backtest":
            seasons = run.add_mutually_exclusive_group()
            seasons.add_argument("--seasons", type=int, default=2,
                                 help="backtest the last N complete seasons (default 2)")
            seasons.add_argument("--season-years", nargs="+", type=int, metavar="YEAR",
                                 help="backtest these seasons (2023 = the season scored "
                                      "June 2023 - May 2024 for an Oct-Mar city)")
        else:
            out.add_argument("--save-models", action="store_true",
                             help="save each trained model in the run folder")
            out.add_argument("--backtest", metavar="FOLDER",
                             help="the backtest folder to check trust against (default: the "
                                  "newest matching one in --out); 'none' skips the check")
    return parser


def _run(args) -> None:
    city = _city(args)
    if args.command == "check":
        print(city.summary())
        return
    common = dict(horizons=args.horizons, configs=_configs(args), tune=args.tune,
                  out_dir=args.out, init_from=_init_from(args), seed=args.seed,
                  progress=not args.no_progress, device=args.device)
    if args.command == "backtest":
        from .backtest import backtest
        seasons = args.season_years or args.seasons
        backtest(city, args.models, seasons=seasons, zero_shot=args.zero_shot, **common)
        return
    from .forecast import forecast
    chosen = args.backtest or "latest"
    if chosen.lower() == "none":
        chosen = None
    # The trust check is printed below; don't repeat each note as a warning.
    from .errors import TrustWarning
    warnings.simplefilter("ignore", TrustWarning)
    result = forecast(city, args.models, zero_shot=args.zero_shot, backtest=chosen,
                      save_models=args.save_models, **common)
    print(result.summary().to_string(index=False))
    if result.warnings:
        print("\nTrust check:\n" + "\n".join(f"  - {w}" for w in result.warnings))


def main(argv=None) -> int:
    from .errors import DataError

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        with warnings.catch_warnings():
            _run(args)
    except UsageError as error:
        parser.print_usage(sys.stderr)
        print(f"areaflu: error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except (DataError, FileNotFoundError, IsADirectoryError, json.JSONDecodeError) as error:
        print(f"areaflu: error: {error}", file=sys.stderr)
        return EXIT_DATA
    except ValueError as error:
        print(f"areaflu: error: {error}", file=sys.stderr)
        return EXIT_USAGE
    return 0


if __name__ == "__main__":
    sys.exit(main())
