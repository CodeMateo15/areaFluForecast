"""Command line: areaflu check | backtest | forecast.

    areaflu check    --rates rates.csv --areas areas.geojson --id-col name
    areaflu backtest --rates rates.csv --areas areas.geojson --id-col name --weather
    areaflu forecast --rates rates.csv --areas areas.geojson --id-col name --weather
    areaflu backtest --sample boston --seasons 1 --quick       # try it on bundled data

The city can also come from a JSON file of `City.from_files` arguments:
    areaflu forecast --city city.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

from .configs import DEFAULT_MODELS

QUICK = {"n_seeds": 1, "epochs": 5, "mc_samples": 4}


def _city(args):
    from .city import City
    from .datasets import load_sample

    if args.sample:
        city = load_sample(args.sample)
    elif args.city:
        with open(args.city) as handle:
            spec = json.load(handle)
        city = City.from_files(**spec)
    else:
        if not args.rates:
            sys.exit("Give the data: --rates (with --areas or --edges), --city, or --sample.")
        city = City.from_files(args.rates, areas=args.areas, edges=args.edges, id_col=args.id_col,
                               name=args.name, flu_months=_months(args.flu_months),
                               anchors=args.anchors or (), static=args.static,
                               weather=args.weather_csv)
    if getattr(args, "weather", False) and not city.has_weather:
        city.fetch_weather()
    return city


def _months(text: str):
    return tuple(int(m) for m in text.split(","))


def _configs(args) -> dict:
    overrides = {}
    if args.quick:
        for model in args.models:
            overrides[model] = {k: v for k, v in QUICK.items()
                                if not (k == "mc_samples" and not model.startswith("gnn_st"))}
    if args.seeds is not None:
        for model in args.models:
            overrides.setdefault(model, {})["n_seeds"] = args.seeds
    return overrides


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="areaflu", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "backtest", "forecast"):
        p = sub.add_parser(name)
        p.add_argument("--rates")
        p.add_argument("--areas", help="GeoJSON boundary file")
        p.add_argument("--edges", help="CSV edge list (alternative to --areas)")
        p.add_argument("--id-col", help="GeoJSON property holding the area id")
        p.add_argument("--name")
        p.add_argument("--flu-months", default="10,11,12,1,2,3")
        p.add_argument("--anchors", nargs="*")
        p.add_argument("--static")
        p.add_argument("--weather-csv")
        p.add_argument("--city", help="JSON file of City.from_files arguments")
        p.add_argument("--sample", choices=["boston", "chicago"])
        if name == "check":
            continue
        p.add_argument("--weather", action="store_true", help="fetch weather from Open-Meteo")
        p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
        p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4])
        p.add_argument("--seeds", type=int, help="ensemble size per model (default 10)")
        p.add_argument("--tune", action="store_true", help="choose settings from 12 candidates")
        p.add_argument("--quick", action="store_true", help="tiny models, to try the pipeline")
        p.add_argument("--out", default="areaflu_runs")
        if name == "backtest":
            p.add_argument("--seasons", nargs="+", type=int, default=[2],
                           help="one number n = the last n complete seasons; or season years")
    args = parser.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        city = _city(args)
        if args.command == "check":
            print(city.summary())
            return
        if args.command == "backtest":
            from .backtest import backtest
            seasons = args.seasons[0] if len(args.seasons) == 1 and args.seasons[0] < 100 else args.seasons
            backtest(city, args.models, seasons=seasons, horizons=args.horizons,
                     configs=_configs(args), tune=args.tune, out_dir=args.out)
        else:
            from .forecast import forecast
            result = forecast(city, args.models, horizons=args.horizons, configs=_configs(args),
                              tune=args.tune, out_dir=args.out)
            print(result.summary().to_string(index=False))
            if result.warnings:
                print("\nTrust check:\n" + "\n".join(f"  - {w}" for w in result.warnings))


if __name__ == "__main__":
    main()
