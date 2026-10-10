"""Gate B, part 1: does the package reproduce the paper's tuned backtest?

Runs the package's graph model (paper configuration), LSTM and Cola-GNN with the
exact per-fold tuned settings from the research repo's Code/sweep/tuned_configs.json,
10 seeds, on Boston's three unseen seasons at 1, 2 and 4 weeks ahead: 27 tasks.

    python scripts/parity/run_parity.py list
    python scripts/parity/run_parity.py run --index 7 --out parity_out

On Explorer this is a job array; see scripts/explorer/areaflu.sbatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_bridge import boston_from_research, maybe_quick, research_repo  # noqa: E402

from areafluforecast import configs  # noqa: E402
from areafluforecast.engine import run_model  # noqa: E402
from areafluforecast.progress import Progress  # noqa: E402

SEASONS = ("2022_23", "2023_24", "2024_25")
HORIZONS = (1, 2, 4)
# research arm -> (tuned_configs.json key, package preset)
ARMS = {"gnn_st_tuned": ("gnn_st", "gnn_st_paper"),
        "lstm_tuned": ("lstm", "lstm"),
        "colagnn_tuned": ("colagnn", "colagnn")}


def tasks() -> list[tuple[str, int, str]]:
    return [(s, h, a) for s in SEASONS for h in HORIZONS for a in ARMS]


def run(index: int, out: Path) -> Path:
    season, horizon, arm = tasks()[index]
    repo = research_repo()
    tuned = json.loads((repo / "Code" / "sweep" / "tuned_configs.json").read_text())
    key, preset = ARMS[arm]
    flags = tuned["boston"][season][str(horizon)][key]["flags"]
    config = maybe_quick(replace(configs.with_flags(configs.PRESETS[preset], flags), n_seeds=10))
    year = int(season[:4])
    city = boston_from_research()
    start, end = city.season_window(year)

    target = out / f"backtest_{season}" / f"horizon_{horizon:02d}" / arm
    target.mkdir(parents=True, exist_ok=True)
    print(f"{arm} | {season} | h={horizon} | flags {' '.join(flags)}", flush=True)
    bar = Progress(config.n_seeds).start()
    output = run_model(city, config, horizon=horizon, test_start=start, test_end=end,
                       on_unit=bar.step)
    output.test.rename(columns={"area": "neighborhood"}).to_csv(target / "predictions.csv",
                                                                index=False)
    (target / "run_config.json").write_text(json.dumps(
        {"arm": arm, "season": season, "horizon": horizon, "flags": flags,
         "config": configs.to_json(config), "meta": output.meta}, indent=2, default=str))
    bar.finish(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--out", type=Path, default=Path("parity_out"))
    args = parser.parse_args()
    if args.command == "list":
        for i, t in enumerate(tasks()):
            print(i, *t)
        print(f"{len(tasks())} tasks")
    else:
        run(args.index, args.out)


if __name__ == "__main__":
    main()
