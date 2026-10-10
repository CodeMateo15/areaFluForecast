"""Gate B, part 2: pretrained Boston weights, and do they transfer?

1. `train --horizon h`: train the transferable graph model (`gnn_st_noadapt`:
   no learned adjacency, so it fits any number of areas; weather on; no static
   characteristics) on all of Boston, 10 seeds, and save
   models/boston_gnn_st_noadapt_h{h}.pt.
2. `transfer --horizon h`: apply that checkpoint to Columbus's unseen 2024-25
   season with no training on Columbus (zero-shot), and compare with persistence
   and with a model trained on Columbus itself. The paper found Boston -> Columbus
   zero-shot tied a locally trained model and beat persistence.
3. `report`: one table across horizons.

    python scripts/pretrained/train_pretrained.py train --horizon 1 --out pretrained_out
    python scripts/pretrained/train_pretrained.py transfer --horizon 1 --out pretrained_out
    python scripts/pretrained/train_pretrained.py report --out pretrained_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_bridge import boston_from_research, city_from_research, maybe_quick  # noqa: E402

import areafluforecast as af  # noqa: E402
from areafluforecast.checkpoint import save_checkpoint  # noqa: E402
from areafluforecast.engine import run_model  # noqa: E402
from areafluforecast.progress import Progress  # noqa: E402
from areafluforecast.scoring import add_cell_scores  # noqa: E402

PRESET = "gnn_st_noadapt"
TRANSFER_SEASON = 2024


def checkpoint_path(out: Path, horizon: int) -> Path:
    return out / "models" / f"boston_{PRESET}_h{horizon}.pt"


def train(horizon: int, out: Path) -> None:
    city = boston_from_research(with_static=False)
    config = maybe_quick(af.resolve(PRESET))
    last = city.rates.index.max()
    future = pd.date_range(last + pd.Timedelta(days=7), periods=horizon, freq="7D")
    extended = pd.concat([city.rates, pd.DataFrame(float("nan"), index=future, columns=city.areas)])
    bar = Progress(config.n_seeds).start()
    output = run_model(city, config, horizon=horizon, test_start=future[0], test_end=future[-1],
                       rates=extended, only_origin=len(city.rates) - 1, on_unit=bar.step)
    path = save_checkpoint(checkpoint_path(out, horizon), output, city_name="boston",
                           areas=city.areas)
    bar.finish(path)


def transfer(horizon: int, out: Path) -> None:
    city = city_from_research("columbus", with_static=False)
    start, end = city.season_window(TRANSFER_SEASON)
    config, members, _ = af.load_checkpoint(checkpoint_path(out, horizon))
    arms = {
        "zero_shot_from_boston": dict(config=maybe_quick(config), init_members=members,
                                      zero_shot=True),
        "trained_on_columbus": dict(config=maybe_quick(af.resolve(PRESET))),
        "persistence": dict(config=af.resolve("persistence")),
    }
    bar = Progress(sum(getattr(a["config"], "n_seeds", 1) for a in arms.values())).start()
    frames = []
    for name, kwargs in arms.items():
        output = run_model(city, horizon=horizon, test_start=start, test_end=end,
                           on_unit=bar.step, **kwargs)
        frames.append(output.test.assign(model=name, season=city.season_label(TRANSFER_SEASON)))
    target = out / "transfer" / f"columbus_h{horizon}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(target, index=False)
    bar.finish(target)


def report(out: Path) -> None:
    from areafluforecast.scoring import compare_to_reference

    files = sorted((out / "transfer").glob("columbus_h*.csv"))
    scored = add_cell_scores(pd.concat([pd.read_csv(f, parse_dates=["origin_date", "target_date"])
                                        for f in files]))
    table = compare_to_reference(scored, "persistence")
    print(table.round(3).to_string(index=False))
    table.to_csv(out / "transfer" / "report.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "transfer", "report"])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("pretrained_out"))
    args = parser.parse_args()
    {"train": lambda: train(args.horizon, args.out),
     "transfer": lambda: transfer(args.horizon, args.out),
     "report": lambda: report(args.out)}[args.command]()


if __name__ == "__main__":
    main()
