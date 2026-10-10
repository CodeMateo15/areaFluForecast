"""Write a city from the research repo in the layout `load_city` reads.

    AREAFLU_RESEARCH_REPO=~/GNN-Influenza-Boston python scripts/export_sample.py
    python scripts/export_sample.py --out live/boston          # refreshed data, live track
    python scripts/export_sample.py --city chicago --out live/chicago

With no --out it rewrites the bundled sample (src/areafluforecast/sample_data/<city>),
which must stay identical to the data the paper used: refreshed data goes to --out.

Rates (BPHC public dashboard), borders, anchors, centroids and weather
(Open-Meteo). No static demographics: they are off by default, and the paper's
ablations found them inert.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_bridge import city_from_research  # noqa: E402

from areafluforecast import save_city  # noqa: E402

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "src" / "areafluforecast" / "sample_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", default="boston")
    parser.add_argument("--out", type=Path, default=None,
                        help="folder to write (default: the bundled sample)")
    args = parser.parse_args()
    city = city_from_research(args.city, with_static=False)
    out = save_city(city, args.out or SAMPLE_DATA / args.city)
    print(city.summary())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
