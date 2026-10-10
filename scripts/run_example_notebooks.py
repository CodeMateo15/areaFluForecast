"""Execute the example notebooks in order, saving outputs into the .ipynb files.

    python scripts/run_example_notebooks.py               # all four
    python scripts/run_example_notebooks.py 03            # just one
    python scripts/run_example_notebooks.py --quick       # 2-epoch models, nothing saved

Runs in examples/ so the notebooks' relative paths resolve. Prints the Budget /
seed / Epoch / Outputs progress lines (one unit per executed cell).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class Reporting(NotebookClient):
    """A client that reports each finished cell as a progress unit."""

    def __init__(self, *args, progress, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress = progress

    def output(self, outs, msg, display_id, cell_index):
        """Forward the package's own progress lines from inside a long cell.

        A backtest is one cell but most of the run time, so the cell count alone
        sits still for an hour. Its inner "Epoch k | p% | ..." lines are echoed
        with a "  inside:" prefix, which run_progress.py does not count, so the
        cell-level percentage stays correct while the log keeps moving.
        """
        if msg.get("msg_type") == "stream":
            for line in msg["content"].get("text", "").splitlines():
                if line.startswith(("Epoch ", "Budget:")):
                    print(f"  inside {self._name} cell {cell_index}: {line}", flush=True)
        return super().output(outs, msg, display_id, cell_index)

    async def async_execute_cell(self, cell, cell_index, *args, **kwargs):
        result = await super().async_execute_cell(cell, cell_index, *args, **kwargs)
        if cell.cell_type == "code":
            self._progress(f"{self._name} cell {cell_index}")
        return result


def main() -> None:
    quick = "--quick" in sys.argv
    wanted = [a for a in sys.argv[1:] if a != "--quick"]
    books = sorted(p for p in EXAMPLES.glob("[0-9][0-9]_*.ipynb")
                   if not wanted or any(p.name.startswith(w) for w in wanted))
    loaded = [(p, nbformat.read(p, as_version=4)) for p in books]
    total = sum(sum(c.cell_type == "code" for c in nb.cells) + quick for _, nb in loaded)
    print(f"Budget: {total} epochs x 1 seeds", flush=True)
    print("--- seed 1 (1/1) ---", flush=True)
    done, start = 0, time.time()

    def step(message: str) -> None:
        nonlocal done
        done += 1
        print(f"Epoch {done} | {100 * done / total:5.1f}% | {message} | "
              f"{(time.time() - start) / 60:.1f} min", flush=True)

    for path, nb in loaded:
        if quick:   # a hidden first cell shrinks every model; outputs are not saved
            nb.cells.insert(0, nbformat.v4.new_code_cell(
                "import dataclasses, areafluforecast.configs as _c\n"
                "for _k, _v in list(_c.PRESETS.items()):\n"
                "    if hasattr(_v, 'epochs'):\n"
                "        _c.PRESETS[_k] = dataclasses.replace(_v, epochs=2, **({'mc_samples': 4} if hasattr(_v, 'mc_samples') else {}))"))
        client = Reporting(nb, timeout=None, kernel_name="python3",
                           resources={"metadata": {"path": str(EXAMPLES)}}, progress=step)
        client._name = path.stem
        client.execute()
        if not quick:
            nbformat.write(nb, path)
            print(f"saved {path.name}", flush=True)
    print(f"Outputs: {EXAMPLES}", flush=True)


if __name__ == "__main__":
    main()
