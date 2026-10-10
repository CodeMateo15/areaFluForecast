"""Progress lines.

Long runs print a fixed set of status lines so a log can be turned into a
percentage by anything that tails it:

    Budget: <N> epochs x 1 seeds      # total units of work, once at start
    --- seed 1 (1/1) ---
    Epoch <i> | <what just finished>  # after each unit (one model seed, or one baseline)
    Outputs: <path>                   # completion marker

The wording ("epochs", "seeds") matches the research repo's run_progress.py,
which reads exactly these lines; a unit here is one trained ensemble member.
"""

from __future__ import annotations

import sys
import time


class Progress:
    def __init__(self, total: int, *, enabled: bool = True, label: str = "") -> None:
        self.total = max(1, int(total))
        self.done = 0
        self.enabled = enabled
        self.label = label
        self.started = time.time()

    def start(self) -> "Progress":
        if self.enabled:
            print(f"Budget: {self.total} epochs x 1 seeds", flush=True)
            print("--- seed 1 (1/1) ---", flush=True)
        return self

    def step(self, message: str) -> None:
        self.done += 1
        if not self.enabled:
            return
        percent = 100.0 * self.done / self.total
        elapsed = time.time() - self.started
        eta = elapsed / self.done * (self.total - self.done)
        print(f"Epoch {self.done} | {percent:5.1f}% | {message} | "
              f"elapsed {_clock(elapsed)} | left ~{_clock(eta)}", flush=True)

    def finish(self, path) -> None:
        if self.enabled:
            print(f"Outputs: {path}", flush=True)
            sys.stdout.flush()


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"
