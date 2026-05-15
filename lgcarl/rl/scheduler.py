from __future__ import annotations


class LinearEpsilonScheduler:
    def __init__(self, start: float, end: float, decay_steps: int) -> None:
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(1, int(decay_steps))

    def value(self, step: int) -> float:
        fraction = min(1.0, max(0.0, step / self.decay_steps))
        return self.start + fraction * (self.end - self.start)

