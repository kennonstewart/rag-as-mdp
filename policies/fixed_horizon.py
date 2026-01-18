from __future__ import annotations


class FixedHorizonPolicy:
    """Retrieve for k steps, then return."""

    def __init__(self, k: int = 2) -> None:
        self.k = k
        self.steps = 0

    def reset(self) -> None:
        self.steps = 0

    def select_action(self, obs) -> int:
        self.steps += 1
        if self.steps > self.k:
            return 2  # return
        return 0  # retrieve
