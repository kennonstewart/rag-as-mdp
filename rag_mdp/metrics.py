from __future__ import annotations

import math
from typing import Iterable


def entropy_from_feasible_count(count: int) -> float:
    """Compute entropy of a uniform distribution over `count` outcomes."""
    if count <= 1:
        return 0.0
    return math.log(count)


def entropy_from_probs(probabilities: Iterable[float]) -> float:
    """Shannon entropy of a discrete distribution."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def reward_from_entropy_change(
    prev_entropy: float,
    next_entropy: float,
    action_cost: float,
) -> float:
    """Entropy reduction minus action cost."""
    return (prev_entropy - next_entropy) - action_cost
