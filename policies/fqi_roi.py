from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from rag_mdp.synthetic_world import SyntheticWorld


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


def observation_features(world: SyntheticWorld, obs: dict) -> np.ndarray:
    """Simple feature vector for fitted Q-iteration."""
    query_id = int(obs["query_id"])
    location = world.locations[query_id]
    animal = world.spec.location_to_animal[location]

    edges = obs["edges"].tolist()
    edge_pairs = []
    for src_id, dst_id in edges:
        if src_id == 0 or dst_id == 0:
            continue
        src = world.id_to_entity.get(int(src_id))
        dst = world.id_to_entity.get(int(dst_id))
        if src is None or dst is None:
            continue
        edge_pairs.append((src, dst))

    has_location_animal = int((location, animal) in edge_pairs)
    has_any_color_edge = int(
        any(src == animal and dst in world.colors for src, dst in edge_pairs)
    )

    entropy = float(obs["entropy"][0])
    step = float(obs["step"][0])

    return np.array([entropy, step, has_location_animal, has_any_color_edge], dtype=float)


@dataclass
class FittedQPolicy:
    world: SyntheticWorld
    actions: Sequence[int]
    gamma: float = 0.9
    ridge: float = 1e-3
    n_iters: int = 20
    weights: Dict[int, np.ndarray] | None = None

    def fit(self, transitions: Iterable[Transition]) -> None:
        transitions = list(transitions)
        if not transitions:
            raise ValueError("No transitions provided for FQI training.")

        feature_dim = transitions[0][0].shape[0]
        weights = {action: np.zeros(feature_dim) for action in self.actions}

        for _ in range(self.n_iters):
            targets: Dict[int, List[float]] = {action: [] for action in self.actions}
            features: Dict[int, List[np.ndarray]] = {action: [] for action in self.actions}

            for phi, action, reward, next_phi, done in transitions:
                if done:
                    target = reward
                else:
                    next_values = [
                        float(weights[a].dot(next_phi)) for a in self.actions
                    ]
                    target = reward + self.gamma * max(next_values)

                features[action].append(phi)
                targets[action].append(target)

            for action in self.actions:
                if not features[action]:
                    continue
                X = np.vstack(features[action])
                y = np.array(targets[action])
                XtX = X.T @ X + self.ridge * np.eye(feature_dim)
                Xty = X.T @ y
                weights[action] = np.linalg.solve(XtX, Xty)

        self.weights = weights

    def select_action(self, obs: dict) -> int:
        if self.weights is None:
            raise RuntimeError("FittedQPolicy must be trained before use.")

        phi = observation_features(self.world, obs)
        q_values = {action: float(self.weights[action].dot(phi)) for action in self.actions}
        return max(q_values, key=q_values.get)
