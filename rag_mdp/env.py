from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rag_mdp.metrics import reward_from_entropy_change
from rag_mdp.synthetic_world import SyntheticWorld


class RAGEnvironment(gym.Env):
    """Gymnasium environment for the Phase 1 toy RAG-as-MDP setting."""

    metadata = {"render_modes": ["human", "json"]}

    def __init__(
        self,
        synthetic_world: SyntheticWorld,
        max_steps: int = 5,
        max_nodes: int = 16,
        max_edges: int = 16,
        action_costs: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.world = synthetic_world
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.render_mode = render_mode

        self.action_costs = action_costs or {
            "retrieve": 1.0,
            "reflect": 0.2,
            "return": 0.0,
        }

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Dict(
            {
                "query_id": spaces.Discrete(len(self.world.locations)),
                "nodes": spaces.Box(
                    low=0,
                    high=len(self.world.entity_to_id),
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
                "edges": spaces.Box(
                    low=0,
                    high=len(self.world.entity_to_id),
                    shape=(self.max_edges, 2),
                    dtype=np.int32,
                ),
                "entropy": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "step": spaces.Box(
                    low=0,
                    high=self.max_steps,
                    shape=(1,),
                    dtype=np.int32,
                ),
            }
        )

        self.current_episode = None
        self.t = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_episode = self.world.generate_episode(seed=seed)
        self.t = 0

        initial_entropy = self.world.compute_true_entropy(
            self.current_episode, self.current_episode.G_t
        )
        self.current_episode.entropy_history = [initial_entropy]
        self.current_episode.reward_history = []

        return self._get_observation(), self._get_info(terminated_reason="reset")

    def step(self, action: int):
        if self.current_episode is None:
            raise RuntimeError("Environment must be reset before stepping.")

        self.t += 1
        action_cost = 0.0

        if action == 0:  # retrieve
            evidence = self.world.retrieve_evidence(
                self.current_episode, self.current_episode.G_t
            )
            if evidence is not None:
                self.world.add_evidence(self.current_episode.G_t, evidence)
            action_cost = self.action_costs["retrieve"]
        elif action == 1:  # reflect
            self.current_episode.G_t = self.world.consolidate_graph(
                self.current_episode.G_t
            )
            action_cost = self.action_costs["reflect"]
        elif action == 2:  # return
            action_cost = self.action_costs["return"]
        else:
            raise ValueError(f"Unknown action: {action}")

        prev_entropy = self.current_episode.entropy_history[-1]
        next_entropy = self.world.compute_true_entropy(
            self.current_episode, self.current_episode.G_t
        )
        self.current_episode.entropy_history.append(next_entropy)

        reward = reward_from_entropy_change(prev_entropy, next_entropy, action_cost)
        self.current_episode.reward_history.append(reward)

        terminated = action == 2
        truncated = self.t >= self.max_steps

        terminated_reason = "return" if terminated else ("time_limit" if truncated else "running")
        info = self._get_info(terminated_reason=terminated_reason, action_cost=action_cost)

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        nodes, edges = self.world.graph_to_arrays(
            self.current_episode.G_t, self.max_nodes, self.max_edges
        )
        entropy = np.array([self.current_episode.entropy_history[-1]], dtype=np.float32)
        step = np.array([self.t], dtype=np.int32)

        return {
            "query_id": int(self.current_episode.question_id),
            "nodes": nodes,
            "edges": edges,
            "entropy": entropy,
            "step": step,
        }

    def _get_info(
        self,
        terminated_reason: str,
        action_cost: float = 0.0,
    ) -> Dict[str, float]:
        optimal_tau = self.world.compute_optimal_stopping_time(
            self.current_episode, self.max_steps, self.action_costs
        )
        return {
            "true_entropy": float(self.current_episode.entropy_history[-1]),
            "episode_len": int(self.t),
            "action_cost": float(action_cost),
            "terminated_reason": terminated_reason,
            "optimal_stopping_time": int(optimal_tau),
        }
