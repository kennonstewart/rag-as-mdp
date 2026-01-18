from __future__ import annotations

from stable_baselines3.common.env_checker import check_env

from rag_mdp.env import RAGEnvironment
from rag_mdp.synthetic_world import SyntheticWorld


def test_env_checker_passes():
    world = SyntheticWorld(seed=123)
    env = RAGEnvironment(world)
    check_env(env, warn=True)


def test_reset_determinism():
    world = SyntheticWorld(seed=123)
    env = RAGEnvironment(world)

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)

    assert obs1["query_id"] == obs2["query_id"]


def test_observation_shapes():
    world = SyntheticWorld(seed=123)
    env = RAGEnvironment(world, max_nodes=8, max_edges=4)
    obs, _ = env.reset(seed=7)

    assert obs["nodes"].shape == (8,)
    assert obs["edges"].shape == (4, 2)
    assert obs["entropy"].shape == (1,)
    assert obs["step"].shape == (1,)
