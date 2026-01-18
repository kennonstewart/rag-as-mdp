from __future__ import annotations

from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env


def train_dqn(
    env,
    total_timesteps: int = 5_000,
    seed: Optional[int] = None,
    verbose: int = 0,
):
    """Train a small DQN agent on the RAG environment."""
    vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed)
    model = DQN("MultiInputPolicy", vec_env, verbose=verbose, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    return model
