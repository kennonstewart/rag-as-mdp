from __future__ import annotations

from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

DQN_KWARGS = {
    "learning_rate": 0.03,
    "buffer_size": 10000,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.1,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
}


def train_dqn(
    env,
    total_timesteps: int = 5_000,
    seed: Optional[int] = None,
    verbose: int = 0,
):
    """Train a small DQN agent on the RAG environment."""
    vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed)
    model = DQN(
        "MultiInputPolicy",
        vec_env,
        verbose=verbose,
        seed=seed,
        **DQN_KWARGS,
    )
    model.learn(total_timesteps=total_timesteps)
    return model
