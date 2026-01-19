from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List

from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.env_checker import check_env


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from rag_mdp.env import RAGEnvironment
    from rag_mdp.synthetic_world import SyntheticWorld
    from policies.fixed_horizon import FixedHorizonPolicy
    from policies.sb3_dqn import DQN_KWARGS, train_dqn
except Exception:
    raise


def run_fixed_policy(env: RAGEnvironment, episodes: int, k: int, trajectories: bool):
    policy = FixedHorizonPolicy(k=k)
    results = []
    traces = []

    for _ in range(episodes):
        obs, info = env.reset()
        policy.reset()
        done = False
        total_reward = 0.0
        episode_trace = []

        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if trajectories:
                episode_trace.append(
                    {
                        "step": int(info["episode_len"]),
                        "entropy": float(info["true_entropy"]),
                        "action": int(action),
                        "reward": float(reward),
                        "action_cost": float(info["action_cost"]),
                    }
                )

        results.append(
            {
                "total_reward": total_reward,
                "episode_len": int(info["episode_len"]),
                "final_entropy": float(info["true_entropy"]),
                "optimal_stopping_time": int(info["optimal_stopping_time"]),
                "terminated_reason": info["terminated_reason"],
            }
        )
        if trajectories:
            traces.append(episode_trace)

    return results, traces


def run_sb3_policy(
    env: RAGEnvironment,
    episodes: int,
    timesteps: int,
    seed: int | None,
    trajectories: bool,
):
    model = train_dqn(env, total_timesteps=timesteps, seed=seed, verbose=0)
    results = []
    traces = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        episode_trace = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward

            if trajectories:
                episode_trace.append(
                    {
                        "step": int(info["episode_len"]),
                        "entropy": float(info["true_entropy"]),
                        "action": int(action),
                        "reward": float(reward),
                        "action_cost": float(info["action_cost"]),
                    }
                )

        results.append(
            {
                "total_reward": total_reward,
                "episode_len": int(info["episode_len"]),
                "final_entropy": float(info["true_entropy"]),
                "optimal_stopping_time": int(info["optimal_stopping_time"]),
                "terminated_reason": info["terminated_reason"],
            }
        )
        if trajectories:
            traces.append(episode_trace)

    return results, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["fixed", "sb3_dqn"], required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--k", type=int, default=2, help="Fixed horizon steps.")
    parser.add_argument("--timesteps", type=int, default=5000, help="SB3 training steps.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="outputs/phase1")
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument("--skip-check-env", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    world = SyntheticWorld(seed=args.seed)
    base_env = RAGEnvironment(world, max_steps=args.max_steps)
    if not args.skip_check_env:
        check_env(base_env, warn=True)
    env = RecordEpisodeStatistics(base_env)

    config = {
        "policy": args.policy,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "trajectories": bool(args.trajectories),
        "action_costs": base_env.action_costs,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split()[0],
    }
    if args.policy == "fixed":
        config["k"] = args.k
    else:
        config["timesteps"] = args.timesteps
        config["dqn_kwargs"] = DQN_KWARGS

    config_path = outdir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    if args.policy == "fixed":
        results, traces = run_fixed_policy(
            env,
            episodes=args.episodes,
            k=args.k,
            trajectories=args.trajectories,
        )
    else:
        results, traces = run_sb3_policy(
            env,
            episodes=args.episodes,
            timesteps=args.timesteps,
            seed=args.seed,
            trajectories=args.trajectories,
        )

    results_path = outdir / "results.jsonl"
    with results_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    if args.trajectories:
        trajectories_path = outdir / "trajectories.jsonl"
        with trajectories_path.open("w") as f:
            for episode_trace in traces:
                f.write(json.dumps(episode_trace) + "\n")

    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
