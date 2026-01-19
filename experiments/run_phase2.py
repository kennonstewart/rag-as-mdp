from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import dspy
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
    from policies.dspy_cot import DSPyCoTPolicy
    from policies.dspy_react import DSPyReActPolicy
except Exception:
    raise


def configure_dspy(provider: str, model: str, temperature: float, max_tokens: int) -> None:
    api_key = os.environ.get("DSPY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set DSPY_API_KEY or OPENAI_API_KEY for DSPy policies."
        )

    model_name = f"{provider}/{model}"
    lm = dspy.LM(
        model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)


def episode_summary(base_env: RAGEnvironment, info: dict, total_cost: float) -> dict:
    episode = base_env.current_episode
    stopped_with_answer = base_env.world.has_answer_evidence(episode, episode.G_t)
    regret_steps = int(info["episode_len"]) - int(info["optimal_stopping_time"])

    return {
        "total_reward": float(sum(episode.reward_history)),
        "episode_len": int(info["episode_len"]),
        "final_entropy": float(info["true_entropy"]),
        "optimal_stopping_time": int(info["optimal_stopping_time"]),
        "terminated_reason": info["terminated_reason"],
        "total_cost": float(total_cost),
        "stopped_with_answer_evidence": bool(stopped_with_answer),
        "regret_steps": int(regret_steps),
    }


def run_policy(env, policy, episodes: int, trajectories: bool):
    results = []
    traces = []
    base_env = env.unwrapped

    for _ in range(episodes):
        obs, info = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        episode_trace = []
        total_cost = 0.0

        while not done:
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = int(policy.select_action(obs))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_cost += float(info["action_cost"])

            if trajectories:
                episode = base_env.current_episode
                episode_trace.append(
                    {
                        "step": int(info["episode_len"]),
                        "entropy": float(info["true_entropy"]),
                        "action": int(action),
                        "reward": float(reward),
                        "action_cost": float(info["action_cost"]),
                        "compression": float(
                            base_env.world.compression_ratio(episode, episode.G_t)
                        ),
                    }
                )

        results.append(episode_summary(base_env, info, total_cost))
        if trajectories:
            traces.append(episode_trace)

    return results, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["fixed", "sb3_dqn", "dspy_cot", "dspy_react"],
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--k", type=int, default=2, help="Fixed horizon steps.")
    parser.add_argument("--timesteps", type=int, default=5000, help="SB3 training steps.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="outputs/phase2")
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument("--skip-check-env", action="store_true")
    parser.add_argument("--lm-provider", type=str, default="openai")
    parser.add_argument("--lm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
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

    policy = None
    if args.policy == "fixed":
        config["k"] = args.k
        policy = FixedHorizonPolicy(k=args.k)
    elif args.policy == "sb3_dqn":
        config["timesteps"] = args.timesteps
        config["dqn_kwargs"] = DQN_KWARGS
        policy = train_dqn(env, total_timesteps=args.timesteps, seed=args.seed, verbose=0)
    elif args.policy == "dspy_cot":
        configure_dspy(args.lm_provider, args.lm_model, args.temperature, args.max_tokens)
        config.update(
            {
                "lm_provider": args.lm_provider,
                "lm_model": args.lm_model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
        )
        policy = DSPyCoTPolicy(world)
    elif args.policy == "dspy_react":
        configure_dspy(args.lm_provider, args.lm_model, args.temperature, args.max_tokens)
        config.update(
            {
                "lm_provider": args.lm_provider,
                "lm_model": args.lm_model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
        )
        policy = DSPyReActPolicy(world)

    config_path = outdir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    results, traces = run_policy(
        env, policy, episodes=args.episodes, trajectories=args.trajectories
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
