from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import dspy

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from rag_mdp.env import RAGEnvironment
    from rag_mdp.synthetic_world import SyntheticWorld
    from rag_mdp.observation_text import (
        evidence_summary_from_graph_arrays,
        query_text_from_id,
    )
except Exception:
    raise


class StoppingSignature(dspy.Signature):
    query_text: str = dspy.InputField()
    evidence_summary: str = dspy.InputField()
    entropy: float = dspy.InputField()
    step: int = dspy.InputField()
    confidence_threshold: float = dspy.InputField()

    action: str = dspy.OutputField(desc="One of: retrieve, return.")


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


def build_training_set(
    env: RAGEnvironment, episodes: int, confidence_threshold: float
) -> list[dspy.Example]:
    examples = []
    base_env = env

    for _ in range(episodes):
        obs, info = base_env.reset()
        episode = base_env.current_episode
        optimal_tau = base_env.world.compute_optimal_stopping_time(
            episode, base_env.max_steps, base_env.action_costs
        )

        done = False
        while not done:
            query_text = query_text_from_id(base_env.world, obs["query_id"])
            evidence_summary = evidence_summary_from_graph_arrays(
                base_env.world, obs["nodes"], obs["edges"]
            )
            entropy = float(obs["entropy"][0])
            step = int(obs["step"][0])

            label_action = "return" if step >= optimal_tau else "retrieve"

            examples.append(
                dspy.Example(
                    query_text=query_text,
                    evidence_summary=evidence_summary,
                    entropy=entropy,
                    step=step,
                    confidence_threshold=confidence_threshold,
                    action=label_action,
                ).with_inputs(
                    "query_text",
                    "evidence_summary",
                    "entropy",
                    "step",
                    "confidence_threshold",
                )
            )

            obs, _, terminated, truncated, _ = base_env.step(0)
            done = terminated or truncated

    return examples


def action_accuracy(example, prediction, trace=None) -> float:
    return 1.0 if str(prediction.action).strip().lower() == example.action else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="outputs/phase3_mipro")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--lm-provider", type=str, default="openai")
    parser.add_argument("--lm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--save-program", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    configure_dspy(args.lm_provider, args.lm_model, args.temperature, args.max_tokens)

    world = SyntheticWorld(seed=args.seed)
    env = RAGEnvironment(world, max_steps=args.max_steps)

    trainset = build_training_set(env, args.episodes, args.confidence_threshold)

    teleprompter = dspy.MIPROv2(metric=action_accuracy, auto="light")
    program = dspy.ChainOfThought(StoppingSignature)
    compiled = teleprompter.compile(program, trainset=trainset)

    compiled_path = outdir / "compiled_policy.json"
    compiled.save(str(compiled_path), save_program=bool(args.save_program))

    config = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "confidence_threshold": args.confidence_threshold,
        "lm_provider": args.lm_provider,
        "lm_model": args.lm_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "save_program": bool(args.save_program),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with (outdir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote compiled policy to {compiled_path}")


if __name__ == "__main__":
    main()
