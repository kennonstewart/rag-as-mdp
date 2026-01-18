from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "episodes": int(len(df)),
        "mean_reward": float(df["total_reward"].mean()),
        "mean_length": float(df["episode_len"].mean()),
        "mean_final_entropy": float(df["final_entropy"].mean()),
        "mean_optimal_stopping_time": float(df["optimal_stopping_time"].mean()),
    }


def plot_entropy_trajectories(trajectories_path: Path, outdir: Path) -> None:
    trajectories = []
    with trajectories_path.open() as f:
        for line in f:
            trajectories.append(json.loads(line))

    if not trajectories:
        return

    max_len = max(len(t) for t in trajectories)
    entropies = []
    for t in range(max_len):
        step_vals = []
        for episode in trajectories:
            if t < len(episode):
                step_vals.append(episode[t]["entropy"])
        if step_vals:
            entropies.append(sum(step_vals) / len(step_vals))

    plt.figure()
    plt.plot(range(1, len(entropies) + 1), entropies)
    plt.xlabel("Step")
    plt.ylabel("Mean entropy")
    plt.title("Entropy trajectory")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_trajectory.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to results.jsonl")
    parser.add_argument("--outdir", default="outputs/phase1")
    parser.add_argument("--plot-entropy", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_results(input_path)
    summary = summarize(df)

    summary_path = outdir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    trajectories_path = input_path.parent / "trajectories.jsonl"
    if args.plot_entropy and trajectories_path.exists():
        plot_entropy_trajectories(trajectories_path, outdir)

    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
