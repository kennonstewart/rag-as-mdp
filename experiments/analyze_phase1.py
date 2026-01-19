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


def flatten_config(config: dict) -> dict:
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    results_path = run_dir / "results.jsonl"
    df = load_results(results_path)
    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
    return df, config


def summarize(df: pd.DataFrame) -> dict:
    return {
        "episodes": int(len(df)),
        "mean_reward": float(df["total_reward"].mean()),
        "mean_length": float(df["episode_len"].mean()),
        "mean_final_entropy": float(df["final_entropy"].mean()),
        "mean_optimal_stopping_time": float(df["optimal_stopping_time"].mean()),
    }


def plot_entropy_trajectories(trajectories_path: Path, outdir: Path, prefix: str = "") -> None:
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
    filename = "entropy_trajectory.png"
    if prefix:
        filename = f"{prefix}_entropy_trajectory.png"
    plt.savefig(outdir / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to results.jsonl")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        help="One or more run directories containing results.jsonl",
    )
    parser.add_argument("--outdir", default="outputs/phase1")
    parser.add_argument("--plot-entropy", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.run_dirs and not args.input:
        parser.error("--input or --run-dirs is required")

    summaries = []
    if args.run_dirs:
        for run_dir in [Path(p) for p in args.run_dirs]:
            results_path = run_dir / "results.jsonl"
            if not results_path.exists():
                print(f"Skipping {run_dir}: missing results.jsonl")
                continue
            df, config = load_run(run_dir)
            summary = summarize(df)
            summary["run_dir"] = str(run_dir)
            if config:
                summary.update(config)
                flat = flatten_config(config)
                for key, value in flat.items():
                    df[key] = value
            summaries.append(summary)

            if args.plot_entropy:
                trajectories_path = run_dir / "trajectories.jsonl"
                if trajectories_path.exists():
                    plot_entropy_trajectories(trajectories_path, outdir, prefix=run_dir.name)
    else:
        input_path = Path(args.input)
        df = load_results(input_path)
        summary = summarize(df)
        summaries.append(summary)

        trajectories_path = input_path.parent / "trajectories.jsonl"
        if args.plot_entropy and trajectories_path.exists():
            plot_entropy_trajectories(trajectories_path, outdir)

    summary_path = outdir / "summary.json"
    with summary_path.open("w") as f:
        if len(summaries) == 1:
            json.dump(summaries[0], f, indent=2)
        else:
            json.dump(summaries, f, indent=2)

    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
