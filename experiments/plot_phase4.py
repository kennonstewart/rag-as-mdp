from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["episode_idx"] = range(len(df))
    return df


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def flatten_config(config: dict) -> dict:
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def format_run_label(run_dir: Path, config: dict) -> str:
    policy = config.get("policy")
    if policy == "fixed":
        k = config.get("k")
        if k is not None:
            return f"fixed(k={k})"
    if policy == "sb3_dqn":
        timesteps = config.get("timesteps")
        if timesteps is not None:
            return f"dqn(t={timesteps})"
    if policy == "fqi_roi":
        return "fqi_roi"
    if policy == "dspy_optimized":
        return "dspy_optimized"
    if policy == "dspy_cot":
        return "dspy_cot"
    if policy == "dspy_react":
        return "dspy_react"
    return policy or run_dir.name


def load_run(run_dir: Path) -> Tuple[pd.DataFrame, dict]:
    results_path = run_dir / "results.jsonl"
    df = load_results(results_path)
    config = load_config(run_dir / "config.json")
    return df, config


def load_trajectories(path: Path) -> List[List[dict]]:
    trajectories = []
    with path.open() as f:
        for line in f:
            trajectories.append(json.loads(line))
    return trajectories


def add_metadata(df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["run_dir"] = str(run_dir)
    df["run_label"] = format_run_label(run_dir, config)
    df["policy"] = config.get("policy", "unknown")
    flat = flatten_config(config)
    for key, value in flat.items():
        df[key] = value
    return df


def save_fig(outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / name)
    plt.close()


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value)
    return safe.strip("_") or "run"


def plot_accuracy_vs_cost(df: pd.DataFrame, outdir: Path) -> None:
    if "stopped_with_answer_evidence" not in df.columns or "total_cost" not in df.columns:
        return
    plt.figure()
    sns.scatterplot(
        data=df,
        x="total_cost",
        y="stopped_with_answer_evidence",
        hue="run_label",
        alpha=0.4,
        s=12,
    )
    plt.xlabel("Total cost")
    plt.ylabel("Stopped with answer evidence")
    save_fig(outdir, "accuracy_vs_cost.png")


def plot_reward_vs_cost(df: pd.DataFrame, outdir: Path) -> None:
    if "total_cost" not in df.columns:
        return
    plt.figure()
    sns.scatterplot(
        data=df,
        x="total_cost",
        y="total_reward",
        hue="run_label",
        alpha=0.4,
        s=12,
    )
    plt.xlabel("Total cost")
    plt.ylabel("Total reward")
    save_fig(outdir, "reward_vs_cost.png")


def plot_regret_ecdf(df: pd.DataFrame, outdir: Path) -> None:
    if "regret_steps" not in df.columns:
        return
    plt.figure()
    sns.ecdfplot(data=df, x="regret_steps", hue="run_label")
    plt.xlabel("Episode length - optimal stopping time")
    plt.ylabel("ECDF")
    save_fig(outdir, "regret_ecdf.png")


def trajectories_to_frame(run_label: str, trajectories: List[List[dict]]) -> pd.DataFrame:
    rows = []
    for episode_idx, episode in enumerate(trajectories):
        for step in episode:
            row = {
                "run_label": run_label,
                "episode_idx": episode_idx,
                "step": step.get("step"),
                "entropy": step.get("entropy"),
                "reward": step.get("reward"),
                "action": step.get("action"),
            }
            if "action_cost" in step:
                row["action_cost"] = step.get("action_cost")
            if "compression" in step:
                row["compression"] = step.get("compression")
            rows.append(row)
    return pd.DataFrame(rows)


def plot_compression_trajectory(df: pd.DataFrame, outdir: Path) -> None:
    if "compression" not in df.columns:
        return
    mean_df = df.groupby(["run_label", "step"])["compression"].mean().reset_index()
    plt.figure()
    sns.lineplot(data=mean_df, x="step", y="compression", hue="run_label")
    plt.xlabel("Step")
    plt.ylabel("Mean compression ratio")
    save_fig(outdir, "compression_trajectory_mean.png")


def plot_entropy_trajectory(df: pd.DataFrame, outdir: Path) -> None:
    if "entropy" not in df.columns:
        return
    mean_df = df.groupby(["run_label", "step"])["entropy"].mean().reset_index()
    plt.figure()
    sns.lineplot(data=mean_df, x="step", y="entropy", hue="run_label")
    plt.xlabel("Step")
    plt.ylabel("Mean entropy")
    save_fig(outdir, "entropy_trajectory_mean.png")


def plot_results_suite(df: pd.DataFrame, outdir: Path) -> None:
    plot_accuracy_vs_cost(df, outdir)
    plot_reward_vs_cost(df, outdir)
    plot_regret_ecdf(df, outdir)


def plot_trajectory_suite(df: pd.DataFrame, outdir: Path) -> None:
    plot_entropy_trajectory(df, outdir)
    plot_compression_trajectory(df, outdir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--outdir", default="outputs/phase4_figures")
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Write a full figure set under a subdir per run.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    outdir = Path(args.outdir)

    results_frames = []
    trajectory_frames = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        df, config = load_run(run_dir)
        df = add_metadata(df, run_dir, config)
        results_frames.append(df)

        trajectories_path = run_dir / "trajectories.jsonl"
        if trajectories_path.exists():
            trajectories = load_trajectories(trajectories_path)
            trajectory_frames.append(
                trajectories_to_frame(df["run_label"].iloc[0], trajectories)
            )

    results_df = pd.concat(results_frames, ignore_index=True)
    plot_results_suite(results_df, outdir)

    traj_df = None
    if trajectory_frames:
        traj_df = pd.concat(trajectory_frames, ignore_index=True)
        plot_trajectory_suite(traj_df, outdir)

    if args.per_run:
        for run_label, run_df in results_df.groupby("run_label"):
            run_outdir = outdir / slugify(str(run_label))
            plot_results_suite(run_df, run_outdir)
            if traj_df is not None:
                run_traj = traj_df[traj_df["run_label"] == run_label]
                if not run_traj.empty:
                    plot_trajectory_suite(run_traj, run_outdir)


if __name__ == "__main__":
    main()
