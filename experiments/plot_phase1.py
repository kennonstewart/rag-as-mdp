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


def plot_reward_distribution(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    sns.violinplot(data=df, x="run_label", y="total_reward", inner="box", cut=0)
    sns.stripplot(data=df, x="run_label", y="total_reward", size=1, alpha=0.3, color="black")
    plt.xlabel("Run")
    plt.ylabel("Total reward")
    save_fig(outdir, "reward_violin.png")


def plot_length_distribution(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    sns.violinplot(data=df, x="run_label", y="episode_len", inner="box", cut=0)
    sns.stripplot(data=df, x="run_label", y="episode_len", size=1, alpha=0.3, color="black")
    plt.xlabel("Run")
    plt.ylabel("Episode length")
    save_fig(outdir, "length_violin.png")


def plot_stopping_scatter(df: pd.DataFrame, outdir: Path) -> None:
    g = sns.relplot(
        data=df,
        x="optimal_stopping_time",
        y="episode_len",
        hue="terminated_reason",
        col="run_label",
        kind="scatter",
        alpha=0.4,
        s=10,
        facet_kws={"sharex": False, "sharey": False},
    )
    for ax in g.axes.flat:
        if ax is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            min_val = min(xlim[0], ylim[0])
            max_val = max(xlim[1], ylim[1])
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    g.set_axis_labels("Optimal stopping time", "Episode length")
    save_fig(outdir, "stopping_scatter.png")


def plot_regret_ecdf(df: pd.DataFrame, outdir: Path) -> None:
    df = df.copy()
    df["regret_steps"] = df["episode_len"] - df["optimal_stopping_time"]
    plt.figure()
    sns.ecdfplot(data=df, x="regret_steps", hue="run_label")
    plt.xlabel("Episode length - optimal stopping time")
    plt.ylabel("ECDF")
    save_fig(outdir, "regret_ecdf.png")


def plot_reward_vs_length(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    sns.scatterplot(
        data=df,
        x="episode_len",
        y="total_reward",
        hue="run_label",
        alpha=0.3,
        s=12,
    )
    plt.xlabel("Episode length")
    plt.ylabel("Total reward")
    save_fig(outdir, "reward_vs_length.png")


def plot_termination_reasons(df: pd.DataFrame, outdir: Path) -> None:
    counts = (
        df.groupby(["run_label", "terminated_reason"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("run_label")["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    plt.figure()
    sns.barplot(
        data=counts,
        x="run_label",
        y="proportion",
        hue="terminated_reason",
    )
    plt.xlabel("Run")
    plt.ylabel("Proportion")
    save_fig(outdir, "termination_reasons.png")


def plot_final_entropy(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    sns.violinplot(data=df, x="run_label", y="final_entropy", inner="box", cut=0)
    sns.stripplot(data=df, x="run_label", y="final_entropy", size=1, alpha=0.3, color="black")
    plt.xlabel("Run")
    plt.ylabel("Final entropy")
    save_fig(outdir, "final_entropy.png")


def plot_episode_timeseries(df: pd.DataFrame, outdir: Path, metric: str) -> None:
    if metric not in df.columns:
        return
    plt.figure()
    sns.lineplot(data=df, x="episode_idx", y=metric, hue="run_label", alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel(metric.replace("_", " ").title())
    save_fig(outdir, f"timeseries_episode_{metric}.png")


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
            rows.append(row)
    return pd.DataFrame(rows)


def plot_entropy_trajectories(df: pd.DataFrame, outdir: Path) -> None:
    mean_df = df.groupby(["run_label", "step"])["entropy"].mean().reset_index()
    plt.figure()
    sns.lineplot(data=mean_df, x="step", y="entropy", hue="run_label")
    plt.xlabel("Step")
    plt.ylabel("Mean entropy")
    save_fig(outdir, "entropy_trajectory_mean.png")


def plot_timeseries_metric(df: pd.DataFrame, outdir: Path, metric: str) -> None:
    if metric not in df.columns:
        return
    mean_df = df.groupby(["run_label", "step"])[metric].mean().reset_index()
    plt.figure()
    sns.lineplot(data=mean_df, x="step", y=metric, hue="run_label")
    plt.xlabel("Step")
    plt.ylabel(f"Mean {metric}")
    save_fig(outdir, f"timeseries_{metric}.png")


def plot_action_by_step(df: pd.DataFrame, outdir: Path) -> None:
    action_map = {0: "retrieve", 1: "reflect", 2: "return"}
    df = df.copy()
    df["action_label"] = df["action"].map(action_map).fillna("unknown")
    counts = (
        df.groupby(["run_label", "step", "action_label"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(["run_label", "step"])["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    g = sns.relplot(
        data=counts,
        x="step",
        y="proportion",
        hue="run_label",
        col="action_label",
        kind="line",
        facet_kws={"sharey": True, "sharex": True},
    )
    g.set_axis_labels("Step", "Action probability")
    save_fig(outdir, "action_by_step.png")


def plot_cost_trajectory(df: pd.DataFrame, outdir: Path) -> None:
    if "action_cost" not in df.columns:
        return
    mean_df = df.groupby(["run_label", "step"])["action_cost"].mean().reset_index()
    plt.figure()
    sns.lineplot(data=mean_df, x="step", y="action_cost", hue="run_label")
    plt.xlabel("Step")
    plt.ylabel("Mean action cost")
    save_fig(outdir, "cost_trajectory_mean.png")


def plot_results_suite(df: pd.DataFrame, outdir: Path) -> None:
    plot_reward_distribution(df, outdir)
    plot_length_distribution(df, outdir)
    plot_stopping_scatter(df, outdir)
    plot_regret_ecdf(df, outdir)
    plot_reward_vs_length(df, outdir)
    plot_termination_reasons(df, outdir)
    plot_final_entropy(df, outdir)
    plot_episode_timeseries(df, outdir, "total_reward")
    plot_episode_timeseries(df, outdir, "episode_len")
    plot_episode_timeseries(df, outdir, "final_entropy")
    plot_episode_timeseries(df, outdir, "optimal_stopping_time")


def plot_trajectory_suite(df: pd.DataFrame, outdir: Path) -> None:
    plot_entropy_trajectories(df, outdir)
    plot_timeseries_metric(df, outdir, "entropy")
    plot_timeseries_metric(df, outdir, "reward")
    plot_timeseries_metric(df, outdir, "action_cost")
    plot_action_by_step(df, outdir)
    plot_cost_trajectory(df, outdir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--outdir", default="outputs/phase1_figures")
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
