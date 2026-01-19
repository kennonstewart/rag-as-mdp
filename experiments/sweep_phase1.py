from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def expand_grid(grid: Dict[str, List[object]]) -> List[Dict[str, object]]:
    if not grid:
        return []
    keys = sorted(grid.keys())
    values = [grid[key] for key in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append({key: value for key, value in zip(keys, combo)})
    return combos


def slugify(value: object) -> str:
    text = str(value).strip().replace(" ", "_")
    safe = "".join(ch for ch in text if ch.isalnum() or ch in ("_", "-", ".", "="))
    return safe or "value"


def run_label(params: Dict[str, object], keys: Iterable[str]) -> str:
    parts = []
    for key in keys:
        if key in params:
            parts.append(f"{key}={slugify(params[key])}")
    label = "__".join(parts) if parts else "run"
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    return f"{label}__{digest}"


def validate_combo(params: Dict[str, object], fixed: Dict[str, object]) -> None:
    policy = params.get("policy", fixed.get("policy"))
    if policy is None:
        raise ValueError("policy must be specified in grid or fixed params")
    if policy == "fixed":
        if "k" not in params and "k" not in fixed:
            raise ValueError("fixed policy requires k in grid or fixed params")
    elif policy == "sb3_dqn":
        if "timesteps" not in params and "timesteps" not in fixed:
            raise ValueError("sb3_dqn policy requires timesteps in grid or fixed params")


def filter_params(params: Dict[str, object], fixed: Dict[str, object]) -> Dict[str, object]:
    policy = params.get("policy", fixed.get("policy"))
    filtered = dict(params)
    if policy == "fixed":
        filtered.pop("timesteps", None)
    elif policy == "sb3_dqn":
        filtered.pop("k", None)
    return {key: value for key, value in filtered.items() if value is not None}


def build_args(params: Dict[str, object], fixed: Dict[str, object], outdir: Path) -> List[str]:
    merged = {**fixed, **params}
    policy = merged["policy"]
    args = ["--policy", str(policy), "--outdir", str(outdir)]

    def add_flag(key: str) -> None:
        if merged.get(key):
            args.append(f"--{key.replace('_', '-')}")

    def add_value(key: str) -> None:
        if key in merged and merged[key] is not None:
            args.extend([f"--{key.replace('_', '-')}", str(merged[key])])

    add_value("episodes")
    add_value("max_steps")
    add_value("seed")
    add_flag("trajectories")
    add_flag("skip_check_env")

    if policy == "fixed":
        add_value("k")
    elif policy == "sb3_dqn":
        add_value("timesteps")

    return args


def run_command(args: List[str]) -> None:
    subprocess.run(args, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to sweep YAML config")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    sweep = load_yaml(config_path)

    sweep_name = sweep.get("output", {}).get("name", "phase1")
    output_root = Path(sweep.get("output", {}).get("root", "outputs/sweeps"))
    sweep_root = output_root / sweep_name

    fixed = sweep.get("fixed", {}) or {}
    grid = sweep.get("grid", {}) or {}
    postprocess = sweep.get("postprocess", {}) or {}

    combos = expand_grid(grid)
    if not combos:
        raise ValueError("Sweep grid is empty. Provide at least one parameter list.")

    run_dirs: List[Path] = []
    seen: set[Tuple[Tuple[str, object], ...]] = set()

    for params in combos:
        validate_combo(params, fixed)
        filtered_params = filter_params(params, fixed)
        label_keys = sorted(filtered_params.keys())
        key = tuple(sorted(filtered_params.items()))
        if key in seen:
            continue
        seen.add(key)
        label = run_label(filtered_params, label_keys)
        run_dir = sweep_root / label
        run_dirs.append(run_dir)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "run_phase1.py"),
        ]
        cmd.extend(build_args(filtered_params, fixed, run_dir))
        print(f"Running {label} -> {run_dir}")
        run_command(cmd)

        config_snapshot = {
            "sweep_config": str(config_path),
            "sweep_label": label,
            "sweep_name": sweep_name,
            "grid_params": filtered_params,
            "fixed_params": fixed,
        }
        with (run_dir / "sweep_config.json").open("w") as f:
            json.dump(config_snapshot, f, indent=2)

    run_dir_args = [str(path) for path in run_dirs]
    do_analyze = args.analyze or postprocess.get("analyze", True)
    do_plot = args.plot or postprocess.get("plot", True)

    if do_analyze:
        summary_outdir = postprocess.get("summary_outdir", str(sweep_root / "summary"))
        analyze_cmd = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "analyze_phase1.py"),
            "--run-dirs",
            *run_dir_args,
            "--outdir",
            summary_outdir,
        ]
        if postprocess.get("plot_entropy", False):
            analyze_cmd.append("--plot-entropy")
        run_command(analyze_cmd)

    if do_plot:
        figures_outdir = postprocess.get("figures_outdir", str(sweep_root / "figures"))
        plot_cmd = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "plot_phase1.py"),
            "--run-dirs",
            *run_dir_args,
            "--outdir",
            figures_outdir,
        ]
        if postprocess.get("per_run_plots", True):
            plot_cmd.append("--per-run")
        run_command(plot_cmd)


if __name__ == "__main__":
    main()
