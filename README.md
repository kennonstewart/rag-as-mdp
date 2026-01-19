## RAG-as-MDP

Toy **RAG-as-MDP** environment (Gymnasium) + baseline policies + a small experiment/sweep pipeline for studying **optimal stopping** under an information gain vs cost reward.

### Repo architecture

```text
rag-as-mdp/
  rag_mdp/                 # Environment + synthetic world + metrics
    env.py                 # Gymnasium Env: state/action/reward/termination
    synthetic_world.py     # Toy 2-hop QA world + exact entropy + τ* oracle
    metrics.py             # Entropy + reward helpers
    types.py               # Dataclasses used across env/world
  policies/                # Baseline + learned policies
    fixed_horizon.py       # “retrieve for k steps then return”
    sb3_dqn.py             # SB3 DQN training helper
    dspy_cot.py            # DSPy Chain-of-Thought policy
    dspy_react.py          # DSPy ReAct policy
    fqi_roi.py             # Fitted Q-iteration ROI stopping policy
  experiments/             # CLI runners, sweeps, analysis, plotting
    run_phase1.py          # Run one config → writes results + (optional) trajectories
    sweep_phase1.py        # YAML grid sweep → runs + analyze + plots
    analyze_phase1.py      # Aggregate run dirs → summary.json (+ optional entropy plot)
    plot_phase1.py         # Figures for outcomes + (if present) trajectory plots
    sweeps/phase1.yaml     # Default Phase 1 sweep definition
    run_phase2.py          # Phase 2 runner (DSPy policies + baselines)
    sweep_phase2.py        # Phase 2 sweep runner
    sweeps/phase2.yaml     # Phase 2 sweep definition
    train_phase3_mipro.py  # Phase 3 MIPROv2 optimization
    run_phase3.py          # Phase 3 runner (FQI/optimized DSPy + baselines)
    sweep_phase3.py        # Phase 3 sweep runner
    sweeps/phase3.yaml     # Phase 3 sweep definition
    plot_phase4.py         # Phase 4 plots (accuracy/cost, compression, regret)
    run_phase4_interchangeability.py  # Worker interchangeability experiment
  rag_mdp/                 # Environment + synthetic world + metrics
    observation_text.py    # Observation → text adapters for DSPy policies
  outputs/                 # Example sweep outputs + figures (regeneratable)
  tests/                   # Environment/API sanity checks
  Makefile                 # Convenience targets (e.g. sweep-phase1)
  paper/                   # Paper draft artifacts
```

### Environment (`rag_mdp`)

- **State**: a Gymnasium `Dict` observation with:
  - `query_id`: discrete question identifier
  - `nodes`, `edges`: padded graph arrays representing \(G_t\)
  - `entropy`: current **true posterior entropy** \(H(\text{answer}\mid G_t)\) (exact in this toy world)
  - `step`: current step index
- **Actions** (`Discrete(3)`):
  - `0 = retrieve`: add the next missing evidence edge (if any)
  - `1 = reflect`: consolidate graph (Phase 1: no-op placeholder)
  - `2 = return`: stop the episode
- **Reward**:
  - \(r_t = (H_{t-1} - H_t) - \text{action\_cost}(a_t)\)
- **Termination**:
  - `return` action terminates; otherwise truncation at `--max-steps`
- **Ground-truth oracle metric**:
  - `optimal_stopping_time` is computed by brute-forcing “retrieve \(t\) times then return” in the synthetic world.

### Policies (`policies`)

- **Fixed horizon** (`fixed_horizon.py`): retrieves for `k` steps then returns.
- **SB3 DQN baseline** (`sb3_dqn.py`): trains a small `DQN("MultiInputPolicy")` agent, then evaluates it deterministically.
- **DSPy CoT** (`dspy_cot.py`): Chain-of-Thought LLM policy that chooses among `{retrieve, reflect, return}` using a text summary of the evidence graph.
- **DSPy ReAct** (`dspy_react.py`): ReAct-style LLM policy with the same action space and observation summary.
- **FQI ROI** (`fqi_roi.py`): Offline fitted Q-iteration policy using simple features (entropy/step/evidence flags) for ROI-style stopping.

### Experiment runner (`experiments/run_phase1.py`)

Runs one configuration and writes an output directory containing:

- **`config.json`**: run configuration + action costs + timestamp
- **`results.jsonl`**: one JSON object per episode with:
  - `total_reward`, `episode_len`, `final_entropy`, `optimal_stopping_time`, `terminated_reason`
- **`trajectories.jsonl`** (optional via `--trajectories`): one JSON array per episode; each element logs `step`, `entropy`, `action`, `reward`, `action_cost`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### LLM setup (Phase 2/3 DSPy policies)

The DSPy policies require a real LLM API key. Set one of the following:

```bash
export OPENAI_API_KEY="your-key"
# or
export DSPY_API_KEY="your-key"
```

### Phases 1–4 at a glance

#### Phase 1: Foundation (toy RAG-as-MDP)
- **Goal**: Validate the Gymnasium environment + entropy-based reward + oracle optimal stopping.
- **Policies**: Fixed horizon, SB3 DQN.
- **Runner**: `experiments/run_phase1.py`
- **Key outputs**: `results.jsonl`, optional `trajectories.jsonl`, oracle `optimal_stopping_time`.

#### Phase 2: Policy integration (LLM policies + comparison)
- **Goal**: Add DSPy Chain-of-Thought and ReAct policies to the same environment and compare against baselines.
- **Policies**: `dspy_cot`, `dspy_react`, plus `fixed` and `sb3_dqn`.
- **Runner**: `experiments/run_phase2.py`
- **New metrics**: `total_cost`, `stopped_with_answer_evidence`, `regret_steps`.

#### Phase 3: Optimization (DSPy MIPROv2 + ROI stopping)
- **Goal**: Optimize stopping behavior and compare to FQI ROI and SB3 baselines.
- **Training**: `experiments/train_phase3_mipro.py` compiles a DSPy stopping policy.
- **Policies**: `dspy_optimized`, `fqi_roi`, plus `fixed` and `sb3_dqn`.
- **Runner**: `experiments/run_phase3.py`

#### Phase 4: Analysis + interchangeability
- **Goal**: Generate paper-style plots and test robustness to worker variability.
- **Plots**: `experiments/plot_phase4.py` (accuracy/cost, compression, regret).
- **Interchangeability**: `experiments/run_phase4_interchangeability.py` with noisy retrieval and reflective pruning.

### Run a single experiment

Fixed-horizon baseline:

```bash
python experiments/run_phase1.py \
  --policy fixed \
  --episodes 5000 \
  --k 4 \
  --max-steps 5 \
  --seed 0 \
  --trajectories \
  --outdir outputs/fixed_k4_seed0
```

SB3 DQN baseline (train then evaluate):

```bash
python experiments/run_phase1.py \
  --policy sb3_dqn \
  --episodes 5000 \
  --timesteps 10000 \
  --max-steps 5 \
  --seed 0 \
  --trajectories \
  --outdir outputs/dqn_t10000_seed0
```

### Phase 2: Run one config (LLM policies + baselines)

Phase 2 supports `fixed`, `sb3_dqn`, `dspy_cot`, and `dspy_react` (the DSPy policies require an API key; see “LLM setup” above).

DSPy CoT:

```bash
python experiments/run_phase2.py \
  --policy dspy_cot \
  --episodes 200 \
  --max-steps 5 \
  --seed 0 \
  --trajectories \
  --lm-provider openai \
  --lm-model gpt-4o-mini \
  --temperature 0.2 \
  --max-tokens 256 \
  --outdir outputs/phase2_dspy_cot_seed0
```

DSPy ReAct:

```bash
python experiments/run_phase2.py \
  --policy dspy_react \
  --episodes 200 \
  --max-steps 5 \
  --seed 0 \
  --trajectories \
  --lm-provider openai \
  --lm-model gpt-4o-mini \
  --temperature 0.2 \
  --max-tokens 256 \
  --outdir outputs/phase2_dspy_react_seed0
```

Fixed horizon (same idea as Phase 1, but Phase 2 writes extra metrics like `total_cost`):

```bash
python experiments/run_phase2.py \
  --policy fixed \
  --episodes 200 \
  --k 4 \
  --max-steps 5 \
  --seed 0 \
  --trajectories \
  --outdir outputs/phase2_fixed_k4_seed0
```

### Phase 2: Run the default sweep

```bash
python experiments/sweep_phase2.py --config experiments/sweeps/phase2.yaml
```

Outputs land under:

- `outputs/sweeps/phase2/summary/summary.json`
- `outputs/sweeps/phase2/figures/`
- `outputs/sweeps/phase2/<run_dir>/` (per-run artifacts)

### Phase 3: Train the DSPy stopping policy (MIPROv2)

Phase 3’s `dspy_optimized` policy loads a compiled program from `--compiled-path` (defaults to `outputs/phase3_mipro/compiled_policy.json`).

```bash
python experiments/train_phase3_mipro.py \
  --episodes 200 \
  --max-steps 5 \
  --seed 0 \
  --confidence-threshold 0.5 \
  --lm-provider openai \
  --lm-model gpt-4o-mini \
  --temperature 0.2 \
  --max-tokens 256 \
  --outdir outputs/phase3_mipro
```

### Phase 3: Run one config (FQI ROI + optimized DSPy + baselines)

Optimized DSPy (uses the compiled policy from the previous step):

```bash
python experiments/run_phase3.py \
  --policy dspy_optimized \
  --episodes 200 \
  --max-steps 5 \
  --seed 0 \
  --trajectories \
  --compiled-path outputs/phase3_mipro/compiled_policy.json \
  --confidence-threshold 0.5 \
  --lm-provider openai \
  --lm-model gpt-4o-mini \
  --temperature 0.2 \
  --max-tokens 256 \
  --outdir outputs/phase3_dspy_optimized_seed0
```

FQI ROI (offline fitted Q-iteration over `{retrieve, return}`):

```bash
python experiments/run_phase3.py \
  --policy fqi_roi \
  --episodes 200 \
  --max-steps 5 \
  --seed 0 \
  --trajectories \
  --fqi-episodes 200 \
  --fqi-iters 20 \
  --fqi-ridge 1e-3 \
  --fqi-gamma 0.9 \
  --outdir outputs/phase3_fqi_roi_seed0
```

### Phase 3: Run the default sweep

If you haven’t trained MIPRO yet, run the Phase 3 training step above first (the default sweep expects `outputs/phase3_mipro/compiled_policy.json` to exist).

```bash
python experiments/sweep_phase3.py --config experiments/sweeps/phase3.yaml
```

Outputs land under:

- `outputs/sweeps/phase3/summary/summary.json`
- `outputs/sweeps/phase3/figures/`
- `outputs/sweeps/phase3/<run_dir>/` (per-run artifacts)

### Analyze results

Analyze one `results.jsonl`:

```bash
python experiments/analyze_phase1.py \
  --input outputs/dqn_t10000_seed0/results.jsonl \
  --outdir outputs/dqn_t10000_seed0
```

Analyze multiple run directories (also joins `config.json` fields into the summary):

```bash
python experiments/analyze_phase1.py \
  --run-dirs outputs/fixed_k4_seed0 outputs/dqn_t10000_seed0 \
  --outdir outputs/phase1_summary
```

### Plot results

```bash
python experiments/plot_phase1.py \
  --run-dirs outputs/fixed_k4_seed0 outputs/dqn_t10000_seed0 \
  --outdir outputs/phase1_figures \
  --per-run
```

### Sweep logic (`experiments/sweep_phase1.py`)

The sweep config is a YAML with three key sections:

- **`fixed`**: parameters applied to every run (e.g. `episodes`, `max_steps`, `trajectories`)
- **`grid`**: parameter lists; the sweep runs the Cartesian product
- **`postprocess`**: whether to run analysis/plots after the sweep, plus output locations

Important details of the sweep implementation:

- **Policy-aware grids**: combinations are validated and then **filtered** so incompatible parameters don’t get passed (e.g. `k` is removed for DQN runs; `timesteps` removed for fixed-horizon runs).
- **Run directory naming**: each run is labeled as `key=value__key=value__<8-char-sha1>`, where the hash prevents collisions from long labels and makes paths stable.
- **Artifacts**: each run directory also gets a `sweep_config.json` snapshot with the grid/fixed params and sweep metadata.

Run the default Phase 1 sweep:

```bash
python experiments/sweep_phase1.py --config experiments/sweeps/phase1.yaml
```

Or via Makefile:

```bash
make sweep-phase1
```

You can override the sweep config path:

```bash
make sweep-phase1 SWEEP_CONFIG=experiments/sweeps/phase1.yaml
```

### Visualizations (`experiments/plot_phase1.py`)

For a set of run dirs, the plotter writes a suite of figures to `--outdir`:

- **Outcome plots** (always, from `results.jsonl`):
  - `reward_violin.png`, `length_violin.png`, `final_entropy.png`
  - `reward_vs_length.png`
  - `stopping_scatter.png` (episode length vs oracle `optimal_stopping_time`)
  - `regret_ecdf.png` where regret is `(episode_len - optimal_stopping_time)`
  - `termination_reasons.png`
  - `timeseries_episode_total_reward.png`, `timeseries_episode_episode_len.png`, etc.
- **Trajectory plots** (only if `trajectories.jsonl` exists in any run dir):
  - `entropy_trajectory_mean.png`
  - `timeseries_entropy.png`, `timeseries_reward.png`, `timeseries_action_cost.png`
  - `action_by_step.png`, `cost_trajectory_mean.png`

Example sweep outputs (generated by the default sweep) live under `outputs/sweeps/phase1/`:

- `outputs/sweeps/phase1/summary/summary.json`
- `outputs/sweeps/phase1/figures/` (global figures)
- `outputs/sweeps/phase1/<run_dir>/` (per-run artifacts)

Selected example figures:

![Reward vs length](outputs/sweeps/phase1/figures/reward_vs_length.png)
![Stopping scatter](outputs/sweeps/phase1/figures/stopping_scatter.png)
![Termination reasons](outputs/sweeps/phase1/figures/termination_reasons.png)

### Tests

```bash
pytest -q
```
