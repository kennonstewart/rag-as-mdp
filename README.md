# RAG-as-MDP Phase 1

This repo contains a minimal Phase 1 implementation of a RAG-as-MDP environment with a toy 2-hop QA synthetic world, baseline policies, and a small evaluation harness.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Phase 1 experiments

Fixed-horizon baseline:

```bash
python experiments/run_phase1.py --policy fixed --episodes 50 --k 2 --outdir outputs/fixed
```

SB3 DQN baseline (quick sanity run):

```bash
python experiments/run_phase1.py --policy sb3_dqn --episodes 50 --outdir outputs/dqn
```

## Analyze results

```bash
python experiments/analyze_phase1.py --input outputs/fixed/results.jsonl --outdir outputs/fixed
```

## Tests

```bash
pytest -q
```
