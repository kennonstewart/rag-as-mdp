SWEEP_CONFIG ?= experiments/sweeps/phase1.yaml

.PHONY: sweep-phase1
sweep-phase1:
	python experiments/sweep_phase1.py --config $(SWEEP_CONFIG)
