# Neuro-Symbolic LLM: Stage A2

Stage A2 trains residual adapters on a frozen transformer with hidden prediction,
contrastive future prediction, and stability objectives using backpropagation.
See [the A2 implementation guide](stages/stage_A/README_A2.md) for the objective,
ablation arms, gradient ownership, diagnostics, and checkpoint format.

GPU commands below assume Linux with an NVIDIA GPU and Python 3.12. Run them
from the repository root. Installing the GPU extra supplies JAX CUDA support.

```bash
git switch a2-final
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,gpu]'
export JAX_PLATFORMS=cuda
```

Run the residual tests and a small model/data integration experiment on your GPU:

```bash
python -m pytest tests/residual/ -v --tb=short
python experiments/run_stage_a2.py --config configs/stage_A/a2_smoke.yaml --arm full --audit --run-name a2-smoke
python experiments/run_stage_a2.py --config runs/a2-smoke/config.yaml --eval-checkpoint runs/a2-smoke/params/task_0_wikitext.npz --split validation --report-internals
```

For the main three-domain A2 experiment and its matched A1 control:

```bash
python experiments/run_stage_a2.py --config configs/stage_A/a2_pc_inspired.yaml --arm full --seed 0 --audit --run-name a2-full
python experiments/run_stage_a2.py --config configs/stage_A/a2_pc_inspired.yaml --arm a1_mode --seed 0 --run-name a1-control
python experiments/run_stage_a2.py --config runs/a2-full/config.yaml --eval-checkpoint runs/a2-full/params/task_2_scotus.npz --split test --report-internals
```

Use a new run name for each invocation; existing run directories are protected
against overwriting. The `full` arm couples hidden prediction to the adapter;
the shipped YAML's detached predictor is a separate predictor-only diagnostic.
Use `--hidden-target base_anchor` for the research plan's clean-base target,
and compare it separately with the clarification's `corrected_teacher` default.

`results.json` contains causal task loss, continual-learning summaries, frozen
weight integrity, timing, skipped CPC counts, and device memory availability.
Checkpoints preserve auxiliary heads; causal checkpoint evaluation selects and
strictly validates only the adapter parameters.

The PR #17 review changes include regression tests for checkpoint reload,
zero-alpha HEP audits, masked diagnostics, inactive CPC, empty hidden reductions,
nonfinite settings, and selected-target reporting. These changes have not been
runtime-validated here; tests and model runs are left for your GPU environment.
