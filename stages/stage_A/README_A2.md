# Stage A2: auxiliary losses on a frozen transformer

A2 trains a residual adapter with hidden prediction, contrastive future prediction
(CPC), and stability objectives. Training uses ordinary backpropagation; FabricPC
supplies differentiable heads and energy components. Latent settling belongs to A3.

## Files

| File | Purpose |
| --- | --- |
| [a2.py](a2.py) | Objective, configuration, ablation arms, optimizer, and audit |
| [a2_reporting.py](a2_reporting.py) | Internal metrics for saved checkpoints |
| [run_stage_a2.py](../../experiments/run_stage_a2.py) | Training and evaluation CLI |
| [a2_pc_inspired.yaml](../../configs/stage_A/a2_pc_inspired.yaml) | Main GPT-2 experiment across three LexGLUE domains |
| [a2_smoke.yaml](../../configs/stage_A/a2_smoke.yaml) | Small GPT-2/WikiText integration configuration |
| [residual/](../../residual/) | Predictors, CPC heads, stability metrics, and loss functions |

## Objective and gradient ownership

```text
L_A2 = L_task + lambda_kl * KL(base || adapted) + lambda_wd * ||adapter||^2
       + lambda_hid * L_hidden + lambda_cp * L_CPC + lambda_stab * L_stability

L_stability = w_anch * L_base_anchor + w_smooth * L_residual_smoothness
              + w_hep * L_projected_error_norm + w_step * L_step
```

| Term | Parameters updated |
| --- | --- |
| Task, base KL, adapter weight regularization | Adapter |
| Hidden prediction, detached input | Hidden predictor only |
| Hidden prediction, coupled input | Hidden predictor and adapter |
| CPC | CPC heads and adapter |
| Base-state anchor and residual smoothness | Adapter |
| Projected-error norm control | Trainable highway only |
| Settling-step loss | None: zero in A2 |

Setting all auxiliary weights to zero gives the A1 control. Teachers and CPC
positive targets are detached. Evaluation uses causal feedforward predictions.

### Hidden targets

- `corrected_teacher`: the next frozen block's output on the actual corrected
  sequence trajectory. This default follows the auxiliary-loss review and October
  implementation clarification.
- `base_anchor`: the next-layer state from a separate clean-base forward pass,
  matching the example in research plan v8, Section 7.4.

Report these as separate experiments. A flexible predictor fitting the base
target does not itself bound hidden-state drift.

The main YAML uses `detach_input: true` as a predictor-only diagnostic. The
`full` and `hid_coupled` arms enable hidden-loss gradients into the adapter.
`predictor_lr_mult` scales the predictor LR relative to `training.learning_rate`.

### HEP and stability interpretation

A2 does not inject a highway error signal into latent states. The
`hep_norm_control` arm minimizes a projected-error norm and trains only the
highway; a zero highway minimizes that penalty. `stab_hep` is a compatibility
alias. Fixed-random projections are diagnostics with `w_hep: 0`.

The base-state anchor regularizes drift from the clean frozen trajectory.
`L_step` is zero because A2 performs no settling steps; it does not establish
convergence of an iterative process.

## GPU setup

Run all commands below from the **repository root**, using Linux with an NVIDIA
GPU and Python 3.12:

```bash
git switch a2-final
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,gpu]'
export JAX_PLATFORMS=cuda
```

Use the dependencies declared by the checkout you are validating. After
integration into `Base`, validate against that branch's dependency set.
The A2 dependency pin is FabricPC v0.6.0, matching `Base`. Graph assembly uses
`fabricpc.graph_assembly`; initialization uses `fabricpc.graph_initialization`.

## Run A2

Start with the small integration configuration:

```bash
python experiments/run_stage_a2.py \
  --config configs/stage_A/a2_smoke.yaml \
  --arm full --audit --run-name a2-smoke
```

Run the main A2 experiment and its matched A1 control:

```bash
python experiments/run_stage_a2.py \
  --config configs/stage_A/a2_pc_inspired.yaml \
  --arm full --seed 0 --audit --run-name a2-full

python experiments/run_stage_a2.py \
  --config configs/stage_A/a2_pc_inspired.yaml \
  --arm a1_mode --seed 0 --run-name a1-control
```

Use a new run name each time; existing run directories are protected against
overwriting. The default output directory is `runs/`. Use `--output-dir` to
change it and `--epochs` to override epochs for every domain.

| Arm | Experiment |
| --- | --- |
| `a1_mode` | Task/KL/weight regularization without auxiliary objectives |
| `hid_detached` / `hid_coupled` | Predictor-only control / hidden-loss adapter regularization |
| `cpc_base` / `cpc_corrected` | CPC with clean-base / corrected future targets |
| `cpc_multi` | CPC with horizons 1, 2, and 4 |
| `stab_anch` | Base-state anchor |
| `stab_smooth` | Anchor plus residual smoothness |
| `hep_norm_control` | Anchor plus trainable highway norm-only control |
| `stab_all` | All stability components, including the inactive step term |
| `full` | Base-target CPC, base-state anchor, and coupled hidden prediction |

Use `--hidden-target base_anchor` to select the clean-base hidden target.
Explicit `--lambda-*` arguments override the arm's weights.

## Evaluate a saved checkpoint

Always use the saved run configuration so the adapter and auxiliary heads match
the checkpoint:

```bash
python experiments/run_stage_a2.py \
  --config runs/a2-smoke/config.yaml \
  --eval-checkpoint runs/a2-smoke/params/task_0_wikitext.npz \
  --split validation --report-internals

python experiments/run_stage_a2.py \
  --config runs/a2-full/config.yaml \
  --eval-checkpoint runs/a2-full/params/task_2_scotus.npz \
  --split test --report-internals
```

Checkpoints preserve the combined adapter/head tree. Causal evaluation selects
only adapter parameters before strict key and shape validation.
Omit `--report-internals` to return task evaluation only.

Internal reporting includes per-layer residual size, read/write redundancy,
selected-target predictor and copy errors, and task/auxiliary adapter-gradient
conflict. Gradient conflict uses the first evaluation batch; representation
statistics use all supplied batches. Reporting applies no parameter updates.

## Results and diagnostics

Each run writes:

- `config.yaml` and `meta.json`: resolved settings and experiment metadata.
- `metrics.jsonl`: training/evaluation events, losses, and diagnostics.
- `params/task_<index>_<domain>.npz`: combined parameter checkpoints.
- `results.json`: causal losses, continual-learning summaries, frozen-weight
  integrity, timing, device memory availability, and skipped CPC counts.
- `audit.json`: audit results when training with `--audit`.

`--audit-only` loads the model and data, prints checks, and exits without training.
`--audit` stops training if a check fails. Checks include initialization parity,
loss decomposition, gradient ownership/finiteness, and frozen weights.

Predictor summaries `pred_rel_selected_mean` and `copy_rel_selected_mean`
follow the configured hidden target. The older `pred_rel_mean` and
`copy_rel_mean` continue to describe the corrected teacher.

## Tests and limits

Run the full residual suite in your configured GPU environment:

```bash
python -m pytest tests/residual/ -v --tb=short
```

Regression coverage includes combined-checkpoint CLI reload, strict adapter
validation, zero-alpha HEP audits, masked stability metrics, inactive CPC,
nonfinite settings, empty hidden reductions, and selected-target reporting.
Runtime and GPU validation remain pending; no A2-versus-A1 quality improvement
or benchmark result is claimed.

CPC needs at least two sequences per training batch and valid cross-sequence
negatives. Inactive masked batches report zero contributing anchors, warn once,
and increment `cpc_skipped_batches`. Next-token training requires `seq_len >= 2`.

CPC allocates an `N x N` similarity matrix, where `N` is batch size times sampled
positions. Reduce batch size, sequence length, or the anchor cap before scaling.
Device memory statistics are available only when supplied by the backend; they
are not measured peaks. Auxiliary weights still require experimental calibration.
