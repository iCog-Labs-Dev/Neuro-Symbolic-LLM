# Stage A3 — FabricPC Predictive-Coding Residual

Stage A3 adapts a frozen language model with low-rank residuals whose latent activations are
iteratively refined by predictive coding. This document describes the A3 implementation only.

## Overview

At each configured layer \(l\), a predictor initializes a latent from the pristine hidden state:

$$z_l^{(0)} = \sigma(h_l^0 A_l)$$

During training, FabricPC updates the latents for \(K\) inference steps. The residual added to the
live hidden-state stream is

$$\tilde h_l = h_l + s z_l B_l$$

and the energy combines the next-token task loss, an optional KL trust region, and a Gaussian
spring that keeps inferred latents near their predictor values:

$$\mathcal{E} = \mathcal{L}_{\text{task}} +
\lambda_{\text{kl}}\mathrm{KL}(p_{\text{base}}\|p) +
\sum_l \frac{1}{2\sigma^2}\|z_l-z_l^{(0)}\|^2$$

The base model stays frozen; only A3's predictor and projection parameters are trainable.
Residuals are injected through the substrate's `modify_fn`, so an earlier selected layer's
residual flows through later layers as part of the live model computation.

At deployment, the model uses \(z_l=\sigma(h_l^0 A_l)\) without iterative inference or labels.
When `lambda_kl` is positive, baseline log-probabilities are obtained during the clean pass and
provided to the graph as data, avoiding a separate baseline forward pass inside the tail. The KL
term uses the same \(T-1\) next-token prediction positions as the task loss, with cached baseline
log-probabilities aligned to those positions.

## FabricPC Implementation vs. Manual Reference

The production A3 path is the FabricPC implementation. There is no separate manual-reference
trainer or alternative CLI.

| Role | File | Responsibility |
|---|---|---|
| A3 model | [`A3_model.py`](A3_model.py) | Defines `A3Config` and `A3Model`; builds the FabricPC graph; obtains pristine hidden states through the substrate cache; runs latent inference; refreshes state and energy at the final \(z^{(K)}\); computes local parameter gradients; provides energy diagnostics and deployment logits. |
| A3 tail node | [`a3_tail.py`](a3_tail.py) | Implements the FabricPC tail node, validates inputs, injects residuals through `modify_fn`, computes task/KL energy, and propagates gradients to latent inputs. |
| Experiment runner | [`run_stage_a3.py`](../../experiments/run_stage_a3.py) | Orchestrates configured experiments, optimizer updates, evaluation, logs, metrics, and checkpoints. Shared config and data helpers are in [`stage_base.py`](../stage_base.py). |
| Configuration | [`a3_true_pc.yaml`](../../configs/stage_A/a3_true_pc.yaml) | Sets adapter, objective, training, and domain options. |
| Tests and equation reference | [`test_a3_true_pc.py`](../../tests/stages/stage_A/test_a3_true_pc.py) | Tests the FabricPC implementation. Test helpers such as `settle_by_hand` calculate expected updates independently from the A3 equations; they are test-only references, not a second implementation used for training. |

### Final-state energy refresh

FabricPC's inference step evaluates energy before updating latents. After \(K\) updates, the
energy stored by the last inference step can therefore describe \(z^{(K-1)}\), rather than the
final latent \(z^{(K)}\). `A3Model.train_grads` refreshes the state after inference, so its reported
graph energy, latent predictions, and local gradients correspond to the final latent. It also
computes an exact final-energy cross-check through the substrate's loss API.

## Files and responsibilities

- [`A3_model.py`](A3_model.py): graph construction, cached clean pass, final-state refresh,
  inference, gradients, deployment, and energy curve.
- [`a3_tail.py`](a3_tail.py): tail energy and gradient behavior, input validation, and residual
  injection.
- [`run_stage_a3.py`](../../experiments/run_stage_a3.py): experiment orchestration and logging.
- [`a3_true_pc.yaml`](../../configs/stage_A/a3_true_pc.yaml): default experiment configuration.
- [`test_a3_true_pc.py`](../../tests/stages/stage_A/test_a3_true_pc.py): integration and
  equation-reference tests.

## Configuration

The default configuration is [`a3_true_pc.yaml`](../../configs/stage_A/a3_true_pc.yaml). It has
four sections:

- `adapter`: latent rank, activation, scale, and selected layers.
- `objective`: inference steps (`infer_steps`), latent step size (`eta_z`), spring variance
  (`sigma2`), optional KL weight (`lambda_kl`), and predictor/projection initialization scales.
- `training`: model, sequence and batch sizes, epochs, learning rate, evaluation cadence, seed,
  optimizer reset policy, output directory, and JIT setting.
- `domains`: datasets, splits, text separators, and optional per-domain epoch settings.

Unknown config sections and keys are rejected by the shared configuration loader.

## Running

Run the experiment with the default A3 configuration:

```bash
python experiments/run_stage_a3.py
```

Run separate experiments for several inference-step counts:

```bash
python experiments/run_stage_a3.py --K 1 2 4
```

Useful overrides include:

```text
--config PATH
--model MODEL
--rank RANK
--eta-z STEP_SIZE
--lambda-kl WEIGHT
--epochs COUNT
--output-dir PATH
--run-name NAME
--eval-batches COUNT
--split test|train|<split-name>
--no-jit
--energy-curve
```

Each `--K` value runs a separate experiment. `--energy-curve` records energy at the initial
latent and after each inference step; it is an eager diagnostic and can be slower.

## Run outputs and logging

The runner writes the resolved configuration, run metadata, metrics, checkpoints, and a final
summary under the configured output directory. Training and evaluation logs report per-domain
and per-epoch progress, task loss/perplexity, energy terms, validation metrics, and final
test-split metrics. The runner also verifies that the frozen substrate parameters remain
unchanged.

## Tests

Run the focused A3 test module:

```bash
pytest tests/stages/stage_A/test_a3_true_pc.py -v
```

The module skips when `fabricpc.graph_assembly` is unavailable. Tests cover:

- hidden states from the substrate cache, including JIT behavior;
- predictor initialization and \(K\)-step latent updates against hand-computed equations;
- exact energy at the final latent and the stale-energy regression;
- gradient values against closed forms and autodiff, plus propagation through selected layers;
- KL energy and use of cached baseline outputs, including alignment to the \(T-1\) next-token
  prediction positions;
- frozen substrate parameter immutability during optimization;
- deployment behavior, residual wiring, and invalid configuration/input handling, including
  initialization-time checks for required `ids` and `base_logp` edges and compatible shapes.
