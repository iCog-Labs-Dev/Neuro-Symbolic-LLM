# Stage A2 — PC-inspired auxiliary losses on the residual adapter

A2 keeps the A1 setup (a small trainable residual on the hidden states of a frozen language model) and adds **auxiliary losses on the hidden states**. Training is still plain backpropagation with one optimiser: there is **no settling** in A2, so "PC-inspired" describes the shape of the losses, not the way the weights are trained. FabricPC supplies the nodes where it applies (the contrastive heads and the squared-error energies).



> **Status.** Built and tested: the objective wiring, the contrastive future loss, the stability loss, the hidden-prediction loss `L_hid` with its local predictors, the stability monitor (residual smoothness, settling-step metrics), the runner, ablation arms and the audit. 

---

## Contents

- [The objective](#the-objective)
- [The loss terms](#the-loss-terms)
- [Where FabricPC is used](#where-fabricpc-is-used)
- [Parameters](#parameters)
- [Configuration](#configuration)
- [Ablation arms](#ablation-arms)
- [Running](#running)
- [What is logged](#what-is-logged)
- [Internal metrics](#internal-metrics)
- [The audit (checkpoints you can inspect)](#the-audit-checkpoints-you-can-inspect)
- [Code layout](#code-layout)
- [Tests and guarantees](#tests-and-guarantees)
- [Design notes and caveats](#design-notes-and-caveats)
- [Adding a loss](#adding-a-loss)

---

## The objective

$$\mathcal{L}_{A2} = \underbrace{\mathcal{L}_{task} + \lambda_{kl}\,\mathrm{KL}(p_{F_0}\,\|\,p_{F_0+R}) + \lambda_{wd}\lVert\phi\rVert^2}_{\text{A1, unchanged}} + \lambda_{cp}\,\mathcal{L}_{cpc} + \lambda_{stab}\,\mathcal{L}_{stab} + \lambda_{hid}\,\mathcal{L}_{hid}$$

$$\mathcal{L}_{stab} = w_{anch}\,\mathcal{L}_{zanch} + w_{hep}\,\mathcal{L}_{hep\text{-}nudge} + w_{smooth}\,\mathcal{L}_{res\text{-}smooth} + w_{step}\,\mathcal{L}_{step}$$

With `cpc.weight = stab.weight = lambda_hid = 0` the objective, the parameter tree and the gradients are exactly A1's (a test checks it).

```text
input ids ─► frozen model ─────────────────────────────► h^base_l  (stop-gradient), base logits
         │
         └─► frozen model + adapter hook on block l ───► h~_l = h_l + R(h_l), adapted logits
                                                          │
   task CE  +  λ_kl·KL  +  λ_wd·‖φ‖²  +  λ_cp·L_cpc  +  λ_stab·L_stab  +  λ_hid·L_hid
                                  │
                      jax.grad over the whole parameter tree ─► optax.adam
```

`h_l` is the block output before the adapter; `h~_l` is what the next block receives. The base forward is run only when a term needs it (KL, a base-state positive, or the anchor).

## The loss terms

| Term | Formula | Trains |
|---|---|---|
| `L_cpc` | InfoNCE: from `h~_{l,t}` predict the stop-gradient code of the base (or corrected) state `k` positions ahead | adapter, `cpcq_<l>`, `cpca_<l>_<k>` |
| `L_zanch` | `‖h~_l − sg(h^base_l)‖²`, normalised by the base norm (`normalize: relative`) | adapter |
| `L_hep-nudge` | `‖α · sg(ε_y) · V‖²` with `ε_y = softmax(logits) − onehot(next token)` | highway `hep_<l>` only |


**Contrastive future loss.** Code `c = q(h~_t)` (parameter-free LayerNorm, then a bias-free linear projection to `dim`), prediction `p = A_{l,k} c` (bias-free linear, initialised near identity), both L2-normalised. The positive is `q(h_{t+k})` under stop-gradient, taken from the base state (`positive: base`) or the corrected state (`positive: corrected`). Negatives are the positives of the **other sequences** in the batch. Anchors are subsampled by a deterministic stride (`max_anchors_per_seq`). Padding is masked: an anchor counts only if both `t` and `t+k` are real tokens and at least one real negative exists. Several offsets (`offsets: [1, 2, 4]`) each get their own predictor; the term is the mean over sites `(l, k)`.

**Anchor.** A squared distance between the corrected and the base state, averaged over real tokens and sites. It is 0 at initialisation (`B = 0`) and pulls the adapter back toward the base model.

**HEP nudge.** The output error `ε_y` is a training-time-only signal that is projected into the hidden space by a "highway" matrix `V`. Because `ε_y` is under stop-gradient, this loss reaches **only `V`**, never the adapter. With a trainable highway (`dense`, `lowrank`) it trains `V`; with a fixed one (`fixed_random`) it has no gradient and is a **logged metric** (`hep_nudge_norm`, `hep_nudge_ratio`), so the config rejects `w_hep > 0` for fixed highways.

**Residual smoothness.** `R = h~ − h` is the correction at a site and `h` the hidden state the adapter receives. The loss is the squared difference of the correction between neighbouring tokens, divided by the same quantity for the input (`normalize: relative`, a scale-free ratio; below 1 means the correction is smoother than its input). The input's jitter is under stop-gradient, so the term only moves the adapter. It is 0 at initialisation. Pairs of tokens count only if both are real. The squared differences are FabricPC `IdentityNode` energies (`backend: fabricpc`) or plain arithmetic (`jax`); the two agree in value and gradient.



## Parameters

One flat, two-level tree, the same shape as A1's, so `run_sequential`, checkpoints (`RunWriter.save_params`) and `evaluate` work unchanged:

| Key | Leaves | Present when |
|---|---|---|
| `layer_<l>` | `A [d,r]`, `B [r,d]` | always (the adapter) |
| `cpcq_<l>` | `W [d,dim]` | `cpc.weight > 0` |
| `cpca_<l>_<k>` | `M [dim,dim]` | `cpc.weight > 0`, one per offset |
| `hep_<l>` | `V [vocab,d]` (`dense`) or `U, W` (`lowrank`) | `stab.weight > 0` and a trainable `hep.kind` |
| `pred_layer_<l>` | FabricPC node weights of `P_l`, keyed `pred_hidden.w.W_ff1`, `pred_output.w.W`, … | `hid.weight > 0`, one per predictor site |

`init_a2_params(config, adapter)` builds it; with every auxiliary term off it equals `adapter.init_params()`. `adapter_params(tree)` returns the `layer_<l>` sub-tree; the weight decay `λ_wd‖φ‖²` and `ResidualAdapter.num_params` use **only** that sub-tree. `param_counts(tree)` reports `adapter`, `cpc`, `hep`, `pred` and `total`. Head initialisation uses its own seeded stream, so it never collides with the adapter's.

## Configuration

The λ's and every loss setting are in the `objective:` section of [`configs/stage_A/a2_pc_inspired.yaml`](../../configs/stage_A/a2_pc_inspired.yaml); the dataclasses are in `a2.py`. Unknown keys are rejected at every level. `adapter`, `training` and `domains` are identical to A1's YAML (a test checks it), so an A2 run differs from A1 only in `objective`.

```yaml
objective:                  # → A2Config
  lambda_kl: 0.0            # A1 terms
  lambda_wd: 0.0
  seed: 0                   # seed of the heads and of a fixed-random highway
  cpc:                      # → CPCConfig
    weight: 0.05            # λ_cp; 0 switches the term and its heads off
    offsets: [1]            # future offsets k (paper: 1, 2, 4)
    dim: 32                 # width of the CPC code
    tau: 0.1                # temperature
    positive: base          # base | corrected
    max_anchors_per_seq: 256
    a_init_noise: 0.01
    heads: fabricpc         # fabricpc | jax
  stab:                     # → StabLossConfig (the stability *loss*)
    weight: 0.1             # λ_stab; 0 switches the term off
    w_anch: 1.0
    w_hep: 0.0              # > 0 needs hep.kind: lowrank | dense
    w_smooth: 0.1           # residual smoothness
    w_step: 1.0             # settling-step loss (exactly 0 in A2)
    normalize: relative     # relative | none
    backend: fabricpc       # fabricpc | jax
    hep:                    # → HEPConfig
      kind: fixed_random    # null | dense | lowrank | fixed_random
      rank: 16
      alpha: 1.0
  hid:                      # → HiddenConfig (hidden prediction)
    lambda_hid: 0.1         # λ_hid (alias of `weight`); 0 switches the term and the predictors off
    hidden_target: corrected_teacher  # corrected_teacher (Option B) | base_anchor (Option A)
    hidden_norm: target_var # target_var | none
    skip_first_token: true
    predictor_lr_mult: 1.0  # predictor lr = training.learning_rate * this
    predictor:              # → PredictorConfig (residual/local_predictor.py)
      ff_dim: 256
      out_init_std: null
      predict_delta: false  # true: P_l(u) = u + MLP(u)
      detach_input: true    # true: only P_l trains (negative control); false: reaches the adapter
      layers: null          # null: every adapted layer with a following block
      seed: 0               # seed of the predictor weights
  monitor:                  # → residual.stability.StabilityConfig (metrics, never in the loss)
    mode: diagnostic        # off | diagnostic
    layers: null
    delta: 1.0e-6
```

| Setting | Default | Notes |
|---|---|---|
| `cpc.weight` (λ_cp) | 0.0 (YAML: 0.05) | first runs: small |
| `stab.weight` (λ_stab) | 0.0 (YAML: 0.1) | |
| `hid.lambda_hid` / `hid.weight` (λ_hid) | 0.0 (YAML: 0.1) | the error is target-variance normalised, so it is about 1 for a copy |
| `stab.w_anch` / `stab.w_hep` | 1.0 / 0.0 | HEP needs a trainable highway |
| `stab.w_smooth` / `stab.w_step` | 0.0 / 0.0 (YAML: 0.1 / 1.0) | `w_step` has no effect until a stage has settling |
| `cpc.tau` | 0.1 | 0.05 – 0.2 |
| `cpc.dim` | 32 | 32 or 64 |


## Ablation arms

An arm chooses which terms run and their variant; `λ_kl`, `λ_wd`, data, optimiser and seeds are shared, so contrasts against `a1_mode` isolate one term. The λ values come from the config (a weight of 0 falls back to cpc 0.05 / stab 0.1 / hid 0.1). The monitor is measurement-only and is never switched by an arm: set `stability.mode` (or `--stability-mode`) to log it in any arm.

| `--arm` | Terms on |
|---|---|
| `a1_mode` | none: the A1 objective |
| `stab_anch` | anchor only |
| `stab_hep` | anchor + trainable low-rank HEP highway |
| `stab_smooth` | anchor + residual smoothness |
| `stab_all` | all four stability losses (anchor, HEP nudge on a trainable low-rank highway, residual smoothness, step) |
| `cpc_base` | contrastive future, positive from the base state |
| `cpc_corrected` | contrastive future, positive from the corrected state |
| `cpc_multi` | contrastive future, offsets 1, 2, 4 |
| `hid_detached` | hidden prediction with `detach_input: true`: trains `P_l` only (negative control) |
| `hid_coupled` | hidden prediction with `detach_input: false`: the prediction error reaches the adapter |
| `full` | `cpc_base` + `stab_anch` + `hid_coupled` |

## Running

```bash
# The shipped configuration
python experiments/run_stage_a2.py --config configs/stage_A/a2_pc_inspired.yaml

# Ablation arms, one seed each (sets adapter, training and objective seeds)
python experiments/run_stage_a2.py --arm a1_mode --seed 0
python experiments/run_stage_a2.py --arm cpc_base --seed 0
python experiments/run_stage_a2.py --arm full --seed 0

# Weights from the command line (override the YAML and the arm)
python experiments/run_stage_a2.py --lambda-cp 0.1 --lambda-stab 0 --lambda-hid 0
python experiments/run_stage_a2.py --arm hid_coupled --stability-mode diagnostic
python experiments/run_stage_a2.py --lambda-hid 0.1 --hidden-target base_anchor --no-detach-input
python experiments/run_stage_a2.py --lambda-kl 0.1 --lambda-wd 1e-4

# Smoke run, and evaluating a saved checkpoint (the auxiliary heads are ignored)
python experiments/run_stage_a2.py --rank 8 --epochs 1 --run-name smoke
python experiments/run_stage_a2.py --eval-checkpoint runs/<run>/params/task_2_scotus.npz
```


> `run_sequential` JIT-compiles the training step (`training.jit: true`); a test compares jit and eager values and gradients of the A2 objective.

## What is logged

`terms` always holds `total` and `task`; every key is logged as `loss_<name>` in `metrics.jsonl` and averaged per epoch.

| Term | Meaning |
|---|---|
| `total`, `task`, `wd`, `kl` | A1 terms (`kl` appears when computed) |
| `cpc`, `stab`, `hid` | the auxiliary terms **before** their λ |
| `cpc_top1`, `cpc_pos_sim`, `cpc_neg_sim`, `cpc_n_anchors`, `cpc_valid` | contrastive diagnostics (top-1 accuracy of the positive, similarities, counted anchors) |
| `stab_anch`, `anch_dist`, `anch_ratio` | anchor loss, raw distance, relative distance |
| `stab_hep`, `hep_nudge_norm`, `hep_nudge_ratio` | HEP loss and the size of the nudge |
| `stab_smooth`, `smooth_res`, `smooth_ratio` | residual-smoothness loss, its numerator and the ratio to the input's jitter |
| `stab_step`, `step_sq` | step loss and its unnormalised value (0 in A2) |
| `hid_mse`, `hid_rel`, `hid_copy_rel` | hidden prediction: squared error, relative to the target variance, and the same for the copy baseline |
| `res_smooth`, `input_smooth`, `res_vs_input`, `res_smooth_l<l>`, `step_*` | stability monitor (only with `stability.mode: diagnostic`; `step_*` are 0 in A2) |

## Internal metrics

Added to [`metrics/representation.py`](../../metrics/representation.py) (pure functions, tested):

| Function | Computes |
|---|---|
| `residual_norm_ratio(h, h_corrected, mask=None)` | mean `‖R‖ / ‖h‖` over tokens |
| `direction_redundancy(vectors)` | mean off-diagonal `|cos|` among the rows |
| `residual_redundancy(A, B)` | `{read, write}`: redundancy of the columns of `A` and the rows of `B` |
| `gradient_conflict(g1, g2)` | `[−cos(g1, g2)]₊` of two gradient PyTrees |

The post-hoc script that applies them to saved checkpoints is not written yet.

## The audit (checkpoints you can inspect)

`a2_audit(substrate, adapter, config, input_ids)` runs the checkpoints below once and returns a list of `Check(name, ok, value, detail)`; `format_audit` prints it, `audit_to_dict` makes it JSON, `audit_ok` is `False` if any check failed. It uses a fresh parameter tree and a **probe** (the same tree with a random `B`, scale 0.05), because at init `B = 0`, so `A` gets no gradient and the anchor is 0.

| Checkpoint | Pass means |
|---|---|
| `param_groups` | the tree has the cpc heads / trainable highway / hidden predictors exactly when the config asks for them |
| `frozen_before`, `frozen_after` | the base model's weights are unchanged |
| `init_equals_base` | adapted logits equal base logits at init |
| `finite_init`, `finite_probe` | every logged term is finite (the value column lists them all) |
| `total_decomposition` | `total` = task + λ_kl·kl + λ_wd·wd + λ_cp·cpc + λ_stab·stab + λ_hid·hid |
| `a1_parity` | with the auxiliary weights at 0, total and adapter gradient equal `a1_objective`'s |
| `monitor_measurement_only` | (monitor on) switching the monitor on changes neither the total nor any gradient |
| `grad_routing_task / cpc / hid / stab_anch / stab_smooth / stab_step / stab_hep` | each term's gradient reaches exactly the intended groups: task → adapter; cpc → adapter and cpc heads; hid → predictors (and the adapter when `detach_input: false`); anchor and smoothness → adapter; step → nothing in A2; HEP → highway only |
| `jit_equals_eager` | the jitted objective equals the eager one |
| `finite_with_ignored_labels` | terms stay finite when half the labels are `-100` |
| `grad_ratio_cpc`, `grad_ratio_stab_anch`, `grad_ratio_stab_smooth`, `grad_ratio_hid` | **information, no pass/fail**: `λ · ‖∇_adapter term‖ / ‖∇_adapter task‖`; the plan aims at about 0.1 to 1 per auxiliary term, so use it to choose the λ's (`hid` only when coupled) |
| `hid_vs_copy` | **information**: `hid_rel` against `hid_copy_rel` for the untrained predictor |

From the runner:

```bash
# Audit only: prints the checkpoints as JSON, exit code 1 if one fails, no training
python experiments/run_stage_a2.py --config configs/stage_A/a2_pc_inspired.yaml --audit-only

# Audit on the first training batch, then train; writes runs/<name>/audit.json,
# and does not train if a check fails
python experiments/run_stage_a2.py --arm full --seed 0 --audit
```

The audit uses the first `batch_size` blocks of the first domain, so it needs the data (and the model) to be loaded.

## Code layout

| File | Purpose |
|---|---|
| [`stages/stage_A/a2.py`](a2.py) | **Everything stage-level**: configs, parameter tree, `cpc_term`, `stability_term`, `hid_term`, `a2_objective`, arms, audit |
| [`residual/a2_losses.py`](../../residual/a2_losses.py) | Pure losses: `contrastive_future_loss`, `anchor_loss`, `output_error`, `hep_nudge_loss`, `hidden_prediction_loss` |
| [`residual/local_predictor.py`](../../residual/local_predictor.py) | Hidden predictors `P_l` (FabricPC graph) and their pure-JAX reference |

| [`residual/cpc_heads.py`](../../residual/cpc_heads.py) | CPC heads `q` and `A_{l,k}` (FabricPC nodes) and their reference |

| [`residual/fpc_energy.py`](../../residual/fpc_energy.py) | FabricPC energies for the anchor and the HEP norm |
| [`metrics/representation.py`](../../metrics/representation.py) | Internal metrics (above) |
| [`experiments/run_stage_a2.py`](../../experiments/run_stage_a2.py) | CLI runner (`--arm`, `--seed`, `--lambda-*`) |
| [`configs/stage_A/a2_pc_inspired.yaml`](../../configs/stage_A/a2_pc_inspired.yaml) | Run configuration |

`a2.py` is ordered: 1 configuration, 2 parameters, 3 terms, 4 objective, 5 arms, 6 audit. Each term is a pure function of dictionaries of hidden states `{layer: [B, T, d]}` and never touches the substrate.

## Tests and guarantees

```bash
pytest tests/residual/ -v
```

The A2 tests run on tiny GPT-2 and GPT-NeoX fixtures on CPU: nothing here has been run on a real checkpoint or a GPU (use `--audit-only` for that, see above).

| File | Covers |
|---|---|
| `stage_A/test_a2_objective.py` | `A2Config` and the YAML, the parameter tree, A1 parity, term weighting, gradient audit, padding, jit vs eager, arms, `run_sequential`, the runner end to end |
| `stage_A/test_a2_audit.py` | the audit passes on a correct objective, catches a wrong total, a gradient leak and a NaN, and the runner flags |
| `stage_A/test_a2_cpc.py`, `test_a2_stability.py` | the two terms through the frozen model (GPT-2 and NeoX), configs, reduction over sites |
| `stage_A/test_a2_smoothness_step.py` | residual smoothness and step losses: numpy reference, both backends, masks, scale invariance, gradient routing, arms, audit with all four stability losses |
| `stage_A/test_a2_hidden.py` | flat hidden-prediction config and `stability`, predictor parameters, `hid_term`, the target (next block of the adapted pass), gradient routing for both `detach_input` settings against an independent reference, the monitor never changing a gradient, training, the new arms, the audit |
| `test_contrastive_loss.py`, `test_contrastive_mask.py` | InfoNCE, anchor, output error, HEP nudge, padding mask, backend switch |
| `test_cpc_heads*.py`, `test_stability_losses.py`, `test_stability_fabricpc.py`, `test_hep.py` | heads, energies and highways; FabricPC against the plain-JAX reference |
| `test_local_predictor.py` | predictor sites, parameter layout, FabricPC graph = pure-JAX reference, token-wise application, `detach_input` and `predict_delta` |
| `test_hidden_loss.py` | `hidden_prediction_loss`: values against NumPy, normalisation, `skip_first_token`, the mask, stop-gradients |
| `test_stability_monitor.py` | residual smoothness, step metrics, the monitor over sites, no gradient |

Guarantees enforced by tests:

- **A1 parity**: with the auxiliary weights at 0, total, terms and gradients equal `a1_objective`'s, and a short sequential run matches A1's.
- **Gradient audit**: `L_hep-nudge` reaches only the highway; the anchor and `L_cpc` reach the adapter; each term enters the gradient linearly in its weight; the weight decay covers the adapter only.
- **FabricPC = reference**: values, gradients and `jit` agree to about 1e-5; spy tests prove the `fabricpc` setting really runs through FabricPC nodes. The predictor graph equals the pure-JAX `predict_reference` to 1e-5, on FabricPC 0.3.0 and 0.6.0.
- **Hidden prediction**: the target is the next block's output of the adapted pass (independent path in the test); with `detach_input: true` the adapter gets exactly zero gradient from `L_hid`, with `false` it gets the gradient of a reference with a stopped target.
- **Monitor**: switching it on changes neither the total nor any gradient.
- **Padding**: padded tokens have no influence on the masked terms and get no gradient.
- **Frozen base**: base weights are unchanged after every objective call and run.
- **Mutation checks** (run by hand, not in CI): planted bugs in the objective, the terms and the heads were caught, apart from changes that alter nothing observable.

## Design notes and caveats

- **No settling.** The nodes are used as differentiable layers; L_step and the settled representation are for a later stage.
- **Mask source.** The A1 objective receives no attention mask, so the padding mask is `labels != -100`. Tokens whose label is ignored for the task loss are also left out of the contrastive and anchor terms. With no `labels`, every token counts.
- **Where `h^base` comes from.** The adapted and base runs are separate forwards; the base states are the output of the unmodified model, not of the adapted model's earlier layers.
- **Logged names.** The shared loop prefixes every term with `loss_`, so diagnostics appear as `loss_cpc_top1`.
- **Cost not measured.** FabricPC builds a small graph state on every head call; it has not been benchmarked against a plain matrix product, and `mean_step_time` per arm has not been recorded. `L_hid` adds one `[B·T, d] → ff_dim → d` MLP per site and intercepts the block after each site.
- **Hyperparameters are uncalibrated.** A calibration of each term's gradient norm against the task gradient is planned, not written.



