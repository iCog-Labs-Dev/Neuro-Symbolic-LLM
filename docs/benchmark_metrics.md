# A1–A3 benchmark and research measurements

This benchmark compares continuous residuals on the frozen base model. The
research plan sections 13–14 and the accompanying metrics document distinguish
language-model performance, internal organization, and the later symbolic
head. Symbolic-head, bank and column metrics are outside this A1–A3 report.

Install the visualization extra and run from the repository root:

```sh
pip install -e '.[visualization]'
python -m experiments.run_benchmark plan --profile full --output-dir runs/benchmark
python -m experiments.run_benchmark train --output-dir runs/benchmark
python -m experiments.run_benchmark eval --output-dir runs/benchmark
python -m experiments.run_benchmark report --output-dir runs/benchmark
```

`report_and_visualization/stage_A/` contains `A1/`, `A2/`, `A3/`, and
`comparison/`. Each has a Markdown report, internal and external tables and
PNG figures. Missing measurements are explicitly N/A. Re-running reports
overwrites generated files with matching names but does not delete other files.
Visualization dependencies are required unless `--no-visualization` is set.

The full profile trains many models and requires substantial compute and dataset
access. Use `--profile smoke` to validate the pipeline, not as scientific
evidence. Results depend on saved checkpoints and actual measurements; example
data or placeholder images are not benchmark results.

In the current A3 implementation, K controls latent settling during training;
deployed evaluation uses the learned latent predictor without target-conditioned
settling. A K sweep therefore compares training procedures and their deployed
checkpoints, not K rounds of online inference on test targets.

## Research metrics and evidence

| Metric | Measurement required |
|---|---|
| Cross-entropy, perplexity, token accuracy, token ECE | Matched next-token evaluation tokens. PPL = exp(mean token loss); ECE pools confidence/correct/count statistics. |
| Task accuracy, macro/micro F1, task ECE | Held-out labeled predictions from a specified task protocol. Multi-label exact match is not single-label accuracy; report the official task metric. |
| Accuracy forgetting | A checkpoint-by-task accuracy/score matrix. At checkpoint t use mean of the previous best post-acquisition score minus the current score. Loss-based forgetting remains separately labeled. |
| Held-out transfer | Separate held-out domain; improvement is base loss minus adapted loss. |
| Sequential versus joint | Same initial conditions and matched optimizer steps and token budgets. |
| Anchor KL | Fixed anchor corpus; KL(base || adapted) averaged over valid next-token positions. |
| Instruction following, long-range coherence | Independently defined and held-out scored examples; the supplied tiny long-range probes are smoke probes, not a validated benchmark. |
| Compressibility | Matched rank sweep; ratio of improvement relative to a reliably positive highest-rank reference, by domain and seed. |
| Fracture and factor overlap | Nonnegative component importance for *declared* semantic/functional factors. Fracture is entropy effective support divided by component count; overlap is sum of coordinate-wise minima. Negative ablation effects require a declared estimator; they are not silently clipped. |
| Gradient conflict | Matched task gradients, negative part of cosine similarity. Zero gradients do not establish agreement. |
| Mixed curvature | A declared mixed Hessian block; report its Frobenius norm and the block/scale used. |
| Off-support leakage | Per-component gradient magnitudes and independently declared intended support. A dense adapter does not imply a meaningful task support. |
| Cross-support coupling | Maximum Frobenius norm of declared Hessian blocks crossing an independently defined support boundary. |
| Update commutator | Matched alternate task orders, initialization and training budgets; the output-space metric needs paired outputs on identical probes. Parameter and linear-product proxies are not the same as output-space distance. |
| Causal fingerprint separability | Component intervention effects and independently specified component groups; between/within sum of squares. No score is reported for zero within-group variance. |
| Fingerprint stability, role and support explanations | Matched component fingerprints across seeds or perturbations; independently labeled category roles and support explanations. Component alignment must be documented before calculating correlation. |
| Naturality and closure | Paired transformed representations from a declared category probe and transformation; report mean squared Euclidean discrepancy. |
| Ablation fidelity | Predicted and measured held-out intervention effects; report MAE and Spearman correlation. |

PR #20's factor-organization implementation also tracks factors with no
measured contribution. These factors remain undefined and are listed by index;
they are excluded from means and overlap rather than being assigned a score.
Fracture and overlap reuse that implementation from `metrics.a4.representation`.
Until it is present in the checkout (for example, after the Base branch changes
are integrated), those two measurements are reported as N/A, even if factor
importance evidence is supplied. Other audit measurements remain available.
The other metrics introduced by that PR concern symbolic-head retrieval,
template geometry and CPU/GPU transfer, outside this report's scope.

Additional probes are not inferred from legal-domain names: a domain is not
automatically a semantic factor, and next-token accuracy is not legal task
classification accuracy. Fingerprint group labels, causal supports, category
transformations, intervention measurements, instruction scores and task labels
must come from a separately defined and reproducible measurement protocol.

## Supplying additional audit measurements

For a completed run, place a JSON file at
`runs/benchmark/measurement_inputs/<run_id>.json`, then rerun `eval` and
`report`. This file is outside immutable run artifacts. It must contain:

```json
{
  "schema": "a1-a3-research-audit/v1",
  "run_id": "the-exact-run-id",
  "checkpoint_sha256": "sha256-of-the-final-task-checkpoint",
  "protocol_sha256": "sha256-of-the-common-probe-and-evaluation-protocol",
  "factor_importance": [[1.0, 0.0], [0.2, 0.8]],
  "mixed_hessian": [[0.1, 0.0], [0.0, 0.2]],
  "cross_support_hessians": [[[0.1]], [[0.2]]],
  "component_gradients": [0.4, 0.1],
  "support": [true, false],
  "fingerprints": [[0.3, 0.1], [0.2, 0.1], [0.0, 0.4], [0.0, 0.3]],
  "fingerprint_groups": ["a", "a", "b", "b"],
  "fingerprint_stability": {"first": [[0.1, 0.2]], "second": [[0.2, 0.4]]},
  "category_roles": {"targets": [0, 1], "predictions": [0, 1]},
  "support_explanations": {"targets": [0, 1], "predictions": [0, 1]},
  "naturality": {"left": [[1.0, 0.0]], "right": [[0.9, 0.1]]},
  "closure": {"left": [[1.0, 0.0]], "right": [[1.0, 0.0]]},
  "ablation": {"predicted": [0.2, 0.1], "actual": [0.18, 0.12]},
  "ordered_outputs": {"first": [[0.4, 0.6]], "reverse": [[0.3, 0.7]]},
  "accuracy_matrix": [[0.9, null], [0.8, 0.85]],
  "classification": {"targets": [0, 1], "predictions": [0, 1], "confidence": [0.8, 0.9]},
  "instruction_scores": [1.0, 0.0]
}
```

The example values are illustrative only. Each measurement field is optional;
the schema, run ID, final checkpoint hash and protocol hash are required. The
protocol hash should cover the probe examples, splits, labels, component
definitions and scoring/normalization policy. All three methods must use the
same hash to appear in a common audit comparison. For multi-label task
classification use equal binary target and prediction matrices and
`"multilabel": true`; calibration needs a separately defined binary protocol.
The evaluator validates the checkpoint binding and hashes the input file. It
cannot establish that supplied labels, probes, or interventions are scientifically
valid; retain the underlying evidence and report its provenance.
