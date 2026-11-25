# Stage A benchmark metrics

Run the complete pipeline from the repository:

```bash
bash scripts/run_benchmark_full.sh --profile headline
```

Or run its independent steps:

```bash
python -m benchmark.benchmark run --output-dir runs/benchmark
python -m benchmark.summary --output-dir runs/benchmark --report-dir docs/stage_A_results
python -m benchmark.visualization --output-dir runs/benchmark --report-dir docs/stage_A_results
```

The benchmark defaults to `run`: plan if no manifest exists, train/resume, then
evaluate. `plan`, `train` and `eval` are standalone commands. Summary and
visualization consume saved artifacts without retraining. Use module invocation,
not direct execution of the Python files.

## Data path audit

The old benchmark used both paths: `_command`/`train_manifest` launched the
three `experiments/run_stage_a*.py` runners as subprocesses;
`_load_deployed_metrics` independently imported `load_runtime_data` to rebuild
checkpoint evaluation data. The runners use the shared loader, training loop
and stage evaluation. This split is retained.

The benchmark no longer invents domains or requires LexGLUE names.
`--runtime-config` selects YAML (default `configs/runtime_datasets.yaml`), and
`--runtime-source` optionally overrides `runtime.source`. The shipped YAML
defaults to ROCStories (`stories`); `--runtime-source separate` selects its
existing LexGLUE configuration. Task discovery and name validation belong to
the loader. `--task-order` and `--heldout-task` are forwarded only when supplied.
Config-defined ordering and held-out settings remain available to the loader.

Planning uses a neutral source tag when no order is supplied. Completion reads
`meta.json` → `runtime_data.task_order` into the manifest and condition, and adds
a deterministic task-order digest to the public run ID. `planned_run_id` and
`requested_task_order` retain original identity for paths and resume. Evaluation
replays the materialized runtime YAML and realized order. The `heldout` protocol
requires explicit `--heldout-task`; `commutator` requires explicit `--task-order`.
No names are guessed, including in stories mode.

## Exactly 13 reported metrics

No benchmark scores are embedded here. Evidence descriptions below specify
implemented measurements, not claims that a model run has succeeded.

| Group | Metric | Evidence and availability on ROCStories |
|---|---|---|
| Internal | Residual compressibility | Shared guarded ratio of validation-loss improvements across matched ranks. N/A for a single rank or nonpositive/too-small reference improvement. |
| Internal | Task-gradient conflict | Shared gradient-conflict formula on mean CE gradients from fixed, task-specific training batches at the final checkpoint. Diagnostic only; no updates. Requires at least two tasks. |
| Internal | Residual direction redundancy | Mean absolute pairwise cosine of distinct columns of A (read directions) and rows of B (write directions), using the shared residual_redundancy formula. Report averages the read/write values across layers; evaluation.json retains both. |
| Internal | Ablation prediction MAE | MAE between quadratic gate-space predictions and actual test-loss changes after zeroing sampled rank-channel write rows. |
| Internal | Ablation rank correlation (Spearman) | Shared tie-aware Spearman on the same paired predictions/effects. N/A only when fewer than two pairs exist or either vector is constant (or evaluation fails). |
| External | Validation loss | Mean final validation CE across tasks from runner results. |
| External | Perplexity | Exponential of mean final validation loss. |
| External | Held-out test loss | Mean task test CE from runner results; test split is distinct from validation. |
| External | Held-out test perplexity | Exponential of mean task test CE. |
| External | Validation-loss forgetting | Shared runner's final mean validation-loss forgetting. |
| External | On-domain KL drift | Final checkpoint versus frozen base on loader-provided held-out test batches. |
| External | Token accuracy | Correct next-token predictions using the shared masked-token formula on test batches. |
| External | Calibration ECE | Shared calibration bins pooled over test batches within each dataset. |

Missing measurements are N/A with reasons. Checkpoint metrics are aggregated
across dataset values in report tables. Token accuracy is a language-model measure,
not classification or story-ending task accuracy. Reports also show the token
accuracy breakdown for each dataset. Reports compare A1/A2/A3 only
on matching conditions and data fingerprints. A3 K selection uses validation
loss on common seeds/data across K; test values do not select K. Operational
scores are excluded; `params_unchanged` remains a run-validity field.

ROCStories supplies `rocstories_task_1..N` training chunks but one shared test
split keyed by the configured dataset name (normally `rocstories`). Evaluation
iterates the loader's test dictionary, not the task list. It does not duplicate
the same test set under each task name. Task-gradient conflict uses each task's
own training blocks because stories validation/test data are shared; these
diagnostic gradients must not be interpreted as held-out generalization evidence.

### Ablation probe and research-plan alignment

The probe follows the quadratic approximation in research plan v8 section 7.8
and Appendix B. Each selected residual rank channel has a scalar gate on its
row of B. All gates start at 1; an ablation changes one to 0, so its perturbation
is -1. Predictions are `-g_i + F_ii / 2`, where `g_i` is the held-out mean CE
gate gradient and `F_ii` is the mean squared **per-sequence** CE gate gradient.
Both are weighted by valid next-token counts. This is a diagonal empirical
Fisher in gate coordinates, a sequence-level curvature approximation, not the
exact hidden-state Hessian or token-level Fisher. Prediction quality is measured,
not assumed; negative correlation or large errors are retained.

Up to 16 evenly spaced channels across sorted layers are selected without
consulting outcomes. Actual signed effects are `CE(ablated) - CE(full)` on the
same fixed test sequences. No ablation outcomes are used to fit the predictor.
Evaluation saves gradients, Fisher diagonals, predictions, measured effects,
baseline/ablated losses, dataset names and sample counts under
`deployed_evaluation.ablation_fidelity`. MAE and Spearman pool component/dataset
pairs equally. A3 uses deployed K=0 behavior, as for all checkpoint metrics.
This adds reverse passes per test sequence and forward passes per selected
channel; it does not retrain or alter the checkpoint.

Residual direction redundancy matches sections 2.5 and 7.9: overlap of the
actual read/write directions. It replaces invariant redundancy in benchmark
tables and figures; the shared metric formulas remain unchanged. Previously
saved invariant values are not reinterpreted as direction redundancy.

### Refresh an existing run without retraining

On the original training host, update `benchmark-final`, retain the original
run directory with its manifest, materialized configs, metadata and checkpoints,
then run the following with `RUNS_DIR` set to that directory:

```bash
python -m benchmark.benchmark eval --output-dir "$RUNS_DIR"
python -m benchmark.summary --output-dir "$RUNS_DIR" --report-dir docs/stage_A_results
python -m benchmark.visualization --output-dir "$RUNS_DIR" --report-dir docs/stage_A_results
```

The general report is generated at
`docs/stage_A_results/stage_A/comparison/stage_a_general_report.md`.
An uploaded Markdown report by itself cannot reconstruct checkpoint metrics.
Do not run `plan` or `train` to refresh completed runs. Summary/visualization
alone also cannot repair an old evaluation.json; run checkpoint `eval` first.

Smoke, headline and full profiles default to ranks `(4, 8, 16)` through
`DEFAULT_RANKS`. Use `--ranks` to override the sweep. Compressibility requires
at least two matched ranks and a sufficiently positive reference improvement;
otherwise it remains N/A. The full profile also requests named heldout and
commutator protocols: supply their task names or use
`--protocols sequential joint`. Rank comparisons match method, seed, K,
protocol, realized order, held-out set and runtime data fingerprint.

## Files and output ownership

- [benchmark.py](../benchmark/benchmark.py): planning, orchestration, evaluation,
  immutable artifact validation and shared pure helpers. Heavy imports are local.
- [summary.py](../benchmark/summary.py): Markdown/JSON tables, stage reports and
  dashboard. No matplotlib import; figure links require existing files.
- [visualization.py](../benchmark/visualization.py): matplotlib and PNG generation.
  Refreshes summary links after figures exist. Missing matplotlib produces an
  actionable error only when figures are requested.

Raw artifacts remain in `runs/benchmark`. Reports go to `docs/stage_A_results`,
resolved against the repository root regardless of current directory. Both
reporting CLIs accept `--report-dir`. Layout: `index.html`, `REPORT.md`,
`metrics.json`, `stage_A/A1`, `A2`, `A3`, `comparison`, and figure folders.
Matching files are overwritten; unrelated files are never removed.

`docs/stage_A_results` is deliberately tracked, not ignored. It is excluded from
both tracked diffs and untracked source fingerprints, so regenerating reports
does not invalidate resume. Source changes still invalidate provenance. Custom
report directories are not automatically excluded; use the default location
for this guarantee.

The wrapper supports `PROFILE`, `OUTPUT_DIR`, `REPORT_DIR`, `RUNTIME_CONFIG`,
`RUNTIME_SOURCE`, corresponding lowercase flags, and extra benchmark arguments
after `--`. Steps fail fast. JupyterHub notebooks are local-only; use the module
commands or shell wrapper above to run the pipeline from a fresh checkout.

## Validation limits

Planning and contract tests cover stories and separate sources. These checks
do not establish successful model training or ROCStories numeric performance.
Validate both runtime sources on the intended training host before drawing
performance conclusions.
