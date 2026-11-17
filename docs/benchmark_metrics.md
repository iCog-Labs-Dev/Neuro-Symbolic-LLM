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
| Internal | Task-gradient conflict | Shared gradient-conflict formula on gradients of mean held-out CE at the final checkpoint. Needs batches from at least two tasks. |
| Internal | Invariant redundancy | Shared formula applied to final adapter A/B matrices; aggregated across layers. |
| Internal | Ablation prediction MAE | N/A: no validated component-effect predictor with paired measured held-out ablations. |
| Internal | Ablation rank correlation (Spearman) | N/A for the same reason; effect magnitude alone is not prediction fidelity. |
| External | Validation loss | Mean final validation CE across tasks from runner results. |
| External | Perplexity | Exponential of mean final validation loss. |
| External | Held-out test loss | Mean task test CE from runner results; test split is distinct from validation. |
| External | Held-out test perplexity | Exponential of mean task test CE. |
| External | Validation-loss forgetting | Shared runner's final mean validation-loss forgetting. |
| External | On-domain KL drift | Final checkpoint versus frozen base on each task's held-out test batches. |
| External | Token accuracy | Correct next-token predictions using the shared masked-token formula on test batches. |
| External | Calibration ECE | Shared calibration bins pooled over test batches within each task. |

Missing measurements are N/A with reasons. Checkpoint metrics are aggregated
across task values in report tables. Token accuracy is a language-model measure,
not classification or story-ending task accuracy. Reports also show the token
accuracy breakdown for each task. Reports compare A1/A2/A3 only
on matching conditions and data fingerprints. A3 K selection uses validation
loss on common seeds/data across K; test values do not select K. Operational
scores are excluded; `params_unchanged` remains a run-validity field.

Smoke and headline profiles have one rank (16), so compressibility is explicitly
N/A. Pass `--ranks 4 16` or a wider sweep to measure it. The full profile uses
ranks 1, 2, 4, 8, 16, 32, 64, but also requests named heldout and commutator
protocols: supply their names or use `--protocols sequential joint`. Rank
comparisons match method, seed, K, protocol, realized order, held-out set and
runtime data fingerprint.

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

Planning and contract tests cover stories and separate sources. A real local
smoke attempt cannot reach data loading because the Python environment lacks
`optax`. Successful model training and ROCStories numeric results are therefore
not claimed. Install the project training dependencies and run both source smoke
profiles on the intended training host before accepting runtime performance.
