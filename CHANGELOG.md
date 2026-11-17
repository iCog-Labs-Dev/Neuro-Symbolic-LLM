# Unreleased

- Fix ROCStories checkpoint evaluation to respect dataset-level test keys, and
  use distinct task training probes for diagnostic gradient conflict.
- Measure held-out rank-channel ablations against quadratic gate/Fisher
  predictions; report actual MAE/Spearman with explicit undefined-score reasons.
- Replace benchmark invariant redundancy with read/write residual direction
  redundancy from the existing shared formula. Training and loader behavior are
  unchanged. Existing checkpoints can be re-evaluated without retraining.

- Remove the JupyterHub notebook from version control; keep local notebook copies ignored.

- Split Stage A benchmark orchestration, summaries and plotting into independent
  module CLIs; default orchestration runs planning, training and evaluation.
- Use YAML runtime source and loader-realized task order for dataset-independent
  runs, checkpoint evaluation, conditions and resumable identities.
- Limit reported metrics to the requested five internal and eight external
  metrics; explain unavailable ablation prediction fidelity and rank sweeps.
- Track reports under `docs/stage_A_results`, exclude them from source
  fingerprints, and add a fail-fast full-pipeline shell script and notebook.
- Local stories/separate plans and orchestration tests run; real training smoke
  attempts stop at the environment's missing `optax` dependency. No benchmark
  measurements are asserted or committed.
