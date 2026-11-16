# A1–A3 metrics, visualizations and run configurations

This report compares continuous residual methods A1, A2 and A3 on a frozen
language model. It excludes A4 symbolic-head/retrieval/bank metrics and all
separately supplied factor, category, intervention and probe audits.

There are **5 internal metrics**, **13 external metrics**, and separate
operational measurements. These are the only metrics included in the main
comparison tables, individual-method metric tables and final rankings.
Measured values are plotted; missing prerequisites produce N/A, never zero.

## Internal metrics

| Metric | What it measures | Required configuration |
|---|---|---|
| Residual compressibility | Retained loss improvement at a smaller adapter rank relative to the highest-rank reference | A matched rank sweep, e.g. ranks 4 and 16. The reference must improve reliably on the frozen base. A single rank does not establish a useful compression curve. |
| Effective rank | Spectral effective rank of each learned residual operator | Any trained checkpoint; deployed representation evaluation must succeed. This is descriptive: larger is not automatically better. |
| Invariant redundancy | Redundancy of the residual factors under the implementation's invariant estimator | Any trained checkpoint; deployed representation evaluation must succeed. |
| Task-gradient conflict | Negative cosine interaction between gradients for the dataset domains | Multiple domains evaluated at a common checkpoint on matched batches. Zero gradients do not establish agreement. |
| Parameter-order commutator proxy | Normalized parameter distance after training in opposite domain orders | `commutator` protocol, matched initialization, rank, K, optimizer steps and training tokens. This is a parameter-space proxy, not an output-space or symbolic commutator. |

## External metrics

| Metric | What it measures | Required evidence/configuration |
|---|---|---|
| Validation loss | Mean next-token validation cross-entropy across domains | Sequential or joint run with validation data. |
| Perplexity | Reported validation perplexity | Same validation evaluation. PPL is derived from cross-entropy; a mean of domain PPLs is not necessarily exp(mean domain loss). |
| Held-out test loss | Next-token loss on the separate test splits of the trained domains | Test splits and saved checkpoints. |
| Held-out test perplexity | Perplexity on those separate test splits | Same test evaluation. |
| Deployed LM loss | Next-token loss after loading the saved deployment parameters | Common deployed evaluator on matched held-out batches. |
| Deployed LM perplexity | Perplexity from that deployed evaluation | Same deployed evaluation. |
| Token accuracy | Fraction of valid next-token predictions that are correct | Token predictions and next-token labels; this is not legal-document classification accuracy. |
| Token calibration ECE | Difference between token confidence and empirical correctness | Pooled confidence/correct/count statistics on valid tokens. |
| On-domain KL drift | KL(base || adapted) on the trained domains | Frozen-base and adapted predictions on the same tokens. |
| Validation-loss forgetting | Increase in earlier-domain loss relative to the previous best after learning that domain | Sequential checkpoint-by-domain loss matrix. This is loss-based forgetting, not accuracy forgetting. |
| Held-out transfer | Base loss minus adapted loss on a separately configured unseen domain | An evaluation-only held-out domain. The current runtime config includes UnfairTOS; it is not used for training. |
| Sequential versus joint gap | Final sequential validation loss minus matched joint loss | Both protocols with the same seed, rank, K, domains, optimizer-step count and training-token budget. Budget mismatches are reported as N/A. |
| Anchor KL drift | KL(base || adapted) on fixed general-language anchor text | The configured anchor text and common next-token evaluation. No A4 head or category labels are needed. |

EURLEX, LEDGAR and SCOTUS provide real text for language-model adaptation.
Their legal classification labels are not used by this benchmark. Test loss
on those domains and transfer to an unseen domain are different measurements.

## Operational measurements and training diagnostics

The comparison visualization also reports trainable parameter count, training
tokens/second, mean training step time, inference latency and peak evaluator
memory. These are not combined into a single performance score.

Individual method reports include available training-loss curves,
checkpoint-by-domain validation loss/perplexity heatmaps, forgetting and KL
trajectories. A2 additionally plots hidden/CPC/stability auxiliary losses.
A3 plots loss/forgetting versus K and rank, step-time overhead and a PC energy
curve when that curve was recorded. Missing diagnostics are labeled explicitly.
An A3 K sweep changes training-time latent inference; deployed evaluation does
not settle latents using the test targets. K is separate from the epoch count.

## Excluded from these reports

- Representation fracture and factor overlap, including their shared A4-module formulas.
- Mixed-curvature coupling, cross-support curvature coupling and off-support leakage.
- Causal fingerprint separability/stability, category-role accuracy and support explanations.
- Category naturality/closure, ablation fidelity and the output-order commutator probe.
- Long-range multiple-choice probes, instruction-following probes, labeled-task accuracy/F1/ECE and accuracy-based forgetting.
- A4-only symbolic-head, retrieval, template-bank, column and symbolic geometry/transfer metrics.

Mixed-Hessian metrics do not intrinsically require A4, but need separately
declared measurements; they are excluded with the other supplied audits.
The A1–A3 evaluator no longer reads `measurement_inputs` audit files. Legacy
audit fields in previously saved evaluation files cannot add excluded rows.
Shared metric libraries remain available to other project code.

## Configurations that make the retained metrics available

All commands run the single [benchmark runner](../benchmark/run_benchmark.py).
Use a new output directory when changing an experiment. Reusing a directory
resumes its existing manifest; changing CLI flags does not replace that plan.
The examples explicitly use one epoch for turnaround. Use adequate data and
multiple seeds for research conclusions; document sampling and compute budgets.

| Configuration | Seeds | Ranks | Protocols | A3 K | Training runs | Purpose |
|---|---|---|---|---|---:|---|
| Quick K comparison | 0 | 16 | sequential | 1, 4, 6, 8 | 6 | Basic internal/external metrics and A3 K comparison; no useful rank curve, joint gap or order proxy. |
| Rank comparison | 0 | 4, 16 | sequential | 4 | 6 | Compressibility plus basic metrics; lower rank is compared with the rank-16 reference. |
| Sequential/joint comparison | 0 | 16 | sequential, joint | 4 | 6 | Sequential-versus-joint gap when budgets match. |
| Task-order comparison | 0 | 16 | commutator | 4 | 6 | Each method runs EURLEX→LEDGAR→SCOTUS and the reverse order. |
| Combined coverage | 0 | 4, 16 | sequential, joint, commutator | 4 | 24 | Prerequisite runs for all retained metric families. Values are still N/A if a measurement is undefined or evaluation fails. |
| Combined coverage, repeated seeds | 0, 1, 2 | 4, 16 | sequential, joint, commutator | 4 | 72 | Repeated-seed means, variation and paired comparisons. |

Quick K comparison:

```sh
python benchmark/run_benchmark.py run --profile headline --seeds 0 --ranks 16 --k 1 4 6 8 --protocols sequential --epochs 1 --report-rank 16 --output-dir runs/a1-a3-quick
```

Combined coverage with one fixed K to keep the experiment count smaller:

```sh
python benchmark/run_benchmark.py run --profile headline --seeds 0 --ranks 4 16 --k 4 --protocols sequential joint commutator --epochs 1 --report-rank 4 --output-dir runs/a1-a3-coverage
```

Use `--seeds 0 1 2` with a new output directory for the 72-run version. Using
all four K values in the combined configuration gives 48 runs for one seed,
or 144 runs for three seeds. These commands use the checked-in real-data
subsets; the separate quick notebook's smaller data overrides do not apply to
CLI commands automatically.

The default headline profile has three seeds, rank 16, K=[1,2,4,8] and the
sequential/joint protocols: 36 runs. Its fixed rank and lack of reversed task
orders do not cover every internal metric. The `full` profile adds wider
rank/protocol sweeps and can require substantial compute.

## Reports, existing results and JupyterHub

To regenerate the narrower report from saved evaluation, **without retraining**:

```sh
python benchmark/run_benchmark.py report --output-dir runs/your-existing-run --report-rank 16
```

To recompute deployed measurements from saved checkpoints and then regenerate
reports, use `eval` instead. Source/checkpoint integrity checks still apply.

The HTML overview is generated at `report_and_visualization/index.html`.
The Markdown comparison, individual A1/A2/A3 reports and PNGs are under
`report_and_visualization/stage_A/`. Tables show metric availability and the
summary reports the best observed value for each applicable metric on matched
seeds. It does not assert statistical significance or select a universal winner.
Existing unreferenced image files are not deleted; notebook galleries display
only images linked by the current report so excluded figures cannot reappear.

The separately supplied JupyterHub notebook clones the
branch or a selected PR, installs dependencies on the server, checks GPU and
training imports, runs the pipeline, displays tables/figures and exports a ZIP.
The downloadable small-data notebook uses all three domains, one seed, one
epoch and six runs. It can show the basic metrics, but not a joint gap or
alternate-order proxy, and its one-rank setting does not establish a compression
curve. No dataset or seed setting guarantees a fixed completion time.
