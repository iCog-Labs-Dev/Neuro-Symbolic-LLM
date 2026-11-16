# Neuro-Symbolic LLM

The Stage A benchmark compares A1, A2 and A3 residual training on a frozen LLM.

## Run on JupyterHub

Upload [run_benchmark_jupyterhub.ipynb](run_benchmark_jupyterhub.ipynb), select a
Python 3.11 or 3.12 kernel on an NVIDIA GPU server, and run the cells in order.
The notebook clones `benchmark-final` or a selected PR, installs dependencies
on JupyterHub, trains, evaluates, displays internal/external metric tables and
all plots, and creates a downloadable results ZIP.

## One benchmark file, one test file

With dependencies installed in your execution environment:

```sh
python benchmark/run_benchmark.py --profile headline
python -m pytest tests/benchmark/test_benchmark.py -q
```

[benchmark/run_benchmark.py](benchmark/run_benchmark.py) performs planning,
training, evaluation and visualization in one command. The browser report is
generated at `report_and_visualization/index.html`; the notebook displays
the tables and images inline. To regenerate visuals without retraining, use:

```sh
python benchmark/run_benchmark.py report --output-dir runs/benchmark
```

See [benchmark metrics and evidence requirements](docs/benchmark_metrics.md)
for profiles, measurement definitions, missing metrics and audit inputs.
The default headline profile trains on real LexGLUE subsets with three seeds
and the configured training epochs. Model training runs in the selected environment and requires
model/dataset downloads and sufficient compute.
