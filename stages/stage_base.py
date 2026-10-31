"""Shared stage machinery: configs, data loading, the sequential loop and summaries.

A stage plugs in its objective config class (``load_configs``) and its objective
function (``run_sequential``); stage-specific code lives in ``stages/stage_<X>/``.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

try:
    from datasets import load_dataset
except (ImportError, RuntimeError):
    load_dataset = None

from frozenllm.substrate import FrozenSubstrate, identity_modify
from metrics.performance import (
    mean_forgetting,
    mean_seen_accuracy,
    perplexity,
    training_step_time,
)
from metrics.representation import loss_improvement
from residual import AdapterParams, ResidualAdapter, ResidualConfig
from stages.results import RunWriter, load_params

SECTIONS = ("adapter", "objective", "training", "domains")
# Domain names become file names (params/task_<t>_<name>.npz).
DOMAIN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

log = logging.getLogger(__name__)


def checked_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {unknown}. Allowed: {sorted(known)}."
        )
    return dict(data)


class Objective(Protocol):
    """Stage objective: differentiates ``params``, returns ``(total, terms)``."""

    # ``terms`` must hold "total" and "task"; extra terms are logged as-is.

    def __call__(
        self,
        params: AdapterParams,
        *,
        substrate: FrozenSubstrate,
        adapter: ResidualAdapter,
        config: Any,
        input_ids: jax.Array,
        labels: jax.Array | None = None,
        base: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]: ...


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "gpt2"
    seq_len: int = 128
    batch_size: int = 8
    # Default epochs per domain; a domain's ``epochs`` overrides it.
    epochs: int = 3
    learning_rate: float = 1.0e-3
    eval_batches: int = 8
    log_every: int = 10
    seed: int = 0
    # Fresh Adam moments at each domain boundary.
    reset_optimizer: bool = True
    output_dir: str = "runs"
    jit: bool = True

    def __post_init__(self) -> None:
        for name in ("seq_len", "batch_size", "eval_batches", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2 for next-token training.")
        if not math.isfinite(self.learning_rate) or not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not isinstance(self.jit, bool):
            raise ValueError(f"jit must be a boolean, got {self.jit!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        return cls(**checked_kwargs(cls, data))


@dataclass(frozen=True)
class DomainConfig:
    name: str
    dataset: str
    dataset_config: str | None = None
    text_field: str = "text"
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    # Joins rows into one stream; use "\n\n" for one-document-per-row datasets.
    separator: str = ""
    epochs: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not DOMAIN_NAME.match(self.name):
            raise ValueError(
                f"Domain name must match {DOMAIN_NAME.pattern}, got {self.name!r}"
            )
        if self.epochs is not None and (
            not isinstance(self.epochs, int) or self.epochs < 0
        ):
            raise ValueError(f"Domain {self.name!r}: epochs must be >= 0 or null.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DomainConfig:
        return cls(**checked_kwargs(cls, data))


@dataclass(frozen=True)
class RunConfig:
    adapter: ResidualConfig
    # Stage objective config: any frozen dataclass with ``from_dict``.
    objective: Any
    training: TrainingConfig
    domains: tuple[DomainConfig, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": asdict(self.adapter),
            "objective": asdict(self.objective),
            "training": asdict(self.training),
            "domains": [asdict(d) for d in self.domains],
        }


def load_configs(path: Path, objective_cls: type[Any]) -> RunConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown config sections in {path}: {unknown}.")
    domains_raw = raw.get("domains")
    if not isinstance(domains_raw, list) or not domains_raw:
        raise ValueError(f"{path}: 'domains' must be a non-empty list of domains.")
    domains = tuple(DomainConfig.from_dict(d) for d in domains_raw)
    names = [d.name for d in domains]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: domain names must be unique, got {names}.")
    return RunConfig(
        adapter=ResidualConfig.from_dict(raw.get("adapter") or {}),
        objective=objective_cls.from_dict(raw.get("objective") or {}),
        training=TrainingConfig.from_dict(raw.get("training") or {}),
        domains=domains,
    )


@dataclass(frozen=True)
class Domain:
    name: str
    train_blocks: np.ndarray
    eval_set: tuple[jax.Array, ...]
    epochs: int


def load_token_blocks(
    substrate: FrozenSubstrate, domain: DomainConfig, split: str, seq_len: int
) -> np.ndarray:
    if load_dataset is None:
        raise ImportError("The 'datasets' package is required to load token blocks.")
    if substrate.tokenizer is None:
        raise ValueError("The substrate has no tokenizer.")
    texts = load_dataset(domain.dataset, domain.dataset_config, split=split)[
        domain.text_field
    ]
    # One contiguous token stream; the ragged tail is dropped.
    ids = substrate.tokenizer(domain.separator.join(texts), return_tensors="np")[
        "input_ids"
    ][0]
    n = len(ids) // seq_len
    if n == 0:
        raise ValueError(
            f"Domain {domain.name!r} split {split!r} is shorter than seq_len={seq_len}."
        )
    # Annotated: numpy is opaque to mypy (follow_imports = "skip").
    blocks: np.ndarray = np.asarray(ids[: n * seq_len], dtype=np.int32)
    return blocks.reshape(n, seq_len)


def make_eval_set(
    blocks: np.ndarray, batch_size: int, eval_batches: int
) -> tuple[jax.Array, ...]:
    if eval_batches < 1:
        raise ValueError(f"eval_batches must be >= 1, got {eval_batches}")
    n_eval = min(len(blocks), eval_batches * batch_size)
    # Ensure static batch shapes for JAX JIT compilation by dropping partial trailing batches.
    n_full = (n_eval // batch_size) * batch_size
    if n_full == 0 and len(blocks) > 0:
        return (jnp.asarray(blocks[:n_eval]),)
    return tuple(
        jnp.asarray(blocks[i : i + batch_size]) for i in range(0, n_full, batch_size)
    )


def batches(blocks: np.ndarray, batch_size: int, seed: int) -> Iterator[jax.Array]:
    if len(blocks) < batch_size:
        raise ValueError(f"{len(blocks)} blocks cannot fill batch_size={batch_size}.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    # The last partial batch is dropped to keep batch shapes static.
    for i in range(0, len(order) - batch_size + 1, batch_size):
        yield jnp.asarray(blocks[order[i : i + batch_size]])


def base_logits(substrate: FrozenSubstrate, input_ids: jax.Array) -> jax.Array:
    result = substrate.run_with_interception(
        input_ids, modify_fn=identity_modify, intercept_layers=()
    )
    return jax.lax.stop_gradient(result.logits)


def evaluate(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    params: AdapterParams,
    eval_batches: Sequence[jax.Array],
    *,
    jit: bool = True,
) -> dict[str, float]:
    if not eval_batches:
        return {
            "base_loss": 0.0,
            "base_ppl": 1.0,
            "loss": 0.0,
            "ppl": 1.0,
            "kl": 0.0,
        }

    def _eval_step(
        p: AdapterParams, ids: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        base = base_logits(substrate, ids)
        adapted = substrate.run_with_interception(
            ids, modify_fn=adapter.modify_fn(p), intercept_layers=adapter.layers
        ).logits
        base_loss = substrate.compute_loss(base, ids, losses=("CE",))
        loss = substrate.compute_loss(adapted, ids, losses=("CE",))
        kl = substrate.compute_loss(adapted, ids, losses=("KL",), base_logits=base)
        return base_loss, loss, kl

    eval_step = jax.jit(_eval_step) if jit else _eval_step

    base_losses, losses, kls = [], [], []
    for ids in eval_batches:
        b_loss, a_loss, kl = eval_step(params, ids)
        base_losses.append(float(b_loss))
        losses.append(float(a_loss))
        kls.append(float(kl))
    base_loss, loss = float(np.mean(base_losses)), float(np.mean(losses))
    return {
        "base_loss": base_loss,
        "base_ppl": float(perplexity(base_loss)),
        "loss": loss,
        "ppl": float(perplexity(loss)),
        "kl": float(np.mean(kls)),
    }


def resolve_split(domain: DomainConfig, split: str) -> str:
    """Map symbolic split name (test, eval/validation, train) to domain split string."""
    if split == "test":
        return domain.test_split
    if split in ("eval", "validation"):
        return domain.eval_split
    if split == "train":
        return domain.train_split
    return split


def validate_checkpoint_params(
    adapter: ResidualAdapter,
    params: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate that checkpoint parameters match the adapter layers, keys, and shapes."""
    adapter.validate_params(params)


def evaluate_checkpoint(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    checkpoint: Path | str | Mapping[str, Mapping[str, Any]],
    domains: Sequence[DomainConfig],
    *,
    split: str = "test",
    seq_len: int = 128,
    batch_size: int = 8,
    eval_batches: int = 8,
    jit: bool = True,
) -> dict[str, dict[str, float]]:
    raw: Mapping[str, Mapping[str, Any]] = (
        load_params(checkpoint) if isinstance(checkpoint, str | Path) else checkpoint
    )
    params = {
        outer: {inner: jnp.asarray(val) for inner, val in inner_dict.items()}
        for outer, inner_dict in raw.items()
    }
    validate_checkpoint_params(adapter, params)
    results = {}
    for d in domains:
        split_name = resolve_split(d, split)
        blocks = load_token_blocks(substrate, d, split_name, seq_len)
        eval_set = make_eval_set(blocks, batch_size, eval_batches)
        results[d.name] = evaluate(substrate, adapter, params, eval_set, jit=jit)
    return results


def summarize(
    names: Sequence[str],
    init: Mapping[str, Mapping[str, float]],
    rows: Sequence[Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    loss = np.array([[row[n]["loss"] for n in names] for row in rows])
    base_loss = [init[n]["base_loss"] for n in names]
    num_tasks = len(rows)
    return {
        "domains": list(names),
        "base_loss": base_loss,
        "loss": loss.tolist(),
        "ppl": [[row[n]["ppl"] for n in names] for row in rows],
        "kl": [[row[n]["kl"] for n in names] for row in rows],
        "improvement": [
            [
                float(loss_improvement(base_loss[j], loss[t, j]))
                for j in range(len(names))
            ]
            for t in range(num_tasks)
        ],
        "mean_seen_loss": [
            float(mean_seen_accuracy(loss, t)) for t in range(num_tasks)
        ],
        # Undefined at t=0 (no earlier task), stored as null.
        "mean_forgetting": [None]
        + [
            float(mean_forgetting(loss, t, higher_is_better=False))
            for t in range(1, num_tasks)
        ],
    }


def run_sequential(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    params: AdapterParams,
    domains: Sequence[Domain],
    objective_fn: Objective,
    objective_cfg: Any,
    training: TrainingConfig,
    writer: RunWriter,
    *,
    optimizer: optax.GradientTransformation | None = None,
) -> tuple[AdapterParams, dict[str, Any]]:
    if not domains:
        raise ValueError("At least one domain is required.")
    names = [d.name for d in domains]

    def eval_all(after_task: int) -> dict[str, dict[str, float]]:
        row = {}
        for d in domains:
            row[d.name] = evaluate(
                substrate, adapter, params, d.eval_set, jit=training.jit
            )
            writer.log("eval", after_task=after_task, domain=d.name, **row[d.name])
        if after_task < 0:
            log.info(
                "[Validation] Baseline zero-shot eval after_task=%d: %s",
                after_task,
                _fmt_losses(row),
            )
        else:
            log.info(
                "[Validation] eval after_task=%d (%s): %s",
                after_task,
                domains[after_task].name,
                _fmt_losses(row),
            )
        return row

    # after_task=-1: before training, where adapted equals base.
    init = eval_all(-1)

    if optimizer is None:
        optimizer = optax.adam(training.learning_rate)
    grad_fn = jax.value_and_grad(objective_fn, has_aux=True)

    def train_step(
        params: AdapterParams, opt_state: optax.OptState, ids: jax.Array
    ) -> tuple[AdapterParams, optax.OptState, dict[str, jax.Array]]:
        # No precomputed base: the objective runs its own base forward if needed.
        (_, terms), grads = grad_fn(
            params,
            substrate=substrate,
            adapter=adapter,
            config=objective_cfg,
            input_ids=ids,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, terms

    if training.jit:
        train_step = jax.jit(train_step)

    # The optimizer only sees the adapter PyTree; base weights live outside it.
    opt_state = optimizer.init(params)
    rows: list[dict[str, dict[str, float]]] = []
    step_times: list[float] = []
    global_step = tokens = 0
    cpc_skipped_batches = 0

    for t, domain in enumerate(domains):
        if t > 0 and training.reset_optimizer:
            opt_state = optimizer.init(params)

        num_epochs = domain.epochs
        if num_epochs < 0:
            raise ValueError(
                f"Domain {domain.name!r}: epochs must be >= 0, got {num_epochs}."
            )

        if num_epochs > 0 and len(domain.train_blocks) < training.batch_size:
            raise ValueError(
                f"Domain {domain.name!r} has {len(domain.train_blocks)} blocks, "
                f"which cannot fill batch_size={training.batch_size}."
            )

        batches_per_epoch = (
            len(domain.train_blocks) // training.batch_size if num_epochs > 0 else 0
        )
        total_domain_batches = batches_per_epoch * num_epochs
        log.info("=" * 78)
        log.info(
            ">>> DOMAIN [%d/%d]: '%s' | Epochs: %d | Batches/Epoch: %d (Total: %d)",
            t + 1,
            len(domains),
            domain.name,
            num_epochs,
            batches_per_epoch,
            total_domain_batches,
        )
        log.info("=" * 78)

        domain_step = 0
        for epoch in range(1, num_epochs + 1):
            log.info("  EPOCH %d/%d ['%s']", epoch, num_epochs, domain.name)
            epoch_start_time = time.time()
            epoch_step_times: list[float] = []
            epoch_scalars: list[dict[str, float]] = []
            epoch_tokens = 0
            epoch_seed = training.seed + t * 10000 + (epoch - 1)
            stream = batches(domain.train_blocks, training.batch_size, epoch_seed)

            for batch_idx, ids in enumerate(stream, start=1):
                domain_step += 1
                global_step += 1
                batch_tokens = int(ids.size)
                tokens += batch_tokens
                epoch_tokens += batch_tokens

                (params, opt_state, terms), step_time = training_step_time(
                    train_step, params, opt_state, ids
                )
                step_times.append(step_time)
                epoch_step_times.append(step_time)
                scalars = {k: float(v) for k, v in terms.items()}
                if scalars.get("cpc_valid", 1.0) == 0.0:
                    cpc_skipped_batches += 1
                    if cpc_skipped_batches == 1:
                        log.warning(
                            "CPC batch has no valid cross-sequence negatives; loss is inactive. Skipped batches are logged."
                        )
                epoch_scalars.append(scalars)

                train_writer_metrics = {f"loss_{k}": v for k, v in scalars.items()}
                if "task" in scalars or "task_loss" in scalars:
                    task_val = scalars.get("task", scalars.get("task_loss"))
                    train_writer_metrics["ppl"] = float(perplexity(task_val))

                writer.log(
                    "train",
                    task=t,
                    domain=domain.name,
                    epoch=epoch,
                    batch=batch_idx,
                    step=domain_step,
                    global_step=global_step,
                    tokens=tokens,
                    step_time=step_time,
                    cpc_skipped_batches=cpc_skipped_batches,
                    # Prefixed so the "task" term does not clash with the task index.
                    **train_writer_metrics,
                )
                is_first_batch = batch_idx == 1
                is_last_batch = batch_idx == batches_per_epoch
                is_interval = (
                    training.log_every > 0 and domain_step % training.log_every == 0
                )
                if is_first_batch or is_last_batch or is_interval:
                    terms_str = _fmt_terms(scalars)
                    if "task" in scalars or "task_loss" in scalars:
                        task_val = scalars.get("task", scalars.get("task_loss"))
                        log.info(
                            "  Batch %d/%d %s  ppl=%.2f",
                            batch_idx,
                            batches_per_epoch,
                            terms_str,
                            float(perplexity(task_val)),
                        )
                    else:
                        log.info(
                            "  Batch %d/%d %s",
                            batch_idx,
                            batches_per_epoch,
                            terms_str,
                        )

            epoch_duration = float(time.time() - epoch_start_time)
            if epoch_scalars:
                mean_terms = {
                    k: float(np.mean([s[k] for s in epoch_scalars]))
                    for k in epoch_scalars[0]
                }
                mean_batch_time = (
                    float(np.mean(epoch_step_times)) if epoch_step_times else 0.0
                )
                terms_str = _fmt_terms(mean_terms)
                epoch_writer_metrics = {f"loss_{k}": v for k, v in mean_terms.items()}
                if "task" in mean_terms or "task_loss" in mean_terms:
                    mean_task = mean_terms.get("task", mean_terms.get("task_loss"))
                    epoch_ppl = float(perplexity(mean_task))
                    epoch_writer_metrics["ppl"] = epoch_ppl
                    completion_msg = (
                        f"  Epoch {epoch}/{num_epochs} Finished | "
                        f"Duration: {epoch_duration:.2f}s (avg {mean_batch_time:.3f}s/batch) | "
                        f"{terms_str} | ppl={epoch_ppl:.2f}"
                    )
                else:
                    completion_msg = (
                        f"  Epoch {epoch}/{num_epochs} Finished | "
                        f"Duration: {epoch_duration:.2f}s (avg {mean_batch_time:.3f}s/batch) | "
                        f"{terms_str}"
                    )
                writer.log(
                    "epoch",
                    task=t,
                    domain=domain.name,
                    epoch=epoch,
                    total_epochs=num_epochs,
                    global_step=global_step,
                    batches=len(epoch_scalars),
                    blocks=len(epoch_scalars) * training.batch_size,
                    epoch_tokens=epoch_tokens,
                    tokens=tokens,
                    epoch_time=epoch_duration,
                    mean_step_time=mean_batch_time,
                    **epoch_writer_metrics,
                )
                log.info("%s", completion_msg)

        writer.save_checkpoint(
            f"task_{t}_{domain.name}",
            params,
            opt_state=opt_state,
            global_step=global_step,
            task=t,
            domain=domain.name,
            epoch=num_epochs,
            step=domain_step,
        )
        log.info(
            "Checkpoint saved: task_%d_%s (global_step=%d)",
            t,
            domain.name,
            global_step,
        )
        rows.append(eval_all(t))

    summary = summarize(names, init, rows)
    summary.update(
        epochs={d.name: d.epochs for d in domains},
        steps=global_step,
        tokens=tokens,
        cpc_skipped_batches=cpc_skipped_batches,
        # Exclude the first step, which includes tracing.
        mean_step_time=float(np.mean(step_times[1:])) if len(step_times) > 1 else None,
    )
    return params, summary


def _fmt_losses(row: Mapping[str, Mapping[str, float]]) -> str:
    return " ".join(f"{name}={r['loss']:.4f}" for name, r in row.items())


def _fmt_terms(scalars: Mapping[str, float]) -> str:
    parts: list[str] = []
    if "total" in scalars:
        parts.append(f"total_loss={scalars['total']:.4f}")
    elif "total_loss" in scalars:
        parts.append(f"total_loss={scalars['total_loss']:.4f}")

    if "task" in scalars:
        parts.append(f"task_loss={scalars['task']:.4f}")
    elif "task_loss" in scalars:
        parts.append(f"task_loss={scalars['task_loss']:.4f}")

    if not parts and "loss" in scalars:
        parts.append(f"loss={scalars['loss']:.4f}")

    known_keys = {"total", "total_loss", "task", "task_loss", "loss"}
    for k, v in scalars.items():
        if k not in known_keys:
            try:
                val = float(v)
                if k in ("acc", "accuracy"):
                    parts.append(f"{k}={val:.4f}")
                else:
                    parts.append(f"{k}={val:.3e}")
            except (ValueError, TypeError):
                parts.append(f"{k}={v}")

    return " ".join(parts)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure standard and professional ML training output logging."""
    logging.basicConfig(level=level, format="%(message)s", force=True)

    # Silence verbose 3rd-party loggers
    noisy_loggers = (
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
        "huggingface_hub",
        "transformers",
        "datasets",
        "filelock",
        "jax",
        "jax._src.lib.xla_client",
        "absl",
    )
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Environment flags to quiet HuggingFace and backend spam
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")

    # Filter benign environment warnings (e.g. CPU-only TPU missing backend notices)
    warnings.filterwarnings("ignore", message=r".*Unable to initialize backend 'tpu'.*")
    warnings.filterwarnings(
        "ignore", message=r".*unauthenticated requests to the HF Hub.*"
    )

    try:
        import datasets

        datasets.logging.set_verbosity_error()
        datasets.disable_progress_bar()
    except Exception:
        pass

    try:
        import transformers

        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    try:
        import huggingface_hub.utils.logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def log_experiment_header(
    logger: logging.Logger,
    cfg: RunConfig,
    arch: Any,
    adapter: Any,
    n_trainable: int,
    n_frozen: int,
    run_dir: Any,
    domains: Sequence[Domain],
    start_time: str | None = None,
) -> None:
    """Log structured initialization banner and experiment hyperparameters."""
    train_cfg = cfg.training
    ratio = (n_trainable / n_frozen * 100.0) if n_frozen > 0 else 0.0
    ts = start_time or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    logger.info("=" * 78)
    logger.info("STAGE A1: SEQUENTIAL RESIDUAL ADAPTER TRAINING")
    logger.info("=" * 78)
    logger.info("Started At         : %s", ts)
    logger.info(
        "Model Architecture : %s (%d layers, hidden_dim=%d)",
        train_cfg.model,
        arch.num_layers,
        arch.hidden_size,
    )
    logger.info(
        "Residual Adapter   : target_layers=%s | rank=%d | alpha=%s",
        list(adapter.layers),
        cfg.adapter.rank,
        cfg.adapter.alpha,
    )
    logger.info(
        "Parameters         : %s trainable | %s frozen (%.3f%% trainable)",
        f"{n_trainable:,}",
        f"{n_frozen:,}",
        ratio,
    )
    logger.info(
        "Hyperparameters    : batch_size=%d | seq_len=%d | lr=%.2e | jit=%s",
        train_cfg.batch_size,
        train_cfg.seq_len,
        train_cfg.learning_rate,
        train_cfg.jit,
    )
    logger.info("Run Directory      : %s", run_dir)
    logger.info("Curriculum (%d domain(s)):", len(domains))
    for i, d in enumerate(domains, 1):
        logger.info(
            "  [%d] %-10s -> %d blocks | %d eval batches | %d epoch(s)",
            i,
            d.name,
            len(d.train_blocks),
            len(d.eval_set),
            d.epochs,
        )
    logger.info("-" * 78)


def log_final_summary(
    logger: logging.Logger,
    summary: Mapping[str, Any],
    frozen: Mapping[str, Any],
    start_time: float | None = None,
) -> None:
    """Log senior ML engineer standard summary table and continual learning metrics."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    elapsed_str = ""
    if start_time is not None:
        elapsed_sec = int(time.time() - start_time)
        m, s = divmod(elapsed_sec, 60)
        h, m = divmod(m, 60)
        if h > 0:
            elapsed_str = f" (Total Duration: {h}h {m}m {s}s)"
        else:
            elapsed_str = f" (Total Duration: {m}m {s}s)"

    logger.info("=" * 78)
    logger.info("EXPERIMENT SUMMARY & METRICS")
    logger.info("=" * 78)
    logger.info("Completed At       : %s%s", ts, elapsed_str)

    test_results = summary.get("test")
    if isinstance(test_results, Mapping) and test_results:
        logger.info("Test Evaluation Across Domains:")
        logger.info(
            "  %-12s  %-12s  %-10s  %-12s  %-10s",
            "Domain",
            "Loss",
            "PPL",
            "Base Loss",
            "Base PPL",
        )
        logger.info("  " + "-" * 62)
        for dom, metrics in test_results.items():
            if isinstance(metrics, Mapping):
                logger.info(
                    "  %-12s  %-12.4f  %-10.2f  %-12.4f  %-10.2f",
                    dom,
                    metrics.get("loss", 0.0),
                    metrics.get("ppl", 0.0),
                    metrics.get("base_loss", 0.0),
                    metrics.get("base_ppl", 0.0),
                )
        logger.info("  " + "-" * 62)

    mean_seen = summary.get("mean_seen_loss", [])
    forgetting = summary.get("mean_forgetting", [])
    logger.info("Continual Learning Metrics:")
    if mean_seen:
        seen_str = ", ".join(f"{v:.4f}" for v in mean_seen)
        logger.info("  • Mean Seen Loss : [%s] (Final: %.4f)", seen_str, mean_seen[-1])
    if forgetting:
        forg_str = ", ".join("None" if v is None else f"{v:+.4f}" for v in forgetting)
        final_forg = "None" if forgetting[-1] is None else f"{forgetting[-1]:+.4f}"
        logger.info("  • Mean Forgetting: [%s] (Final: %s)", forg_str, final_forg)

    status_str = (
        "PASSED (Base model parameters 100% frozen)"
        if frozen.get("params_unchanged")
        else "FAILED (Base model parameters modified!)"
    )
    logger.info("Substrate Integrity: %s", status_str)
    logger.info("=" * 78)


__all__ = [
    "DOMAIN_NAME",
    "SECTIONS",
    "Domain",
    "DomainConfig",
    "Objective",
    "RunConfig",
    "TrainingConfig",
    "base_logits",
    "batches",
    "checked_kwargs",
    "configure_logging",
    "evaluate",
    "evaluate_checkpoint",
    "load_configs",
    "load_token_blocks",
    "log_experiment_header",
    "log_final_summary",
    "make_eval_set",
    "resolve_split",
    "run_sequential",
    "summarize",
    "validate_checkpoint_params",
]
