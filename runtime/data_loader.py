"""Unified runtime dataset loader for sequential language-model experiments.

Supported modes:
    separate: Each configured dataset becomes one sequential task.
    stories: Complete stories are grouped and packed into sequential tasks.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import load_dataset
from tqdm.auto import tqdm

from stages.stage_base import Domain, make_eval_set

# ---------------------------------------------------------------------
# Runtime data structures
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeData:
    """Prepared runtime data shared by all training stages."""

    tasks: tuple[Domain, ...]

    test_blocks: dict[str, np.ndarray]
    task_order: tuple[str, ...]
    metadata: dict[str, Any]
    training_tasks: tuple[Domain, ...] | None = None
    heldout_tasks: tuple[str, ...] = ()

    def get_test_sets(self, batch_size: int, eval_batches: int):
        return {
            name: make_eval_set(blocks, batch_size, eval_batches)
            for name, blocks in self.test_blocks.items()
        }


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_runtime_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the runtime YAML configuration."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Runtime configuration not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError("Runtime configuration must be a YAML mapping.")

    runtime = config.get("runtime")
    datasets = config.get("datasets")

    if not isinstance(runtime, dict):
        raise ValueError("Configuration requires a runtime section.")

    if runtime.get("source") not in {"separate", "stories"}:
        raise ValueError("runtime.source must be 'separate' or 'stories'.")

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Configuration requires a nonempty datasets list.")

    return config


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------


def load_dataset_split(
    dataset_name: str,
    split: str,
    *,
    dataset_config: str | None = None,
    revision: str | None = None,
    streaming: bool = False,
):
    """Load one Hugging Face dataset split."""

    if not dataset_name:
        raise ValueError("Dataset name cannot be empty.")

    if not split:
        raise ValueError("Dataset split cannot be empty.")

    return load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        revision=revision,
        streaming=streaming,
    )


def extract_texts(
    dataset,
    text_field: str,
) -> list[str]:
    """Extract nonempty text records from a dataset split."""

    if text_field not in dataset.column_names:
        raise ValueError(
            f"Text field {text_field!r} not found. "
            f"Available fields: {dataset.column_names}"
        )

    texts = []

    for value in dataset[text_field]:
        if isinstance(value, str):
            value = value.strip()

            if value:
                texts.append(value)

    if not texts:
        raise ValueError(f"No usable text found in field {text_field!r}.")

    return texts


def load_split_texts(
    dataset_config: dict[str, Any],
    split: str,
) -> list[str]:
    """Load and extract text from one configured dataset split."""

    dataset = load_dataset_split(
        dataset_name=dataset_config["dataset"],
        dataset_config=dataset_config.get("config"),
        revision=dataset_config.get("revision"),
        split=split,
        streaming=dataset_config.get("streaming", False),
    )

    return extract_texts(
        dataset,
        dataset_config.get("text_field", "text"),
    )


def array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------


def encode_texts(
    texts: list[str],
    tokenizer: Any,
    *,
    separator: str,
    task_name: str,
) -> np.ndarray:
    """Encode text using the frozen LLM tokenizer."""

    if not texts:
        raise ValueError(f"Task {task_name!r} contains no text.")

    if tokenizer is None:
        raise ValueError("A tokenizer must be supplied.")

    text = separator.join(texts)

    encoded = tokenizer(
        text,
        return_tensors="np",
        add_special_tokens=True,
    )

    return np.asarray(
        encoded["input_ids"][0],
        dtype=np.int32,
    )


def tokenize_texts(
    texts: list[str],
    tokenizer: Any,
    *,
    sequence_length: int,
    separator: str,
    task_name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    """Tokenize text into fixed-length blocks."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    input_ids = encode_texts(
        texts,
        tokenizer,
        separator=separator,
        task_name=task_name,
    )

    number_of_blocks = len(input_ids) // sequence_length

    if number_of_blocks == 0:
        if allow_empty:
            return np.empty(
                (0, sequence_length),
                dtype=np.int32,
            )

        raise ValueError(
            f"Task {task_name!r} contains only {len(input_ids)} tokens; "
            f"sequence_length={sequence_length}."
        )

    usable_length = number_of_blocks * sequence_length

    return input_ids[:usable_length].reshape(
        number_of_blocks,
        sequence_length,
    )


def get_separator(
    tokenizer: Any,
    configured_separator: str | None = None,
) -> str:
    """Resolve the document separator."""

    if configured_separator is not None:
        return configured_separator

    return tokenizer.eos_token or "\n\n"


# ---------------------------------------------------------------------
# Separate dataset tasks
# ---------------------------------------------------------------------


def build_separate_tasks(
    datasets_config: list[dict[str, Any]],
    tokenizer: Any,
    *,
    sequence_length: int,
    batch_size: int,
    eval_batches: int,
    default_epochs: int = 3,
) -> tuple[tuple[Domain, ...], dict[str, np.ndarray], dict[str, Any]]:
    """Build one sequential task for every configured dataset."""

    selected = [item for item in datasets_config if item.get("type") == "separate"]

    tasks: list[Domain] = []
    test_blocks: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {}

    total_steps = len(selected) * 6

    with tqdm(
        total=total_steps,
        desc="Preprocessing datasets",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        for item in selected:
            name = item["name"]

            progress.set_postfix_str("Loading training data")
            train_texts = load_split_texts(
                item,
                item["train_split"],
            )
            progress.update(1)

            progress.set_postfix_str("Loading validation data")
            validation_texts = load_split_texts(
                item,
                item["validation_split"],
            )
            progress.update(1)

            progress.set_postfix_str("Loading test data")
            test_texts = load_split_texts(
                item,
                item["test_split"],
            )
            progress.update(1)

            separator = get_separator(
                tokenizer,
                item.get("separator"),
            )

            progress.set_postfix_str("Tokenizing training data")
            train_blocks = tokenize_texts(
                train_texts,
                tokenizer,
                sequence_length=sequence_length,
                separator=separator,
                task_name=name,
            )
            progress.update(1)

            progress.set_postfix_str("Tokenizing validation data")
            validation_blocks = tokenize_texts(
                validation_texts,
                tokenizer,
                sequence_length=sequence_length,
                separator=separator,
                task_name=f"{name}_validation",
            )
            progress.update(1)

            progress.set_postfix_str("Tokenizing test data")
            test_blocks[name] = tokenize_texts(
                test_texts,
                tokenizer,
                sequence_length=sequence_length,
                separator=separator,
                task_name=f"{name}_test",
            )
            progress.update(1)

            eval_set = make_eval_set(
                validation_blocks,
                batch_size,
                eval_batches,
            )

            tasks.append(
                Domain(
                    name=name,
                    train_blocks=train_blocks,
                    eval_set=eval_set,
                    epochs=int(item.get("epochs", default_epochs)),
                )
            )

            stats[name] = {
                "dataset": item["dataset"],
                "config": item.get("config"),
                "revision": item.get("revision"),
                "train_split": item["train_split"],
                "validation_split": item["validation_split"],
                "test_split": item["test_split"],
                "training_blocks": len(train_blocks),
                "validation_blocks": len(validation_blocks),
                "test_blocks": len(test_blocks[name]),
                "dropped_training_blocks": (len(train_blocks) % batch_size),
                "training_sha256": array_fingerprint(train_blocks),
                "validation_sha256": array_fingerprint(validation_blocks),
                "test_sha256": array_fingerprint(test_blocks[name]),
            }

    if not tasks:
        raise ValueError("No datasets with type='separate' were configured.")

    return tuple(tasks), test_blocks, stats


# ---------------------------------------------------------------------
# Story grouping
# ---------------------------------------------------------------------


def group_stories(
    stories: list[str],
    stories_per_task: int,
    *,
    seed: int,
    shuffle: bool,
) -> list[list[str]]:
    """Group complete stories into sequential preprocessing groups."""

    if stories_per_task < 1:
        raise ValueError("stories_per_task must be >= 1.")

    stories = list(stories)

    if shuffle:
        random.Random(seed).shuffle(stories)

    return [
        stories[i : i + stories_per_task]
        for i in range(0, len(stories), stories_per_task)
    ]


# ---------------------------------------------------------------------
# Story task construction with token packing and max_tasks
# ---------------------------------------------------------------------


def build_story_tasks(
    item: dict[str, Any],
    tokenizer: Any,
    *,
    sequence_length: int,
    batch_size: int,
    eval_batches: int,
    seed: int,
    default_epochs: int,
    max_tasks: int | None = None,
) -> tuple[tuple[Domain, ...], dict[str, np.ndarray], dict[str, Any]]:
    """Build sequential story tasks with optional max_tasks limit.

    Token remainders are carried across story groups.
    Complete blocks are accumulated until a full training batch exists.

    max_tasks limits the number of created training tasks.
    Validation and test data are not restricted by max_tasks.
    """

    name = item["name"]

    stories_per_task = int(item.get("stories_per_task", 10))

    if max_tasks is not None and (
        isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or max_tasks < 1
    ):
        raise ValueError("max_tasks must be a positive integer or null.")
    # -------------------------------------------------------------
    # Load all splits
    # -------------------------------------------------------------

    train_stories = load_split_texts(
        item,
        item["train_split"],
    )

    groups = group_stories(
        train_stories,
        stories_per_task,
        seed=seed,
        shuffle=bool(item.get("shuffle_train", True)),
    )

    validation_stories = load_split_texts(
        item,
        item["validation_split"],
    )

    test_stories = load_split_texts(
        item,
        item["test_split"],
    )

    separator = get_separator(
        tokenizer,
        item.get("separator"),
    )

    # -------------------------------------------------------------
    # Validation and test preparation
    # -------------------------------------------------------------

    with tqdm(
        total=len(groups) + 2,
        desc="Preprocessing stories",
        unit="group",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix_str("Tokenizing validation")

        validation_blocks = tokenize_texts(
            validation_stories,
            tokenizer,
            sequence_length=sequence_length,
            separator=separator,
            task_name=f"{name}_validation",
        )

        eval_set = make_eval_set(
            validation_blocks,
            batch_size,
            eval_batches,
        )

        progress.update(1)

        progress.set_postfix_str("Tokenizing test")

        test_blocks = tokenize_texts(
            test_stories,
            tokenizer,
            sequence_length=sequence_length,
            separator=separator,
            task_name=f"{name}_test",
        )

        progress.update(1)

        # ---------------------------------------------------------
        # Training token and block buffers
        # ---------------------------------------------------------

        token_buffer = np.empty(
            (0,),
            dtype=np.int32,
        )

        block_buffer = np.empty(
            (0, sequence_length),
            dtype=np.int32,
        )

        tasks: list[Domain] = []

        total_training_blocks = 0
        retained_blocks = 0
        processed_groups = 0
        processed_stories = 0

        # ---------------------------------------------------------
        # Build tasks
        # ---------------------------------------------------------

        for index, story_group in enumerate(groups):
            progress.set_postfix_str(f"Packing task groups {index + 1}/{len(groups)}")

            group_ids = encode_texts(
                story_group,
                tokenizer,
                separator=separator,
                task_name=f"{name}_group_{index + 1}",
            )

            # Preserve token remainder from previous groups.
            token_buffer = np.concatenate((token_buffer, group_ids))

            # Extract complete fixed-length sequences.
            number_of_blocks = len(token_buffer) // sequence_length

            if number_of_blocks:
                usable_length = number_of_blocks * sequence_length

                new_blocks = token_buffer[:usable_length].reshape(
                    number_of_blocks,
                    sequence_length,
                )

                token_buffer = token_buffer[usable_length:]

                if len(block_buffer):
                    block_buffer = np.concatenate(
                        (block_buffer, new_blocks),
                        axis=0,
                    )
                else:
                    block_buffer = new_blocks

                total_training_blocks += number_of_blocks

            # Assign only complete batches to training tasks.
            complete_blocks = (len(block_buffer) // batch_size) * batch_size

            if complete_blocks:
                task_blocks = block_buffer[:complete_blocks]

                block_buffer = block_buffer[complete_blocks:]

                task_name = f"{name}_task_{len(tasks) + 1}"

                tasks.append(
                    Domain(
                        name=task_name,
                        train_blocks=task_blocks,
                        eval_set=eval_set,
                        epochs=int(item.get("epochs", default_epochs)),
                    )
                )

                retained_blocks += len(task_blocks)

            processed_groups += 1
            processed_stories += len(story_group)

            progress.update(1)

            # Stop processing once max_tasks is reached.
            if max_tasks is not None and len(tasks) >= max_tasks:
                break

        # When stopping early, adjust the progress bar total.
        if max_tasks is not None and len(tasks) >= max_tasks:
            progress.total = progress.n
            progress.refresh()

        # ---------------------------------------------------------
        # Record incomplete data
        # ---------------------------------------------------------

        dropped_blocks = len(block_buffer)
        dropped_tokens = len(token_buffer)

    if not tasks:
        raise ValueError(
            f"No usable training tasks were created for {name!r}. "
            f"Each task requires at least {batch_size} blocks. "
            "Increase stories_per_task or reduce "
            "batch_size/sequence_length."
        )

    stats = {
        "dataset": name,
        "training_stories": len(train_stories),
        "processed_training_stories": processed_stories,
        "original_story_groups": len(groups),
        "processed_story_groups": processed_groups,
        "max_tasks": max_tasks,
        "retained_tasks": len(tasks),
        "stories_per_group": stories_per_task,
        "total_complete_training_blocks": total_training_blocks,
        "retained_training_blocks": retained_blocks,
        "dropped_incomplete_blocks": dropped_blocks,
        "dropped_trailing_tokens": dropped_tokens,
        "validation_blocks": len(validation_blocks),
        "test_blocks": len(test_blocks),
    }

    return tuple(tasks), {name: test_blocks}, stats


# ---------------------------------------------------------------------
# Public loader interface
# ---------------------------------------------------------------------


def load_runtime_data(
    config_path: str | Path,
    tokenizer: Any,
    *,
    source: str | None = None,
    task_order: tuple[str, ...] | list[str] | None = None,
    training_mode: str | None = None,
    heldout_tasks: tuple[str, ...] | list[str] | None = None,
    sequence_length: int = 128,
    batch_size: int = 16,
    eval_batches: int = 8,
    default_epochs: int = 3,
) -> RuntimeData:
    """Prepare runtime datasets for sequential training."""

    config = load_runtime_config(config_path)

    runtime_config = config["runtime"]
    datasets_config = config["datasets"]

    mode = runtime_config["source"] if source is None else source
    seed = int(runtime_config.get("seed", 42))
    requested_order = tuple(task_order or runtime_config.get("task_order", ()))
    training_mode = training_mode or str(
        runtime_config.get("training_mode", "sequential")
    )
    heldout_tasks = tuple(
        heldout_tasks
        if heldout_tasks is not None
        else runtime_config.get("heldout_tasks", ())
    )

    if mode not in {"separate", "stories"}:
        raise ValueError("source must be 'separate' or 'stories'.")
    if training_mode not in {"sequential", "joint"}:
        raise ValueError("training_mode must be 'sequential' or 'joint'.")

    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2 for next-token training.")

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    if eval_batches < 1:
        raise ValueError("eval_batches must be >= 1.")

    if mode == "separate":
        tasks, test_blocks, preprocessing_stats = build_separate_tasks(
            datasets_config,
            tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            eval_batches=eval_batches,
            default_epochs=default_epochs,
        )

    elif mode == "stories":
        story_configs = [
            item for item in datasets_config if item.get("type") == "stories"
        ]

        if len(story_configs) != 1:
            raise ValueError(
                "Story mode requires exactly one dataset with type='stories'."
            )

        tasks, test_blocks, preprocessing_stats = build_story_tasks(
            story_configs[0],
            tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            eval_batches=eval_batches,
            seed=seed,
            default_epochs=default_epochs,
            max_tasks=runtime_config.get("max_tasks"),
        )

    else:
        raise ValueError(f"Unsupported runtime mode: {mode!r}")

    if requested_order:
        available = {task.name: task for task in tasks}
        if set(requested_order) != set(available) or len(requested_order) != len(
            available
        ):
            raise ValueError(
                f"task_order must contain each task exactly once; available={sorted(available)}"
            )
        tasks = tuple(available[name] for name in requested_order)

    heldout = tuple(str(name) for name in heldout_tasks)
    unknown_heldout = sorted(set(heldout) - {task.name for task in tasks})
    if unknown_heldout:
        raise ValueError(f"Unknown held-out tasks: {unknown_heldout}")
    if len(set(heldout)) != len(heldout):
        raise ValueError("heldout_tasks must be unique")
    source_tasks = tuple(task for task in tasks if task.name not in set(heldout))
    if not source_tasks:
        raise ValueError("At least one non-held-out training task is required")

    if training_mode == "joint":
        expanded = [
            np.tile(
                task.train_blocks[
                    : (len(task.train_blocks) // batch_size) * batch_size
                ],
                (task.epochs, 1),
            )
            for task in source_tasks
            if task.epochs > 0
        ]
        if not expanded:
            joint_blocks = np.empty((0, sequence_length), dtype=np.int32)
        else:
            joint_blocks = np.concatenate(expanded, axis=0)
        training_tasks = (
            Domain(
                name="joint",
                train_blocks=joint_blocks,
                eval_set=(),
                epochs=1,
            ),
        )
    else:
        training_tasks = source_tasks

    tasks = source_tasks
    task_order_names = tuple(task.name for task in tasks)

    metadata = {
        "mode": mode,
        "seed": seed,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "task_order": list(task_order_names),
        "training_mode": training_mode,
        "heldout_tasks": list(heldout),
        "datasets": [
            item["name"]
            for item in datasets_config
            if (mode == "separate" and item.get("type") == "separate")
            or (mode == "stories" and item.get("type") == "stories")
        ],
        "preprocessing": preprocessing_stats,
    }

    return RuntimeData(
        tasks=tasks,
        test_blocks=test_blocks,
        task_order=task_order_names,
        metadata=metadata,
        training_tasks=training_tasks,
        heldout_tasks=heldout,
    )


__all__ = [
    "RuntimeData",
    "array_fingerprint",
    "load_runtime_config",
    "load_runtime_data",
    "load_dataset_split",
    "load_split_texts",
    "extract_texts",
    "encode_texts",
    "tokenize_texts",
    "group_stories",
    "build_separate_tasks",
    "build_story_tasks",
]
