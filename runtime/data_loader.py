"""Unified runtime dataset loader for sequential A1-A4 experiments.

Supported runtime modes:
    separate: Each configured dataset becomes one sequential task.
    stories:   Complete stories are grouped into sequential tasks.

The loader prepares shared Domain objects consumed by stage_base.run_sequential.
Stage-specific training objectives and optimizers remain independent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import load_dataset

from stages.stage_base import Domain, make_eval_set

# ---------------------------------------------------------------------
# Runtime data structures
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeData:
    """Prepared runtime data shared by all training stages."""

    tasks: tuple[Domain, ...]

    # Test token blocks, indexed by task name.
    test_blocks: dict[str, np.ndarray]

    # Actual sequential task order.
    task_order: tuple[str, ...]

    # Dataset and preparation metadata.
    metadata: dict[str, Any]

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
    streaming: bool = False,
):
    """Load one Hugging Face dataset split."""

    if not dataset_name:
        raise ValueError("Dataset name cannot be empty.")

    if not split:
        raise ValueError("Dataset split cannot be empty.")

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=streaming,
    )

    return dataset


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
        split=split,
        streaming=dataset_config.get("streaming", False),
    )

    return extract_texts(
        dataset,
        dataset_config.get("text_field", "text"),
    )


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------


def tokenize_texts(
    texts: list[str],
    tokenizer: Any,
    *,
    sequence_length: int,
    separator: str,
    task_name: str,
) -> np.ndarray:
    """Tokenize a collection of documents into fixed-length blocks.

    Each document is preserved as part of its text stream. The separator
    can be used to insert EOS boundaries between documents.
    """

    if not texts:
        raise ValueError(f"Task {task_name!r} contains no text.")

    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    if tokenizer is None:
        raise ValueError("A tokenizer must be supplied.")

    # Avoid joining documents without an explicit boundary.
    text = separator.join(texts)

    encoded = tokenizer(
        text,
        return_tensors="np",
        add_special_tokens=True,
    )

    input_ids = np.asarray(
        encoded["input_ids"][0],
        dtype=np.int32,
    )

    number_of_blocks = len(input_ids) // sequence_length

    if number_of_blocks == 0:
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
    """Resolve a document separator."""

    if configured_separator is not None:
        return configured_separator

    return tokenizer.eos_token or "\n\n"


# ---------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------


def build_separate_tasks(
    datasets_config: list[dict[str, Any]],
    tokenizer: Any,
    *,
    sequence_length: int,
    batch_size: int,
    eval_batches: int,
) -> tuple[tuple[Domain, ...], dict[str, np.ndarray]]:
    """Build one sequential task for every configured dataset."""

    tasks: list[Domain] = []
    test_blocks: dict[str, np.ndarray] = {}

    for index, item in enumerate(datasets_config):
        if item.get("type") != "separate":
            continue

        name = item["name"]

        print(f"\n[Runtime] Loading task {index + 1}: {name}")

        train_texts = load_split_texts(
            item,
            item["train_split"],
        )

        validation_texts = load_split_texts(
            item,
            item["validation_split"],
        )

        test_text = load_split_texts(
            item,
            item["test_split"],
        )

        separator = get_separator(
            tokenizer,
            item.get("separator"),
        )

        train_blocks = tokenize_texts(
            train_texts,
            tokenizer,
            sequence_length=sequence_length,
            separator=separator,
            task_name=name,
        )

        validation_blocks = tokenize_texts(
            validation_texts,
            tokenizer,
            sequence_length=sequence_length,
            separator=separator,
            task_name=f"{name}_validation",
        )

        test_blocks[name] = tokenize_texts(
            test_text,
            tokenizer,
            sequence_length=sequence_length,
            separator=separator,
            task_name=f"{name}_test",
        )

        eval_set = make_eval_set(
            validation_blocks,
            batch_size,
            eval_batches,
        )

        task = Domain(
            name=name,
            train_blocks=train_blocks,
            eval_set=eval_set,
            epochs=int(item.get("epochs", 3)),
        )

        tasks.append(task)

        print(f"  Training blocks: {len(train_blocks)}")
        print(f"  Validation blocks: {len(validation_blocks)}")
        print(f"  Test blocks: {len(test_blocks[name])}")

    if not tasks:
        raise ValueError("No datasets with type='separate' were configured.")

    return tuple(tasks), test_blocks


def group_stories(
    stories: list[str],
    stories_per_task: int,
    *,
    seed: int,
    shuffle: bool,
) -> list[list[str]]:
    """Group complete stories into sequential tasks."""

    if stories_per_task < 1:
        raise ValueError("stories_per_task must be >= 1.")

    stories = list(stories)

    if shuffle:
        random.Random(seed).shuffle(stories)

    return [
        stories[i : i + stories_per_task]
        for i in range(0, len(stories), stories_per_task)
    ]


def build_story_tasks(
    item: dict[str, Any],
    tokenizer: Any,
    *,
    sequence_length: int,
    batch_size: int,
    eval_batches: int,
    seed: int,
    default_epochs: int,
) -> tuple[tuple[Domain, ...], dict[str, np.ndarray]]:
    """Build sequential tasks from complete narrative stories."""

    name = item["name"]
    stories_per_task = int(item.get("stories_per_task", 10))

    print(f"\n[Runtime] Loading story dataset: {name}")

    train_stories = load_split_texts(
        item,
        item["train_split"],
    )

    validation_stories = load_split_texts(
        item,
        item["validation_split"],
    )

    test_stories = load_split_texts(
        item,
        item["test_split"],
    )

    groups = group_stories(
        train_stories,
        stories_per_task,
        seed=seed,
        shuffle=bool(item.get("shuffle_train", True)),
    )

    separator = get_separator(
        tokenizer,
        item.get("separator"),
    )

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

    test_blocks = tokenize_texts(
        test_stories,
        tokenizer,
        sequence_length=sequence_length,
        separator=separator,
        task_name=f"{name}_test",
    )

    tasks: list[Domain] = []

    for index, story_group in enumerate(groups):
        task_name = f"{name}_task_{index + 1}"

        train_blocks = tokenize_texts(
            story_group,
            tokenizer,
            sequence_length=sequence_length,
            separator=separator,
            task_name=task_name,
        )

        task = Domain(
            name=task_name,
            train_blocks=train_blocks,
            eval_set=eval_set,
            epochs=int(item.get("epochs", default_epochs)),
        )

        tasks.append(task)

        print(
            f"  {task_name}: "
            f"{len(story_group)} stories, "
            f"{len(train_blocks)} training blocks"
        )

    if not tasks:
        raise ValueError(f"No sequential story tasks were created for {name!r}.")

    return tuple(tasks), {name: test_blocks}


# ---------------------------------------------------------------------
# Public loader interface
# ---------------------------------------------------------------------


def load_runtime_data(
    config_path: str | Path,
    tokenizer: Any,
    *,
    sequence_length: int = 128,
    batch_size: int = 16,
    eval_batches: int = 8,
    default_epochs: int = 3,
) -> RuntimeData:
    """Prepare the complete runtime dataset for A1-A4.

    The selected runtime mode determines how datasets become tasks.
    All stages should consume the same returned RuntimeData instance
    or reproduce it from the same configuration and seed.
    """

    config = load_runtime_config(config_path)

    runtime_config = config["runtime"]
    datasets_config = config["datasets"]

    mode = runtime_config["source"]
    seed = int(runtime_config.get("seed", 42))

    if sequence_length < 1 or batch_size < 1:
        raise ValueError("sequence_length and batch_size must be positive.")

    if eval_batches < 1:
        raise ValueError("eval_batches must be positive.")

    if mode == "separate":
        tasks, test_blocks = build_separate_tasks(
            datasets_config,
            tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            eval_batches=eval_batches,
        )

    elif mode == "stories":
        story_configs = [
            item for item in datasets_config if item.get("type") == "stories"
        ]

        if len(story_configs) != 1:
            raise ValueError(
                "Story mode requires exactly one dataset " "with type='stories'."
            )

        tasks, test_blocks = build_story_tasks(
            story_configs[0],
            tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            eval_batches=eval_batches,
            seed=seed,
            default_epochs=default_epochs,
        )

    else:
        raise ValueError(f"Unsupported runtime mode: {mode!r}")

    task_order = tuple(task.name for task in tasks)

    metadata = {
        "mode": mode,
        "seed": seed,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "task_order": list(task_order),
        "datasets": [
            item["name"]
            for item in datasets_config
            if (mode == "separate" and item.get("type") == "separate")
            or (mode == "stories" and item.get("type") == "stories")
        ],
    }

    print("\n[Runtime] Dataset preparation complete.")
    print(f"  Mode: {mode}")
    print(f"  Number of tasks: {len(tasks)}")
    print(f"  Task order: {task_order}")

    return RuntimeData(
        tasks=tasks,
        test_blocks=test_blocks,
        task_order=task_order,
        metadata=metadata,
    )
