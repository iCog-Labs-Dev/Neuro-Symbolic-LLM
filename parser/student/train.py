"""parser/student/train.py

Fine-tunes the student parser on (input, json_label) pairs.

LoRA training uses chunked selective vocabulary projection:
the transformer processes the complete sequence, while vocabulary
logits and cross-entropy are computed only at supervised positions.

Chunk gradients are accumulated at the hidden-state level, followed
by one backward pass through the transformer per training batch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from parser.semantic.dataset import (
    StudentBatchCollator,
    StudentDataset,
    split_pairs,
    validate_student_pair,
    verify_split_overlap,
)
from parser.student.student_prompt import STUDENT_PROMPT

if TYPE_CHECKING:
    from parser.ontology.catalog import OntologyCatalog


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_config(
    config_path: str = "configs/parser_config/student_config.yaml",
) -> dict:
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)  # type: ignore[no-any-return]


# ── Load training data ────────────────────────────────────────────────────────


def load_pairs(
    train_file: str,
    *,
    catalog: OntologyCatalog | None = None,
) -> list[dict]:
    train_path = Path(train_file)

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    pairs = []

    with open(train_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                pair = json.loads(line)
                validate_student_pair(pair, catalog=catalog)
            except ValueError as error:
                raise ValueError(
                    f"Invalid pair at {train_path}:{line_number}: {error}"
                ) from error

            pairs.append(pair)

    print(f"  Loaded {len(pairs)} pairs from {train_file}")
    return pairs


def _catalog_training_prompt_version(catalog):
    """Return the active combined student prompt version."""

    from parser.student.student_prompt import build_student_prompt_spec

    return build_student_prompt_spec(
        "Checkpoint compatibility.",
        catalog=catalog,
    ).version


# ── LoRA setup ────────────────────────────────────────────────────────────────


def apply_lora(model, lora_config: dict):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise ImportError("Install peft: pip install peft") from error

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------
# Model components and devices
# ---------------------------------------------------------------------


def _get_causal_lm_components(model):
    """Get the causal LM, transformer backbone, LM head and embeddings."""

    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model

    backbone = getattr(causal_lm, "model", None)
    lm_head = causal_lm.get_output_embeddings()
    embeddings = causal_lm.get_input_embeddings()

    if backbone is None or lm_head is None or embeddings is None:
        raise NotImplementedError(
            "Chunked loss requires a causal LM exposing its transformer "
            "backbone, input embeddings and output embeddings. "
            f"Received: {type(causal_lm).__name__}"
        )

    return causal_lm, backbone, lm_head, embeddings


def _get_input_device(model) -> torch.device:
    """Return the device containing the input embedding layer."""

    _, _, _, embeddings = _get_causal_lm_components(model)
    return embeddings.weight.device


def _move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


# ---------------------------------------------------------------------
# Supervised hidden states
# ---------------------------------------------------------------------


def _get_supervised_hidden_states(
    backbone,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process the complete sequence and select supervised causal positions."""

    backbone_kwargs = {
        "input_ids": batch["input_ids"],
        "use_cache": False,
        "return_dict": True,
    }

    if "attention_mask" in batch:
        backbone_kwargs["attention_mask"] = batch["attention_mask"]

    if "position_ids" in batch:
        backbone_kwargs["position_ids"] = batch["position_ids"]

    outputs = backbone(**backbone_kwargs)

    hidden_states = outputs.last_hidden_state

    # Hidden state at position t predicts the token at position t+1.
    shifted_hidden = hidden_states[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:].to(hidden_states.device)

    target_mask = shifted_labels != -100

    selected_hidden = shifted_hidden[target_mask]
    selected_labels = shifted_labels[target_mask]

    if selected_labels.numel() == 0:
        raise ValueError("Batch contains no supervised target tokens.")

    return selected_hidden, selected_labels


# ---------------------------------------------------------------------
# Chunked selective vocabulary loss
# ---------------------------------------------------------------------


def chunked_causal_lm_loss(
    model,
    batch: dict[str, torch.Tensor],
    chunk_size: int = 128,
    *,
    backward: bool = False,
) -> tuple[torch.Tensor, int, int]:
    """Calculate loss using chunked supervised-token vocabulary projection.

    The transformer processes the complete input sequence once.

    Only hidden states corresponding to supervised target tokens are
    projected into vocabulary space.

    For LoRA training, the vocabulary head is frozen. We calculate
    each chunk's gradient with respect to its hidden states, accumulate
    those gradients, and backpropagate through the transformer once.

    Returns:
        Mean supervised-token loss.
        Number of supervised target tokens.
        Number of correct predictions.
    """

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("loss_chunk_size must be a positive integer.")

    if backward and not torch.is_grad_enabled():
        raise RuntimeError("Training backward requested while gradients are disabled.")

    _, backbone, lm_head, _ = _get_causal_lm_components(model)

    # Hidden-state accumulation assumes the vocabulary head is frozen.
    trainable_head_parameters = [
        parameter for parameter in lm_head.parameters() if parameter.requires_grad
    ]

    if backward and trainable_head_parameters:
        raise NotImplementedError(
            "Chunked hidden-state gradient accumulation requires a "
            "frozen vocabulary head. The current LoRA configuration "
            "must not train the output head."
        )

    # One complete transformer forward pass.
    selected_hidden, selected_labels = _get_supervised_hidden_states(
        backbone,
        batch,
    )

    token_count = selected_labels.numel()

    hidden_device = selected_hidden.device
    head_device = next(lm_head.parameters()).device

    total_loss = torch.zeros(
        (),
        dtype=torch.float32,
        device=head_device,
    )

    total_correct = 0

    hidden_gradient: torch.Tensor | None = None

    if backward:
        hidden_gradient = torch.zeros(
            selected_hidden.shape,
            dtype=torch.float32,
            device=hidden_device,
        )

    # Project only supervised positions, in chunks.
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)

        chunk_hidden = selected_hidden[start:end].to(head_device)
        chunk_labels = selected_labels[start:end].to(head_device)

        chunk_logits = lm_head(chunk_hidden)

        chunk_loss_sum = functional.cross_entropy(
            chunk_logits.float(),
            chunk_labels,
            reduction="sum",
        )

        total_loss = total_loss + chunk_loss_sum.detach().float()

        with torch.no_grad():
            predictions = chunk_logits.argmax(dim=-1)

            total_correct += (predictions == chunk_labels).sum().item()

        if backward:
            # Stop gradient calculation at this hidden-state chunk.
            chunk_gradient = torch.autograd.grad(
                outputs=chunk_loss_sum,
                inputs=chunk_hidden,
                retain_graph=False,
                create_graph=False,
            )[0]

            if hidden_gradient is None:
                raise RuntimeError("Hidden-state gradient buffer was not initialized.")

            hidden_gradient[start:end].add_(
                chunk_gradient.detach().to(
                    device=hidden_device,
                    dtype=torch.float32,
                )
            )

            del chunk_gradient

        del chunk_hidden
        del chunk_labels
        del chunk_logits
        del chunk_loss_sum

    # One backward pass through the transformer.
    if backward:
        if hidden_gradient is None:
            raise RuntimeError("Hidden-state gradient buffer was not initialized.")

        torch.autograd.backward(
            selected_hidden,
            grad_tensors=hidden_gradient.to(dtype=selected_hidden.dtype),
        )

        del hidden_gradient

    mean_loss = (total_loss / token_count).detach()

    return mean_loss, token_count, total_correct


# ---------------------------------------------------------------------
# Standard loss for full fine-tuning
# ---------------------------------------------------------------------


def standard_causal_lm_step(
    model,
    batch: dict[str, torch.Tensor],
    *,
    backward: bool = False,
) -> tuple[torch.Tensor, int, int]:
    """Standard full-logit loss for full fine-tuning."""

    outputs = model(**batch)

    loss = outputs.loss

    labels = batch["labels"]
    shifted_labels = labels[:, 1:]
    shifted_logits = outputs.logits[:, :-1, :]

    predictions = shifted_logits.argmax(dim=-1)

    mask = shifted_labels != -100

    correct = (predictions == shifted_labels) & mask

    token_count = mask.sum().item()
    correct_count = correct.sum().item()

    if backward:
        (loss * token_count).backward()

    return loss.detach(), token_count, correct_count


# ---------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------


def save_model(
    model,
    tokenizer,
    output_dir: str,
    metadata: dict,
) -> None:
    """Save model, tokenizer and training metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    with open(
        output_path / "training_metadata.json",
        "w",
        encoding="utf-8",
    ) as meta_file:
        json.dump(metadata, meta_file, indent=2)

    print(f"\n  Model saved to: {output_dir}")
    print("  Files saved:")

    for file_path in sorted(output_path.iterdir()):
        size = file_path.stat().st_size / 1024

        if size > 1024:
            print(f"    - {file_path.name} ({size / 1024:.1f} MB)")
        else:
            print(f"    - {file_path.name} ({size:.1f} KB)")


# ---------------------------------------------------------------------
# Generation evaluation
# ---------------------------------------------------------------------


def evaluate(
    model_dir: str,
    test_sentences: list[str] | None = None,
    max_tokens: int = 3072,
) -> None:
    """Generate and validate predictions for supplied sources."""

    from parser.semantic.semantic_parser import DistilledSemanticParser

    if not test_sentences:
        raise ValueError(
            "Supply explicit evaluation sources; no synthetic default benchmark"
        )

    parser = DistilledSemanticParser.from_pretrained(
        model_dir,
        max_new_tokens=max_tokens,
    )

    valid = 0

    for sentence in test_sentences:
        try:
            result = parser.parse(sentence)
            print(result.model_dump_json())
            valid += 1

        except (ValueError, RuntimeError) as error:
            print(f"Rejected prediction: {error}")

    print(f"Validated documents: {valid}/{len(test_sentences)}")
    print("No expected labels supplied; semantic accuracy was not measured.")


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------


def train(
    train_file: str = "data/student_train.jsonl",
    config_path: str = "configs/parser_config/student_config.yaml",
    output_dir: str | None = None,
    method: str = "full",
    val_file: str | None = None,
) -> str:
    config = load_config(config_path)

    from parser.ontology.catalog import load_manifest_catalog
    from parser.semantic.schema import SEMANTIC_CONTRACT_VERSION

    # Configuration validation.
    contract_version = config["data"].get("contract_version")

    if contract_version != SEMANTIC_CONTRACT_VERSION:
        raise ValueError(
            f"Student training requires contract version "
            f"{SEMANTIC_CONTRACT_VERSION}"
        )

    manifest = config["data"].get("ontology_bundle_manifest")

    if not isinstance(manifest, str) or not manifest.strip():
        raise ValueError("Document training requires ontology_bundle_manifest")

    catalog, _ = load_manifest_catalog(Path(manifest))

    training_prompt_version = _catalog_training_prompt_version(catalog)

    base_model = config["student"]["base_model"]
    val_split = config["data"]["val_split"]

    max_length = config["data"]["max_length"]

    if max_length == "auto":
        max_length = None

    train_cfg = config["training"]
    lora_cfg = config.get("lora", {})

    accumulation_steps = train_cfg.get(
        "gradient_accumulation_steps",
        1,
    )

    if (
        isinstance(accumulation_steps, bool)
        or not isinstance(accumulation_steps, int)
        or accumulation_steps < 1
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")

    chunk_size = train_cfg.get("loss_chunk_size", 128)

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("loss_chunk_size must be a positive integer")

    bf16 = train_cfg.get("bf16", False)

    if not isinstance(bf16, bool):
        raise ValueError("bf16 must be a boolean")

    use_bf16 = bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if bf16 and not use_bf16:
        print("  BF16 is unavailable on the current device; using float32")

    if output_dir is None:
        output_dir = config["student"]["output_dir"]

    torch.manual_seed(train_cfg.get("seed", 42))

    # Training header.
    print("\n" + "=" * 70)
    print("STUDENT PARSER TRAINING")
    print("=" * 70)

    print(f"  Method:          {method}")
    print(f"  Base model:      {base_model}")
    print(f"  Training data:   {train_file}")
    print(f"  Output dir:      {output_dir}")

    if method == "lora":
        print("  Loss computation: " "Chunked selective supervised-token projection")
        print(f"  Loss chunk size:  {chunk_size}")

    # STEP 1: Load training data.
    print("\nSTEP 1: Loading training data")

    all_pairs = load_pairs(train_file, catalog=catalog)

    if not all_pairs:
        raise ValueError(f"No valid pairs found in {train_file}")

    if val_file is not None:
        train_pairs = all_pairs
        val_pairs = load_pairs(val_file, catalog=catalog)

    else:
        train_pairs, val_pairs, _ = split_pairs(
            all_pairs,
            train_ratio=1 - val_split,
            val_ratio=val_split,
            test_ratio=0,
            seed=train_cfg.get("seed", 42),
        )

    verify_split_overlap(train_pairs, val_pairs, [])

    print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # STEP 2: Load tokenizer.
    print("\nSTEP 2: Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # STEP 3: Build datasets.
    print("\nSTEP 3: Building datasets")

    train_dataset = StudentDataset(
        train_pairs,
        tokenizer,
        max_length,
        catalog=catalog,
    )

    val_dataset = StudentDataset(
        val_pairs,
        tokenizer,
        max_length,
        catalog=catalog,
    )

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    required_length = max(
        getattr(dataset, "required_max_length", 0)
        for dataset in (train_dataset, val_dataset)
    )

    context_limit = getattr(tokenizer, "model_max_length", None)

    if isinstance(context_limit, int) and required_length > context_limit:
        raise ValueError(
            f"Document pairs need {required_length} tokens; "
            f"tokenizer context limit is {context_limit}"
        )

    print(f"  Longest complete prompt and label: {required_length} tokens")

    if not train_dataset:
        raise ValueError(
            "No training examples fit max_length; increase it or check the data"
        )

    if val_pairs and not val_dataset:
        raise ValueError(
            "No validation examples fit max_length; increase it or check the data"
        )

    collator = StudentBatchCollator(tokenizer.pad_token_id)

    # STEP 4: Create dataloaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    # STEP 5: Load model.
    print("\nSTEP 4: Loading base model")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )

    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")

    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # STEP 6: Apply training method.
    if method == "lora":
        print("\nSTEP 5: Applying LoRA adapters")

        model = apply_lora(model, lora_cfg)

        _, _, lm_head, _ = _get_causal_lm_components(model)

        if any(parameter.requires_grad for parameter in lm_head.parameters()):
            raise RuntimeError(
                "The vocabulary head must remain frozen for "
                "chunked hidden-state gradient accumulation."
            )

    else:
        print("\nSTEP 5: Full fine-tuning")

        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print(
            f"  Trainable: {trainable / 1e9:.2f}B "
            f"({100 * trainable / model.num_parameters():.1f}%)"
        )

    if method == "lora":
        input_device = _get_input_device(model)
    else:
        model_device = getattr(model, "device", None)

        if model_device is None:
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")

        input_device = torch.device(model_device)

    print(f"  Input device: {input_device}")
    print("  Device map:", getattr(model, "hf_device_map", None))

    # STEP 7: Optimizer and scheduler.
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError("No trainable model parameters were found.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_cfg["learning_rate"],
    )

    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)

    scheduler = get_scheduler(
        train_cfg.get("lr_scheduler", "constant"),
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=updates_per_epoch * train_cfg["epochs"],
    )

    optimizer.zero_grad()
    model.train()

    # STEP 8: Training loop.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Preserve original global reporting schedule.
    global_step = 0
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for epoch in range(train_cfg["epochs"]):
        print(
            f"Epoch {epoch + 1}/{train_cfg['epochs']}",
            flush=True,
        )

        accumulated_tokens = 0

        for batch_index, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, input_device)

            if method == "lora":
                loss, batch_tokens, batch_correct = chunked_causal_lm_loss(
                    model,
                    batch,
                    chunk_size=chunk_size,
                    backward=True,
                )

            else:
                loss, batch_tokens, batch_correct = standard_causal_lm_step(
                    model,
                    batch,
                    backward=True,
                )

            accumulated_tokens += batch_tokens

            # Token-weighted gradient accumulation.
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(
                train_loader
            ):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_tokens)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                accumulated_tokens = 0

            # Training metrics.
            total_loss += loss.item()

            if batch_tokens > 0:
                total_correct += batch_correct
                total_tokens += batch_tokens

            global_step += 1

            # Original behavior: every 50 global batches across epochs.
            if global_step % 50 == 0:
                avg_loss = total_loss / 50

                accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

                print(
                    f"  Batch {global_step}: "
                    f"loss = {avg_loss:.4f}, "
                    f"acc = {accuracy:.4f}",
                    flush=True,
                )

                total_loss = 0.0
                total_correct = 0
                total_tokens = 0

            del loss

            if method == "lora":
                del batch

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch_to_device(batch, input_device)

                if method == "lora":
                    loss, batch_tokens, batch_correct = chunked_causal_lm_loss(
                        model,
                        batch,
                        chunk_size=chunk_size,
                        backward=False,
                    )

                else:
                    loss, batch_tokens, batch_correct = standard_causal_lm_step(
                        model,
                        batch,
                        backward=False,
                    )

                val_loss += loss.item()

                val_correct += batch_correct
                val_tokens += batch_tokens

                del loss
                del batch

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        val_acc = val_correct / val_tokens if val_tokens > 0 else 0.0

        print(
            f"  Epoch {epoch + 1} complete: "
            f"val_loss = {avg_val_loss:.4f}, "
            f"val_acc = {val_acc:.4f}",
            flush=True,
        )

        model.train()

    # STEP 9: Save model.
    print("\nSTEP 7: Saving model")

    if method == "lora":
        print("  Merging LoRA adapters...")
        model = model.merge_and_unload()

    metadata = {
        "base_model": base_model,
        "method": method,
        "train_file": train_file,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "config": config,
        "prompt_format": STUDENT_PROMPT,
        "target_contract_version": contract_version,
        "ontology_versions": list(catalog.versions),
        "prompt_version": training_prompt_version,
        "epochs": train_cfg["epochs"],
        "batch_size": train_cfg["batch_size"],
        "gradient_accumulation_steps": accumulation_steps,
        "learning_rate": train_cfg["learning_rate"],
        "loss_computation": (
            "chunked_selective_supervised_hidden_gradient"
            if method == "lora"
            else "standard_causal_lm"
        ),
        "loss_chunk_size": chunk_size if method == "lora" else None,
    }

    save_model(
        model,
        tokenizer,
        output_dir,
        metadata,
    )

    return output_dir


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-file",
        default="data/student_train.jsonl",
    )

    parser.add_argument(
        "--val-file",
        help="Separate validation pairs JSONL file",
    )

    parser.add_argument(
        "--config",
        default="configs/parser_config/student_config.yaml",
    )

    parser.add_argument("--output-dir")

    parser.add_argument(
        "--method",
        choices=["lora", "full"],
        default="full",
    )

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--test-sentence", action="append")

    parser.add_argument(
        "--model-dir",
        default="models/semantic_document_parser",
    )

    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.model_dir, args.test_sentence)

    else:
        train(
            train_file=args.train_file,
            val_file=args.val_file,
            config_path=args.config,
            output_dir=args.output_dir,
            method=args.method,
        )


if __name__ == "__main__":
    main()
