"""parser/student/train.py

Fine-tunes the student parser on (input, json_label) pairs.

HOW TO RUN:
    python parser/student/train.py --train-file data/student_train.jsonl
    python parser/student/train.py --eval-only --model-dir models/distil_parser
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from parser.grammar.atomese import validate_metta_string
from parser.semantic.dataset import StudentDataset
from parser.semantic.metta_renderer import render_metta
from parser.semantic.schema import SemanticParseResult

# ── Load config ───────────────────────────────────────────────────────────────


def load_config(config_path: str = "configs/parser_config/student_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


# ── Load training data ────────────────────────────────────────────────────────


def load_pairs(train_file: str) -> list[dict]:
    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    pairs = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    pairs.append(json.loads(line))
    print(f"  Loaded {len(pairs)} pairs from {train_file}")
    return pairs


# ── LoRA setup ────────────────────────────────────────────────────────────────


def apply_lora(model, lora_config: dict):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as err:
        raise ImportError("Install peft: pip install peft") from err

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ── Save model ─────────────────────────────────────────────────────────────────


def save_model(model, tokenizer, output_dir: str, metadata: dict) -> None:
    """Save model, tokenizer, and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    with open(output_path / "training_metadata.json", "w") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    print(f"\n  Model saved to: {output_dir}")
    print("  Files saved:")
    for file_path in sorted(output_path.iterdir()):
        size = file_path.stat().st_size / 1024
        if size > 1024:
            print(f"    - {file_path.name} ({size/1024:.1f} MB)")
        else:
            print(f"    - {file_path.name} ({size:.1f} KB)")


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    model_dir: str,
    test_sentences: list[str] | None = None,
    max_tokens: int = 256,
) -> None:
    if test_sentences is None:
        test_sentences = [
            "Dogs are animals.",
            "Rain causes flooding.",
            "John likes pizza.",
            "Birds can fly.",
            "The dog has fur.",
            "Mary is happy.",
            "The cup is on the table.",
            "Cats are mammals.",
            "Children need education.",
            "The heart is part of the body.",
        ]

    print("\n" + "=" * 70)
    print("EVALUATING STUDENT MODEL")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    valid_json = 0
    valid_metta = 0

    for sentence in test_sentences:
        prompt = (
            f"Extract triples from this sentence as JSON:\nSentence: {sentence}\nJSON:"
        )
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                num_beams=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        new_ids = out_ids[0][prompt_len:]
        json_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        try:
            data = json.loads(json_output)
            valid_json += 1
            status = "JSON OK"

            try:
                result = SemanticParseResult.model_validate(data)
                expressions = render_metta(result)
                all_valid = True
                for expr in expressions:
                    ok, _ = validate_metta_string(expr)
                    if not ok:
                        all_valid = False
                        break
                if all_valid:
                    valid_metta += 1
                    status = "JSON + MeTTa OK"
                else:
                    status = "JSON OK, MeTTa FAIL"
            except Exception as e:
                status = f"JSON OK, Render FAIL: {str(e)[:30]}"
        except json.JSONDecodeError:
            status = "JSON FAIL"

        print(f"  [{status}] '{sentence[:40]:40s}' -> {json_output[:60]}...")

    print(
        f"\nValid JSON: {valid_json}/{len(test_sentences)} ({100*valid_json/len(test_sentences):.0f}%)"
    )
    print(
        f"Valid MeTTa: {valid_metta}/{len(test_sentences)} ({100*valid_metta/len(test_sentences):.0f}%)"
    )


# ── Main training ─────────────────────────────────────────────────────────────


def train(
    train_file: str = "data/student_train.jsonl",
    config_path: str = "configs/parser_config/student_config.yaml",
    output_dir: str | None = None,
    method: str = "full",
) -> str:
    config = load_config(config_path)

    base_model = config["student"]["base_model"]
    val_split = config["data"]["val_split"]
    max_length = config["data"]["max_length"]
    train_cfg = config["training"]
    lora_cfg = config.get("lora", {})

    if output_dir is None:
        output_dir = config["student"]["output_dir"]

    torch.manual_seed(train_cfg.get("seed", 42))

    print("\n" + "=" * 70)
    print("STUDENT PARSER TRAINING")
    print("=" * 70)
    print(f"  Method:          {method}")
    print(f"  Base model:      {base_model}")
    print(f"  Training data:   {train_file}")
    print(f"  Output dir:      {output_dir}")

    # Step 1: Load training pairs
    print("\nSTEP 1: Loading training data")
    all_pairs = load_pairs(train_file)

    if len(all_pairs) == 0:
        raise ValueError(f"No valid pairs found in {train_file}")

    split_idx = int(len(all_pairs) * (1 - val_split))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Step 2: Load tokenizer
    print("\nSTEP 2: Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Step 3: Build datasets
    print("\nSTEP 3: Building datasets")
    train_dataset = StudentDataset(train_pairs, tokenizer, max_length)
    val_dataset = StudentDataset(val_pairs, tokenizer, max_length)
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Step 4: Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"], shuffle=False
    )

    # Step 5: Load model
    print("\nSTEP 4: Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")

    # Step 6: Apply method
    if method == "lora":
        print("\nSTEP 5: Applying LoRA adapters")
        model = apply_lora(model, lora_cfg)
    else:
        print("\nSTEP 5: Full fine-tuning")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"  Trainable: {trainable / 1e9:.2f}B ({100 * trainable / model.num_parameters():.1f}%)"
        )

    # Step 7: Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    model.train()

    # Step 8: Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for epoch in range(train_cfg["epochs"]):
        print(f"Epoch {epoch + 1}/{train_cfg['epochs']}")

        for batch in train_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Calculate accuracy
            logits = outputs.logits
            labels = batch["labels"]
            shifted_labels = labels[:, 1:]
            shifted_logits = logits[:, :-1, :]
            predictions = shifted_logits.argmax(dim=-1)

            mask = shifted_labels != -100
            correct = (predictions == shifted_labels) & mask
            total = mask.sum().item()

            total_loss += loss.item()
            if total > 0:
                total_correct += correct.sum().item()
                total_tokens += total

            global_step += 1

            if global_step % 50 == 0:
                avg_loss = total_loss / 50
                accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
                print(
                    f"  Batch {global_step}: loss = {avg_loss:.4f}, acc = {accuracy:.4f}"
                )
                total_loss = 0.0
                total_correct = 0
                total_tokens = 0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                labels = batch["labels"]
                shifted_labels = labels[:, 1:]
                shifted_logits = logits[:, :-1, :]
                predictions = shifted_logits.argmax(dim=-1)

                mask = shifted_labels != -100
                correct = (predictions == shifted_labels) & mask
                total = mask.sum().item()

                if total > 0:
                    val_correct += correct.sum().item()
                    val_tokens += total

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_tokens if val_tokens > 0 else 0

        print(
            f"  Epoch {epoch + 1} complete: val_loss = {avg_val_loss:.4f}, val_acc = {val_acc:.4f}"
        )
        model.train()

    # Step 9: Save model
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
        "prompt_format": "Extract triples from this sentence as JSON:\nSentence: {text}\nJSON:",
        "epochs": train_cfg["epochs"],
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
    }

    save_model(model, tokenizer, output_dir, metadata)

    return output_dir


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/student_train.jsonl")
    parser.add_argument("--config", default="configs/parser_config/student_config.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--method", choices=["lora", "full"], default="full")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", default="models/distil_parser")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.model_dir)
    else:
        train(
            train_file=args.train_file,
            config_path=args.config,
            output_dir=args.output_dir,
            method=args.method,
        )


if __name__ == "__main__":
    main()
