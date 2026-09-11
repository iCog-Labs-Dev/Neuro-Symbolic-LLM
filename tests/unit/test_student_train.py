"""Exercise training settings with a tiny local model and real optimizer steps."""

from types import SimpleNamespace

import pytest
import torch

from parser.student import train as training


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.device = torch.device("cpu")

    def num_parameters(self):
        return 1

    def forward(self, input_ids, attention_mask, labels):
        targets = labels[:, 1:]
        loss = ((self.weight - targets[targets != -100].float()) ** 2).mean()
        return SimpleNamespace(loss=loss, logits=torch.zeros((*input_ids.shape, 8)))


@pytest.fixture
def run_training(monkeypatch, tmp_path):
    config = {
        "student": {"base_model": "tiny", "output_dir": str(tmp_path)},
        "data": {"val_split": 0, "max_length": 32},
        "training": {
            "batch_size": 1,
            "epochs": 2,
            "learning_rate": 0.1,
            "gradient_accumulation_steps": 4,
            "lr_scheduler": "cosine",
            "bf16": False,
        },
    }
    examples = [
        {
            "input": f"Example {i}",
            "input_ids": torch.ones(i + 2, dtype=torch.long),
            "attention_mask": torch.ones(i + 2, dtype=torch.long),
            "labels": torch.tensor([-100] + [i + 1] * (i + 1)),
        }
        for i in range(5)
    ]
    monkeypatch.setattr(training, "load_config", lambda _: config)
    monkeypatch.setattr(training, "load_pairs", lambda _: examples)
    monkeypatch.setattr(training, "StudentDataset", lambda pairs, *_: pairs)
    monkeypatch.setattr(training, "save_model", lambda *args: None)
    monkeypatch.setattr(
        training.AutoTokenizer,
        "from_pretrained",
        lambda _: SimpleNamespace(pad_token="pad", pad_token_id=0),
    )
    original_adam = torch.optim.AdamW

    def run(**settings):
        config["training"].update(settings)
        model = TinyModel()
        loaded = {}
        updates = []
        gradients = []

        def load_model(*args, **kwargs):
            loaded.update(kwargs)
            return model

        class RecordingAdam(original_adam):
            def step(self, closure=None):
                updates.append(self.param_groups[0]["lr"])
                gradients.append(model.weight.grad.item())
                return super().step(closure)

        monkeypatch.setattr(
            training.AutoModelForCausalLM, "from_pretrained", load_model
        )
        monkeypatch.setattr(torch.optim, "AdamW", RecordingAdam)
        training.train()
        return model.weight.detach(), updates, gradients, loaded

    return run


def test_accumulation_matches_combined_batches_and_flushes_each_epoch(run_training):
    accumulated, updates, gradients, loaded = run_training()
    combined, combined_updates, combined_gradients, _ = run_training(
        batch_size=4,
        gradient_accumulation_steps=1,
    )
    torch.testing.assert_close(accumulated, combined)
    assert gradients == pytest.approx(combined_gradients)
    assert updates == pytest.approx([0.1, 0.085355339, 0.05, 0.014644661])
    assert combined_updates == pytest.approx(updates)
    assert loaded["torch_dtype"] == torch.float32


def test_constant_scheduler_and_cpu_precision_fallback(run_training, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _, updates, _, loaded = run_training(lr_scheduler="constant", bf16=True)
    assert updates == pytest.approx([0.1] * 4)
    assert loaded["torch_dtype"] == torch.float32


@pytest.mark.parametrize(
    "supported, enabled, expected",
    [
        (True, True, torch.bfloat16),
        (False, True, torch.float32),
        (True, False, torch.float32),
    ],
)
def test_cuda_precision_selection(
    run_training, monkeypatch, supported, enabled, expected
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supported)
    _, _, _, loaded = run_training(bf16=enabled)
    assert loaded["torch_dtype"] == expected


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_accumulation_rejected(run_training, value):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        run_training(gradient_accumulation_steps=value)


def test_invalid_precision_flag_rejected(run_training):
    with pytest.raises(ValueError, match="bf16 must be a boolean"):
        run_training(bf16="false")


@pytest.fixture
def capture_training_splits(monkeypatch, tmp_path):
    """Use real JSONL loading and stop before any model is downloaded."""
    import json

    config = {
        "student": {"base_model": "tiny", "output_dir": str(tmp_path / "model")},
        "data": {"val_split": 0.2, "max_length": 32},
        "training": {"seed": 42},
    }
    monkeypatch.setattr(training, "load_config", lambda _: config)
    monkeypatch.setattr(
        training.AutoTokenizer,
        "from_pretrained",
        lambda _: SimpleNamespace(pad_token="pad", pad_token_id=0),
    )

    class SplitsCapturedError(Exception):
        pass

    def capture(pairs, *, seed=42, val_ratio=0.2, validation=None):
        config["training"]["seed"] = seed
        config["data"]["val_split"] = val_ratio
        captured = []

        def dataset(records, *_):
            captured.append(records)
            if len(captured) == 2:
                raise SplitsCapturedError
            return records

        def write(name, records):
            path = tmp_path / name
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            return str(path)

        monkeypatch.setattr(training, "StudentDataset", dataset)
        train_file = write("train.jsonl", pairs)
        val_file = None if validation is None else write("val.jsonl", validation)
        with pytest.raises(SplitsCapturedError):
            training.train(train_file=train_file, val_file=val_file)
        return captured

    return capture


def test_fallback_split_removes_duplicate_inputs(capture_training_splits):
    pairs = [{"input": f"Sentence {i}", "label": "{}"} for i in range(10)]
    pairs.append({"input": "  Sentence 0  ", "label": "{}"})
    train_pairs, val_pairs = capture_training_splits(pairs)
    train_texts = {pair["input"].strip() for pair in train_pairs}
    val_texts = {pair["input"].strip() for pair in val_pairs}
    assert len(train_pairs) == len(train_texts) == 8
    assert len(val_pairs) == len(val_texts) == 2
    assert train_texts.isdisjoint(val_texts)
    assert train_texts | val_texts == {f"Sentence {i}" for i in range(10)}
    assert len(pairs) == 11


def test_fallback_split_uses_reproducible_seed(capture_training_splits):
    pairs = [{"input": f"Sentence {i}", "label": "{}"} for i in range(20)]
    first = capture_training_splits(pairs, seed=42)
    assert first == capture_training_splits(pairs, seed=42)
    assert first != capture_training_splits(pairs, seed=7)


def test_fallback_zero_validation_keeps_all_unique_inputs(capture_training_splits):
    pairs = [{"input": "Dog", "label": "{}"}] * 2
    train_pairs, val_pairs = capture_training_splits(pairs, val_ratio=0)
    assert train_pairs == pairs[:1]
    assert val_pairs == []


def test_explicit_validation_is_preserved(capture_training_splits):
    pairs = [{"input": "Dog", "label": "{}"}]
    validation = [{"input": "Cat", "label": "{}"}]
    assert capture_training_splits(pairs, validation=validation) == [pairs, validation]


def test_explicit_validation_still_rejects_overlap(capture_training_splits):
    pairs = [{"input": "Dog", "label": "{}"}]
    validation = [{"input": " Dog ", "label": "{}"}]
    with pytest.raises(ValueError, match="Train and validation sets overlap"):
        capture_training_splits(pairs, validation=validation)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_fallback_rejects_invalid_ratios(capture_training_splits, ratio):
    with pytest.raises(ValueError, match="Split ratios"):
        capture_training_splits([{"input": "Dog", "label": "{}"}], val_ratio=ratio)


@pytest.mark.parametrize(
    "cuda_available, bf16_supported, expected",
    [
        (False, False, torch.float32),
        (True, False, torch.float32),
        (True, True, "auto"),
    ],
)
def test_evaluation_selects_safe_checkpoint_precision(
    monkeypatch, cuda_available, bf16_supported, expected
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)
    monkeypatch.setattr(
        training.AutoTokenizer,
        "from_pretrained",
        lambda _: SimpleNamespace(pad_token="pad"),
    )
    loaded = {}

    class ModelLoadCapturedError(Exception):
        pass

    def load_model(path, **kwargs):
        loaded.update(kwargs)
        raise ModelLoadCapturedError

    monkeypatch.setattr(training.AutoModelForCausalLM, "from_pretrained", load_model)
    with pytest.raises(ModelLoadCapturedError):
        training.evaluate("saved-model", test_sentences=["Dogs are animals."])
    assert loaded["torch_dtype"] == expected


@pytest.mark.parametrize(
    "content",
    [
        "null",
        "[]",
        "{broken",
        '{"input": null}',
        '{"input": "Dog", "label": null}',
        '{"input": "Dog", "label": "null"}',
        '{"input": "Dog", "label": "{broken"}',
    ],
)
def test_load_pairs_reports_bad_records(tmp_path, content):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n" + content, encoding="utf-8")
    with pytest.raises(ValueError, match=r"pairs.jsonl:2"):
        training.load_pairs(str(path))
