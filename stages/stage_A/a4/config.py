"""Stage A4 configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

TRAINING_STATE_SOURCE = "frozen-a3-prediction-v1"

if TYPE_CHECKING:
    from stages.stage_base import TrainingConfig


@dataclass(frozen=True)
class A4Config:
    residual_config: Path
    training: TrainingConfig
    key_weight: float = 1.0
    value_weight: float = 1.0
    top_m: int = 8
    minimum_templates: int = 10000

    def __post_init__(self) -> None:
        for name in ("key_weight", "value_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("top_m", "minimum_templates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_config": str(self.residual_config),
            "training": {
                name: value
                for name, value in asdict(self.training).items()
                if name != "reset_optimizer"
            },
            "objective": {
                "key_weight": self.key_weight,
                "value_weight": self.value_weight,
            },
            "retrieval": {
                "top_m": self.top_m,
                "minimum_templates": self.minimum_templates,
            },
        }


def load_config(path: Path) -> A4Config:
    """Load A4 settings and resolve the inherited residual configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    sections = {"residual_config", "training", "objective", "retrieval"}
    if not isinstance(raw, dict) or set(raw) != sections:
        raise ValueError(f"A4 configuration requires sections {sorted(sections)}")
    for name, fields in (
        ("objective", {"key_weight", "value_weight"}),
        ("retrieval", {"top_m", "minimum_templates"}),
    ):
        if not isinstance(raw[name], dict) or set(raw[name]) - fields:
            raise ValueError(f"Invalid A4 {name} fields")
    if not isinstance(raw["training"], dict) or not isinstance(
        raw["residual_config"], str
    ):
        raise ValueError("A4 training must be a mapping and residual_config a path")
    training_fields = {
        "model",
        "seq_len",
        "batch_size",
        "epochs",
        "learning_rate",
        "eval_batches",
        "log_every",
        "seed",
        "output_dir",
        "jit",
    }
    if set(raw["training"]) - training_fields:
        raise ValueError("Invalid A4 training fields")
    residual_path = (path.resolve().parent / raw["residual_config"]).resolve()
    if not residual_path.is_file():
        raise ValueError(f"Residual configuration not found: {residual_path}")
    from stages.stage_base import TrainingConfig

    return A4Config(
        residual_config=residual_path,
        training=TrainingConfig.from_dict(raw["training"]),
        **raw["objective"],
        **raw["retrieval"],
    )
