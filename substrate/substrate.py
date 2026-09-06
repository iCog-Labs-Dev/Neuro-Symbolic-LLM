from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .architecture import detect_architecture
from .torchax_backend import enable_torchax
from .torchax_models import functional_model


@dataclass(frozen=True)
class ForwardResult:
    logits: torch.Tensor
    hidden_states: Mapping[int, torch.Tensor] | None = None


class Substrate:
    def __init__(self, model: torch.nn.Module, params: dict[str, torch.Tensor]):
        self.model = model
        self.params = params
        self.arch = detect_architecture(params)

    def __call__(
        self,
        input_ids: torch.Tensor,
        intercept_layers: Sequence[int] | None = None,
        hook: Callable | None = None,
    ) -> Any:
        enable_torchax()
        # Modern functional dispatch via torchax
        return functional_model(self.model, self.params, input_ids)
