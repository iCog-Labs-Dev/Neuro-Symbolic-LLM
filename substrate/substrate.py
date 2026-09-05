
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Mapping

from .torchax_backend import enable_torchax, to_torchax_device
from .torchax_models import functional_model
from .architecture import detect_architecture, validate_interception_layers

@dataclass(frozen=True)
class ForwardResult:
    \"\"\"Result of a forward pass through the substrate.\"\"\"
    logits: torch.Tensor
    hidden_states: Mapping[int, torch.Tensor] = None

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