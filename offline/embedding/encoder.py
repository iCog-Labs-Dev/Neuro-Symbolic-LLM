"""Offline context encoder using the existing FrozenSubstrate."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch

from frozenllm.substrate import FrozenSubstrate, enable_torchax


class FrozenLLMEncoder:
    """Encode text contexts using the existing frozen LLM substrate."""

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        max_length: int = 256,
        layer_idx: int | None = None,
        substrate: FrozenSubstrate | None = None,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive")

        self.model_name = model_name
        self.max_length = max_length

        # Match A1: initialize TorchAX before using the substrate.
        enable_torchax()

        if substrate is None:
            self.substrate = FrozenSubstrate(
                model_id_or_model=model_name,
                intercept_layers=[],
                dtype=torch.float32,
                attn_implementation="eager",
            )
        else:
            self.substrate = substrate

        self.tokenizer = self.substrate.tokenizer

        if self.tokenizer is None:
            raise ValueError("FrozenSubstrate must have a tokenizer")

        num_layers = self.substrate.architecture.num_layers

        # Default to the final transformer block.
        self.layer_idx = num_layers - 1 if layer_idx is None else layer_idx

        if not 0 <= self.layer_idx < num_layers:
            raise ValueError(f"layer_idx must be between 0 and {num_layers - 1}")

        self.hidden_size = self.substrate.architecture.hidden_size
        self.pooling = "mean_final_block_output"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one mean-pooled embedding per input text.

        Args:
            texts: Nonempty sequence of text contexts.

        Returns:
            Float32 NumPy array of shape
            (number_of_contexts, hidden_size).
        """
        if not texts:
            raise ValueError("texts must not be empty")

        vectors: list[np.ndarray] = []

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("each text must be a nonempty string")

            # Tokenize without padding, one context at a time.
            tokens = self.tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )

            # Match A1: pass a JAX array into FrozenSubstrate.
            input_ids = jnp.asarray(
                tokens["input_ids"],
                dtype=jnp.int32,
            )

            if input_ids.shape[1] == 0:
                raise ValueError("tokenizer produced an empty sequence")

            # Execute the existing frozen LLM and intercept
            # the requested transformer block's hidden states.
            result = self.substrate.run_with_interception(
                input_ids=input_ids,
                intercept_layers=[self.layer_idx],
            )

            hidden = result.hidden_state(self.layer_idx)

            # hidden shape: (1, sequence_length, hidden_size)
            # Mean pooling over the sequence dimension.
            pooled = jnp.mean(hidden, axis=1)

            vector = np.asarray(
                jax.device_get(pooled),
                dtype=np.float32,
            )

            expected_shape = (1, self.hidden_size)

            if vector.shape != expected_shape:
                raise ValueError(
                    "Unexpected pooled embedding shape: "
                    f"{vector.shape}; expected {expected_shape}"
                )

            if not np.isfinite(vector).all():
                raise ValueError("encoder produced nonfinite embeddings")

            vectors.append(vector[0])

        return np.stack(vectors, axis=0)
