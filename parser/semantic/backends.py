"""parser/semantic/backends.py

Model backends used by the semantic parsers.

The parser classes never call a provider SDK directly. They depend only on the
ModelBackend protocol below. This keeps the reference parser replaceable across
Gemini, OpenAI, Anthropic, Ollama, local 7B+ models, and the future distilled
1B-3B student.

Directly supported:
  - Gemini API
  - OpenAI Responses API
  - Anthropic Messages API
  - Ollama
  - OpenAI-compatible servers such as many vLLM deployments
  - local Hugging Face/Transformers generation
  - arbitrary Python callables
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

# ── Backend protocol ───────────────────────────────────────────────────────────


class ModelBackend(Protocol):
    """Common text-generation interface for teacher and student models."""

    @property
    def provider_name(self) -> str:
        """Return a stable backend identifier."""
        ...

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate one textual response for the supplied prompt."""
        ...


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _require_text(output: str | None, provider: str) -> str:
    """Return stripped output or raise when no usable text was produced."""
    if output is None or not output.strip():
        raise RuntimeError(f"{provider} returned an empty response")
    return output.strip()


# ── Hosted teacher backends ────────────────────────────────────────────────────


class GeminiBackend:
    """Generate semantic-parser output using Google's Gemini API."""

    provider_name = "gemini"

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_output_tokens: int = 256,
    ) -> None:
        """Create the backend, optionally injecting a client for tests."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if client is None:
            try:
                from google import genai
            except ImportError as error:
                raise ImportError(
                    "GeminiBackend requires 'google-genai'. "
                    "Install it with: pip install google-genai"
                ) from error
            client = genai.Client()
        self._client = client
        self._max_output_tokens = max_output_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using a Gemini model."""
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config={"max_output_tokens": self._max_output_tokens},
        )
        return _require_text(
            getattr(response, "text", None),
            self.provider_name,
        )


class OpenAIBackend:
    """Generate semantic-parser output using the OpenAI Responses API."""

    provider_name = "openai"

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_output_tokens: int = 256,
    ) -> None:
        """Create the backend, optionally injecting a client for tests."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "OpenAIBackend requires 'openai'. "
                    "Install it with: pip install openai"
                ) from error
            client = OpenAI()
        self._client = client
        self._max_output_tokens = max_output_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using an OpenAI model."""
        response = self._client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=self._max_output_tokens,
        )
        return _require_text(
            getattr(response, "output_text", None),
            self.provider_name,
        )


class AnthropicBackend:
    """Generate semantic-parser output using Anthropic's Messages API."""

    provider_name = "anthropic"

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_tokens: int = 256,
    ) -> None:
        """Create the backend, optionally injecting a client for tests."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as error:
                raise ImportError(
                    "AnthropicBackend requires 'anthropic'. "
                    "Install it with: pip install anthropic"
                ) from error
            client = Anthropic()

        self._client = client
        self._max_tokens = max_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using a Claude model."""
        response = self._client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        output = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return _require_text(output, self.provider_name)


# ── Local and compatible backends ─────────────────────────────────────────────


class OpenAICompatibleBackend:
    """Generate output through an OpenAI-compatible chat endpoint."""

    provider_name = "openai-compatible"

    def __init__(self, client: Any, *, max_output_tokens: int = 256) -> None:
        """Store a preconfigured compatible client."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        self._client = client
        self._max_output_tokens = max_output_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text through a chat-completions endpoint."""
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_output_tokens,
        )
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("openai-compatible returned a response without choices")
        return _require_text(
            getattr(choices[0].message, "content", None),
            self.provider_name,
        )


class OllamaBackend:
    """Generate output using a local or remote Ollama server."""

    provider_name = "ollama"

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_output_tokens: int = 256,
    ) -> None:
        """Create the backend, optionally injecting a client for tests."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if client is None:
            try:
                from ollama import Client
            except ImportError as error:
                raise ImportError(
                    "OllamaBackend requires 'ollama'. "
                    "Install it with: pip install ollama"
                ) from error
            client = Client()
        self._client = client
        self._max_output_tokens = max_output_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using an Ollama model."""
        response = self._client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={"num_predict": self._max_output_tokens},
        )
        output = (
            response.get("response")
            if isinstance(response, dict)
            else getattr(response, "response", None)
        )
        return _require_text(output, self.provider_name)


class TransformersBackend:
    """Generate output using a local Hugging Face/Transformers model.

    This backend is the intended deployment path for the distilled 1B-3B
    student after fine-tuning.
    """

    provider_name = "transformers-local"

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        max_new_tokens: int = 256,
    ) -> None:
        """Store a loaded tokenizer and model."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        self._tokenizer = tokenizer
        self._model = model
        self._max_new_tokens = max_new_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate deterministic output using the loaded local model."""
        del model

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to") and hasattr(self._model, "device"):
            inputs = inputs.to(self._model.device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        output = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return _require_text(output, self.provider_name)


class CallableBackend:
    """Adapt an arbitrary Python generation function to ModelBackend."""

    def __init__(
        self,
        generator: Callable[..., str],
        *,
        provider_name: str = "custom",
    ) -> None:
        """Store the generator and its provider identifier."""
        if not provider_name.strip():
            raise ValueError("provider_name cannot be empty")
        self._generator = generator
        self._provider_name = provider_name.strip()

    @property
    def provider_name(self) -> str:
        """Return the configured provider identifier."""
        return self._provider_name

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using the wrapped function."""
        return _require_text(
            self._generator(prompt=prompt, model=model),
            self.provider_name,
        )
