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


class OutputTruncatedError(RuntimeError):
    """Raised when a provider stops at the output-token limit."""


def _reject_truncation(truncated: bool, provider: str) -> None:
    """Fail clearly instead of returning output cut off at the token limit."""
    if truncated:
        raise OutputTruncatedError(
            f"{provider} stopped at the output-token limit; "
            "raise the profile's token budget"
        )


def _field(value: Any, name: str) -> Any:
    """Read a field from an SDK object or a plain mapping response."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


# ── Hosted teacher backends ────────────────────────────────────────────────────


class GeminiBackend:
    """Generate semantic-parser output using Google's Gemini API."""

    provider_name = "gemini"

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_output_tokens: int = 256,
        api_key: str | None = None,
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
            client = genai.Client(api_key=api_key)
        self._client = client
        self._max_output_tokens = max_output_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using a Gemini model."""
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config={"max_output_tokens": self._max_output_tokens},
        )
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            getattr(candidates[0], "finish_reason", None) if candidates else None
        )
        _reject_truncation(
            getattr(finish_reason, "name", finish_reason) == "MAX_TOKENS",
            self.provider_name,
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
        api_key: str | None = None,
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
            client = OpenAI(api_key=api_key)
        self._client = client
        self._max_output_tokens = max_output_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using an OpenAI model."""
        response = self._client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=self._max_output_tokens,
        )
        incomplete = getattr(response, "incomplete_details", None)
        _reject_truncation(
            getattr(incomplete, "reason", None) == "max_output_tokens",
            self.provider_name,
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
        api_key: str | None = None,
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
            client = Anthropic(api_key=api_key)

        self._client = client
        self._max_tokens = max_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using a Claude model."""
        response = self._client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        _reject_truncation(
            getattr(response, "stop_reason", None) == "max_tokens",
            self.provider_name,
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
        _reject_truncation(
            getattr(choices[0], "finish_reason", None) == "length", self.provider_name
        )
        return _require_text(
            getattr(choices[0].message, "content", None),
            self.provider_name,
        )


class OllamaBackend:
    """Generate output using a local or remote Ollama server.

    ``num_ctx`` sets the context window for each request. Ollama's default
    (4,096 tokens in current releases) is smaller than the catalog prompt, and
    Ollama silently drops the start of a prompt that does not fit, so the
    window must cover the prompt plus ``max_output_tokens``.
    """

    provider_name = "ollama"

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_output_tokens: int = 256,
        think: bool | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        num_ctx: int | None = None,
        host: str | None = None,
    ) -> None:
        """Create the backend, optionally injecting a client for tests."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("Ollama seed must be an integer")
        if num_ctx is not None and (
            not isinstance(num_ctx, int) or isinstance(num_ctx, bool) or num_ctx <= 0
        ):
            raise ValueError("Ollama num_ctx must be a positive integer")
        if client is None:
            try:
                from ollama import Client
            except ImportError as error:
                raise ImportError(
                    "OllamaBackend requires 'ollama'. "
                    "Install it with: pip install ollama"
                ) from error
            client = Client(host=host)
        self._client = client
        self._max_output_tokens = max_output_tokens
        self._think = think
        self._temperature = temperature
        self._seed = seed
        self._num_ctx = num_ctx

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate text using an Ollama model."""
        return self._generate(prompt=prompt, model=model)

    def generate_json(self, *, prompt: str, model: str, schema: dict[str, Any]) -> str:
        """Ask Ollama to constrain decoding to the supplied JSON schema."""
        return self._generate(prompt=prompt, model=model, schema=schema)

    def _generate(
        self, *, prompt: str, model: str, schema: dict[str, Any] | None = None
    ) -> str:
        options: dict[str, int | float] = {"num_predict": self._max_output_tokens}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        if self._seed is not None:
            options["seed"] = self._seed
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        thinking = {"think": self._think} if self._think is not None else {}
        formatting = {"format": schema} if schema is not None else {}
        response = self._client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options=options,
            **thinking,
            **formatting,
        )
        _reject_truncation(
            _field(response, "done_reason") == "length", self.provider_name
        )
        return _require_text(_field(response, "response"), self.provider_name)


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
        add_special_tokens: bool = True,
    ) -> None:
        """Store a loaded tokenizer and model."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        self._tokenizer = tokenizer
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._add_special_tokens = add_special_tokens

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate deterministic output using the loaded local model."""
        del model

        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=self._add_special_tokens
        )
        if hasattr(inputs, "to") and hasattr(self._model, "device"):
            inputs = inputs.to(self._model.device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        _reject_truncation(
            len(generated_ids) >= self._max_new_tokens
            and int(generated_ids[-1]) not in self._eos_token_ids(),
            self.provider_name,
        )
        output = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return _require_text(output, self.provider_name)

    def _eos_token_ids(self) -> set[int]:
        """Return the token ids that end generation for the loaded model."""
        config = getattr(self._model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self._tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        return {int(token) for token in (eos if isinstance(eos, list) else [eos])}


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
