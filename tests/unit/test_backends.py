"""Offline tests for semantic-parser model backend adapters."""

from types import SimpleNamespace
from typing import Any

import pytest

from parser.semantic.backends import (
    AnthropicBackend,
    CallableBackend,
    GeminiBackend,
    OllamaBackend,
    OpenAIBackend,
    OpenAICompatibleBackend,
)


def test_callable_backend_forwards_arguments() -> None:
    calls = []

    def generate(*, prompt: str, model: str) -> str:
        calls.append((prompt, model))
        return "  (Has dog fur)  "

    backend = CallableBackend(generate, provider_name="test")

    assert backend.generate(prompt="prompt", model="model") == "(Has dog fur)"
    assert backend.provider_name == "test"
    assert calls == [("prompt", "model")]


def test_gemini_backend_reads_response_text() -> None:
    models = SimpleNamespace(
        generate_content=lambda **kwargs: SimpleNamespace(text="(Has dog fur)")
    )
    backend = GeminiBackend(client=SimpleNamespace(models=models))

    assert backend.generate(prompt="prompt", model="model") == "(Has dog fur)"


def test_openai_backend_reads_output_text() -> None:
    responses = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(output_text="(Has dog fur)")
    )
    backend = OpenAIBackend(client=SimpleNamespace(responses=responses))

    assert backend.generate(prompt="prompt", model="model") == "(Has dog fur)"


def test_anthropic_backend_joins_text_blocks() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="(Has dog "),
            SimpleNamespace(type="tool", text="ignored"),
            SimpleNamespace(type="text", text="fur)"),
        ]
    )
    messages = SimpleNamespace(create=lambda **kwargs: response)
    backend = AnthropicBackend(client=SimpleNamespace(messages=messages))

    assert backend.generate(prompt="prompt", model="model") == "(Has dog fur)"


def test_openai_compatible_backend_reads_first_choice() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="(Has dog fur)"))]
    )
    completions = SimpleNamespace(create=lambda **kwargs: response)
    chat = SimpleNamespace(completions=completions)
    backend = OpenAICompatibleBackend(SimpleNamespace(chat=chat))

    assert backend.generate(prompt="prompt", model="model") == "(Has dog fur)"


@pytest.mark.parametrize(
    "response",
    [{"response": "(Has dog fur)"}, SimpleNamespace(response="(Has dog fur)")],
)
def test_ollama_backend_supports_mapping_and_object_responses(response: Any) -> None:
    client = SimpleNamespace(generate=lambda **kwargs: response)
    backend = OllamaBackend(client=client)

    assert backend.generate(prompt="prompt", model="model") == "(Has dog fur)"


def test_backend_rejects_empty_output() -> None:
    backend = CallableBackend(lambda **kwargs: "   ")

    with pytest.raises(RuntimeError, match="returned an empty response"):
        backend.generate(prompt="prompt", model="model")
