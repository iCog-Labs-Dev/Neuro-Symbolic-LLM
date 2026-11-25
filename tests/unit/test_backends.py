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
    OutputTruncatedError,
    TransformersBackend,
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


def test_ollama_backend_uses_default_decoding_settings() -> None:
    calls: list[dict[str, Any]] = []

    def generate(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs)
        return {"response": "{}"}

    backend = OllamaBackend(client=SimpleNamespace(generate=generate))

    assert backend.generate(prompt="prompt", model="model") == "{}"
    assert calls == [
        {
            "model": "model",
            "prompt": "prompt",
            "stream": False,
            "options": {"num_predict": 256},
        }
    ]


def test_ollama_backend_passes_decoding_settings_to_client() -> None:
    calls: list[dict[str, Any]] = []

    def generate(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs)
        return {"response": "{}"}

    backend = OllamaBackend(
        client=SimpleNamespace(generate=generate),
        max_output_tokens=512,
        think=False,
        temperature=0.0,
    )

    assert backend.generate(prompt="prompt", model="model") == "{}"
    assert calls == [
        {
            "model": "model",
            "prompt": "prompt",
            "stream": False,
            "think": False,
            "options": {"num_predict": 512, "temperature": 0.0},
        }
    ]


def test_backend_rejects_empty_output() -> None:
    backend = CallableBackend(lambda **kwargs: "   ")

    with pytest.raises(RuntimeError, match="returned an empty response"):
        backend.generate(prompt="prompt", model="model")


def test_ollama_backend_forwards_an_explicit_reproducibility_seed() -> None:
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return {"response": "{}"}

    backend = OllamaBackend(client=SimpleNamespace(generate=generate), seed=42)
    backend.generate(prompt="prompt", model="model")
    assert calls[0]["options"] == {"num_predict": 256, "seed": 42}


def test_ollama_backend_requests_json_schema_decoding_only_when_selected() -> None:
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return {"response": "{}"}

    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    backend = OllamaBackend(client=SimpleNamespace(generate=generate))
    assert backend.generate_json(prompt="prompt", model="model", schema=schema) == "{}"
    assert calls[0]["format"] == schema
    backend.generate(prompt="prompt", model="model")
    assert "format" not in calls[1]


def test_document_student_inference_uses_the_training_token_boundary() -> None:
    import torch

    tokenization = []

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            tokenization.append(kwargs)
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, **kwargs):
            assert ids.tolist() == [3]
            return "{}"

    model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[1, 2, 3]]))
    backend = TransformersBackend(
        tokenizer=Tokenizer(), model=model, add_special_tokens=False
    )
    assert backend.generate(prompt="prompt", model="model") == "{}"
    assert tokenization == [{"return_tensors": "pt", "add_special_tokens": False}]


@pytest.mark.parametrize(
    ("backend", "response"),
    [
        (
            "gemini",
            SimpleNamespace(
                text='{"a":',
                candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            ),
        ),
        (
            "openai",
            SimpleNamespace(
                output_text='{"a":',
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            ),
        ),
        (
            "anthropic",
            SimpleNamespace(
                stop_reason="max_tokens",
                content=[SimpleNamespace(type="text", text='{"a":')],
            ),
        ),
        (
            "openai-compatible",
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content='{"a":'),
                    )
                ]
            ),
        ),
        ("ollama", {"response": '{"a":', "done_reason": "length"}),
    ],
)
def test_backends_reject_output_cut_off_at_the_token_limit(
    backend: str, response: Any
) -> None:
    create = SimpleNamespace(create=lambda **kwargs: response)
    instance: Any = {
        "gemini": lambda: GeminiBackend(
            client=SimpleNamespace(
                models=SimpleNamespace(generate_content=lambda **kwargs: response)
            )
        ),
        "openai": lambda: OpenAIBackend(client=SimpleNamespace(responses=create)),
        "anthropic": lambda: AnthropicBackend(client=SimpleNamespace(messages=create)),
        "openai-compatible": lambda: OpenAICompatibleBackend(
            SimpleNamespace(chat=SimpleNamespace(completions=create))
        ),
        "ollama": lambda: OllamaBackend(
            client=SimpleNamespace(generate=lambda **kwargs: response)
        ),
    }[backend]()

    with pytest.raises(OutputTruncatedError, match="output-token limit"):
        instance.generate(prompt="prompt", model="model")


def test_transformers_backend_rejects_generation_that_exhausts_the_budget() -> None:
    import torch

    class Tokenizer:
        eos_token_id = 9

        def __call__(self, prompt, **kwargs):
            return {"input_ids": torch.tensor([[1]])}

        def decode(self, ids, **kwargs):
            return "{}"

    def backend(output: list[int]) -> TransformersBackend:
        model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([output]))
        return TransformersBackend(tokenizer=Tokenizer(), model=model, max_new_tokens=2)

    with pytest.raises(OutputTruncatedError, match="output-token limit"):
        backend([1, 4, 5]).generate(prompt="prompt", model="model")
    assert backend([1, 4, 9]).generate(prompt="prompt", model="model") == "{}"
