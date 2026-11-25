"""Tests for configuration-driven semantic-parser backend selection."""

import pytest
import yaml

from parser.semantic.model_config import default_model_config_path, load_model_profile


def test_loads_active_profile_from_project_config() -> None:
    config = yaml.safe_load(default_model_config_path().read_text(encoding="utf-8"))
    profile = load_model_profile()

    assert profile.name == config["active"]
    assert profile.provider == config["profiles"][profile.name]["provider"]
    assert profile.model == config["profiles"][profile.name]["model"]


def test_can_select_named_profile() -> None:
    profile = load_model_profile(profile_name="ollama_qwen3_8b")

    assert profile.name == "ollama_qwen3_8b"
    assert profile.provider == "ollama"
    assert profile.model


def test_catalog_ollama_profile_is_explicit_and_legacy_profile_is_unchanged() -> None:
    legacy = load_model_profile(profile_name="ollama")
    catalog = load_model_profile(profile_name="ollama_catalog")

    assert legacy.model == catalog.model
    assert "think" not in legacy.settings
    assert "temperature" not in legacy.settings
    assert catalog.settings["think"] is False
    assert catalog.settings["temperature"] == 0.0


def test_distilled_factory_uses_shipped_provider_budget_and_catalog(monkeypatch):
    from dataclasses import replace

    from parser.ontology.catalog import load_ontology_catalog
    from parser.semantic import model_config

    profile = load_model_profile(profile_name="semantic_document_student")
    profile = replace(profile, settings={**profile.settings, "max_new_tokens": 777})
    monkeypatch.setattr(model_config, "load_model_profile", lambda *_, **__: profile)
    calls = []
    monkeypatch.setattr(
        model_config.DistilledSemanticParser,
        "from_pretrained",
        lambda **kwargs: calls.append(kwargs),
    )
    catalog = load_ontology_catalog(modules=("upper",))
    model_config.build_distilled_semantic_parser(catalog=catalog)
    assert calls[0]["max_new_tokens"] == 777
    assert calls[0]["catalog"] is catalog


def _profile(provider: str, **settings):
    from parser.semantic.model_config import ModelProfile

    return ModelProfile(name="test", provider=provider, model="m", settings=settings)


def test_hosted_backend_receives_key_from_configured_variable(monkeypatch) -> None:
    from parser.semantic import backends, model_config

    captured = {}
    monkeypatch.setattr(model_config, "_load_environment", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_OPENAI_KEY", "sk-custom")
    monkeypatch.setattr(
        backends.OpenAIBackend,
        "__init__",
        lambda self, **kwargs: captured.update(kwargs),
    )

    model_config.create_backend(_profile("openai", api_key_env="CUSTOM_OPENAI_KEY"))

    assert captured["api_key"] == "sk-custom"


def test_ollama_backend_receives_host_from_configured_variable(monkeypatch) -> None:
    from parser.semantic import backends, model_config

    captured = {}
    monkeypatch.setattr(model_config, "_load_environment", lambda: None)
    monkeypatch.setenv("CUSTOM_OLLAMA_HOST", "http://gpu-box:11434")
    monkeypatch.setattr(
        backends.OllamaBackend,
        "__init__",
        lambda self, **kwargs: captured.update(kwargs),
    )

    model_config.create_backend(_profile("ollama", host_env="CUSTOM_OLLAMA_HOST"))

    assert captured["host"] == "http://gpu-box:11434"


def test_missing_credentials_and_invalid_settings_name_the_profile(
    monkeypatch,
) -> None:
    from parser.semantic import model_config

    monkeypatch.setattr(model_config, "_load_environment", lambda: None)
    monkeypatch.delenv("UNSET_KEY", raising=False)
    monkeypatch.delenv("UNSET_URL", raising=False)

    with pytest.raises(ValueError, match="'test' requires UNSET_KEY"):
        model_config.create_backend(_profile("anthropic", api_key_env="UNSET_KEY"))
    with pytest.raises(ValueError, match="'test' requires UNSET_URL"):
        model_config.create_backend(
            _profile("openai-compatible", base_url_env="UNSET_URL")
        )
    with pytest.raises(ValueError, match="'max_output_tokens' must be a positive"):
        model_config.create_backend(_profile("ollama", max_output_tokens="512"))
    with pytest.raises(ValueError, match="'seed' must be an integer"):
        model_config.create_backend(_profile("ollama", seed=True))


def test_model_env_resolves_model_and_rejects_ambiguous_profiles(
    tmp_path, monkeypatch
) -> None:
    from parser.semantic import model_config

    monkeypatch.setattr(model_config, "_load_environment", lambda: None)
    monkeypatch.setenv("SERVED_MODEL", " served/model ")
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "active": "env_model",
                "profiles": {
                    "env_model": {"provider": "ollama", "model_env": "SERVED_MODEL"},
                    "both": {
                        "provider": "ollama",
                        "model": "m",
                        "model_env": "SERVED_MODEL",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_model_profile(config)
    assert profile.model == "served/model"
    assert "model_env" not in profile.settings
    with pytest.raises(ValueError, match="either 'model' or 'model_env'"):
        load_model_profile(config, profile_name="both")
