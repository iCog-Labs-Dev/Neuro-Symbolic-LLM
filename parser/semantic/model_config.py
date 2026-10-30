"""parser/semantic/model_config.py

Configuration-driven backend selection for semantic parsers.

This module contains:
  - ModelProfile:                      one named model/provider configuration
  - load_model_profile:                YAML → validated ModelProfile
  - create_backend:                    ModelProfile → provider backend instance
  - build_reference_semantic_parser:   configuration → ready reference parser
  - build_distilled_semantic_parser:   configuration → ready local student

The YAML file contains model choices. The local ``.env`` file contains API
keys, hosts, and other deployment values. Profiles name the environment
variables they read (``api_key_env``, ``base_url_env``, ``host_env``,
``hf_token_env``, ``model_env``), which keeps the YAML safe to commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from parser.ontology.catalog import OntologyCatalog, load_ontology_catalog
from parser.semantic.backends import (
    AnthropicBackend,
    GeminiBackend,
    ModelBackend,
    OllamaBackend,
    OpenAIBackend,
    OpenAICompatibleBackend,
    TransformersBackend,
)
from parser.semantic.semantic_parser import (
    DistilledSemanticParser,
    ReferenceSemanticParser,
    SemanticParserConfig,
)

# ── Model profile ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One named backend profile from the model configuration file."""

    name: str
    provider: str
    model: str
    settings: dict[str, Any]


# ── Configuration loading ────────────────────────────────────────────────────


def default_model_config_path() -> Path:
    """Return the repository's standard model-backend configuration path."""
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "parser_config"
        / "model_backend.yaml"
    )


def load_model_profile(
    config_path: str | Path | None = None,
    *,
    profile_name: str | None = None,
) -> ModelProfile:
    """Load the active (or explicitly named) model profile from YAML."""
    path = Path(config_path) if config_path is not None else default_model_config_path()
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Model configuration file was not found: {path}"
        ) from error
    if not isinstance(raw_config, dict):
        raise ValueError("Model configuration must be a YAML mapping")

    selected_name = profile_name or raw_config.get("active")
    profiles = raw_config.get("profiles")
    if not isinstance(selected_name, str) or not selected_name.strip():
        raise ValueError("Model configuration must define a non-empty 'active' profile")
    if not isinstance(profiles, dict):
        raise ValueError("Model configuration must define a 'profiles' mapping")
    raw_profile = profiles.get(selected_name)
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Unknown model profile: {selected_name!r}")

    provider = raw_profile.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"Profile {selected_name!r} must define a provider")

    model = raw_profile.get("model")
    model_env = raw_profile.get("model_env")
    if model_env is not None:
        if model is not None:
            raise ValueError(
                f"Profile {selected_name!r} must define either 'model' or "
                "'model_env', not both"
            )
        model = _require_environment(selected_name, "model_env", model_env)
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Profile {selected_name!r} must define a model")

    return ModelProfile(
        name=selected_name,
        provider=provider.strip(),
        model=model.strip(),
        settings={
            key: value
            for key, value in raw_profile.items()
            if key not in {"provider", "model", "model_env"}
        },
    )


# ── Environment ───────────────────────────────────────────────────────────────


def _load_environment() -> None:
    """Load local credentials without overwriting deployed environment values."""
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise ImportError(
            "Loading .env requires 'python-dotenv'. Install project dependencies first."
        ) from error
    load_dotenv(override=False)


def _read_environment(
    profile_name: str, setting: str, variable_name: Any
) -> str | None:
    """Return the stripped value of the variable a profile setting names, if set."""
    if not isinstance(variable_name, str) or not variable_name.strip():
        raise ValueError(
            f"Profile {profile_name!r} setting {setting!r} must name an "
            "environment variable"
        )
    _load_environment()
    value = os.getenv(variable_name.strip())
    return value.strip() if value and value.strip() else None


def _require_environment(profile_name: str, setting: str, variable_name: Any) -> str:
    """Return the value of the variable a profile setting names, or fail."""
    value = _read_environment(profile_name, setting, variable_name)
    if value is None:
        raise ValueError(
            f"Profile {profile_name!r} requires {variable_name.strip()}. "
            "Add it to .env or the process environment."
        )
    return value


def _required_environment(profile: ModelProfile, setting: str) -> str:
    """Return the value of a mandatory environment-backed setting."""
    if setting not in profile.settings:
        raise ValueError(f"Profile {profile.name!r} must define {setting!r}")
    return _require_environment(profile.name, setting, profile.settings[setting])


def _optional_environment(profile: ModelProfile, setting: str) -> str | None:
    """Return the value of an optional environment-backed setting, if set."""
    if profile.settings.get(setting) is None:
        return None
    return _read_environment(profile.name, setting, profile.settings[setting])


# ── Typed settings ────────────────────────────────────────────────────────────


def _int_setting(profile: ModelProfile, setting: str, default: int) -> int:
    value = profile.settings.get(setting, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Profile {profile.name!r} setting {setting!r} must be a positive integer"
        )
    return value


def _optional_int_setting(profile: ModelProfile, setting: str) -> int | None:
    value = profile.settings.get(setting)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(
            f"Profile {profile.name!r} setting {setting!r} must be an integer"
        )
    return value


def _optional_float_setting(profile: ModelProfile, setting: str) -> float | None:
    value = profile.settings.get(setting)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(
            f"Profile {profile.name!r} setting {setting!r} must be a number"
        )
    return float(value)


def _optional_bool_setting(profile: ModelProfile, setting: str) -> bool | None:
    value = profile.settings.get(setting)
    if value is not None and not isinstance(value, bool):
        raise ValueError(
            f"Profile {profile.name!r} setting {setting!r} must be a boolean"
        )
    return value


def _str_setting(profile: ModelProfile, setting: str, default: str) -> str:
    value = profile.settings.get(setting, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Profile {profile.name!r} setting {setting!r} must be a non-empty string"
        )
    return value.strip()


# ── Backend factory ───────────────────────────────────────────────────────────


def create_backend(profile: ModelProfile) -> ModelBackend:
    """Instantiate the implementation selected by a model profile."""
    if profile.provider == "openai":
        return OpenAIBackend(
            api_key=_required_environment(profile, "api_key_env"),
            max_output_tokens=_int_setting(profile, "max_output_tokens", 256),
        )
    if profile.provider == "anthropic":
        return AnthropicBackend(
            api_key=_required_environment(profile, "api_key_env"),
            max_tokens=_int_setting(profile, "max_tokens", 256),
        )
    if profile.provider == "gemini":
        return GeminiBackend(
            api_key=_required_environment(profile, "api_key_env"),
            max_output_tokens=_int_setting(profile, "max_output_tokens", 256),
        )
    if profile.provider == "ollama":
        return OllamaBackend(
            host=_optional_environment(profile, "host_env"),
            max_output_tokens=_int_setting(profile, "max_output_tokens", 256),
            think=_optional_bool_setting(profile, "think"),
            temperature=_optional_float_setting(profile, "temperature"),
            seed=_optional_int_setting(profile, "seed"),
            num_ctx=_optional_int_setting(profile, "num_ctx"),
        )
    if profile.provider == "openai-compatible":
        base_url = _required_environment(profile, "base_url_env")
        api_key = _optional_environment(profile, "api_key_env")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("OpenAI-compatible profiles require 'openai'.") from error
        return OpenAICompatibleBackend(
            # The SDK rejects a missing key, but many self-hosted servers need none.
            OpenAI(api_key=api_key or "not-needed", base_url=base_url),
            max_output_tokens=_int_setting(profile, "max_output_tokens", 256),
        )
    if profile.provider == "transformers-local":
        add_special_tokens = _optional_bool_setting(profile, "add_special_tokens")
        if add_special_tokens is None:
            add_special_tokens = True
        max_new_tokens = _int_setting(profile, "max_new_tokens", 256)
        token = _optional_environment(profile, "hf_token_env")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError("Local profiles require 'transformers'.") from error
        tokenizer = AutoTokenizer.from_pretrained(profile.model, token=token)
        model = AutoModelForCausalLM.from_pretrained(profile.model, token=token)
        return TransformersBackend(
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            add_special_tokens=add_special_tokens,
        )
    raise ValueError(
        f"Unsupported provider {profile.provider!r} in profile {profile.name!r}"
    )


# ── Parser builders ───────────────────────────────────────────────────────────


def build_reference_semantic_parser(
    config_path: str | Path | None = None,
    *,
    profile_name: str | None = None,
    catalog: OntologyCatalog | None = None,
    model_revision: str | None = None,
    require_structured_output: bool = False,
) -> ReferenceSemanticParser:
    """Build a reference parser using the active YAML model profile."""
    profile = load_model_profile(config_path, profile_name=profile_name)
    if (
        profile.settings.get("role") == "student"
        or (Path(profile.model) / "training_metadata.json").is_file()
    ):
        from parser.semantic.catalog_student import validate_catalog_student_checkpoint

        validate_catalog_student_checkpoint(
            profile.model,
            catalog=catalog if catalog is not None else load_ontology_catalog(),
        )
    return ReferenceSemanticParser(
        backend=create_backend(profile),
        config=SemanticParserConfig(model_name=profile.model),
        catalog=catalog,
        model_revision=model_revision,
        require_structured_output=require_structured_output,
    )


def build_distilled_semantic_parser(
    config_path: str | Path | None = None,
    *,
    profile_name: str | None = None,
    catalog: OntologyCatalog | None = None,
) -> DistilledSemanticParser:
    """Build the configured local document-target student."""
    profile = load_model_profile(
        config_path, profile_name=profile_name or "semantic_document_student"
    )
    if (
        profile.provider != "transformers-local"
        or profile.settings.get("role") != "student"
    ):
        raise ValueError("Distilled parser requires a local student model profile")
    return DistilledSemanticParser.from_pretrained(
        model_dir=profile.model,
        device=_str_setting(profile, "device", "auto"),
        max_new_tokens=_int_setting(profile, "max_new_tokens", 3072),
        catalog=catalog,
    )
