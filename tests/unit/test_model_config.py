"""Tests for configuration-driven semantic-parser backend selection."""

from parser.semantic.model_config import load_model_profile


def test_loads_active_profile_from_project_config() -> None:
    profile = load_model_profile()

    assert profile.name == "ollama"
    assert profile.provider == "ollama"
    assert profile.model == "qwen3:8b"


def test_can_select_named_profile() -> None:
    profile = load_model_profile(profile_name="ollama")

    assert profile.name == "ollama"
    assert profile.provider == "ollama"
    assert profile.model
