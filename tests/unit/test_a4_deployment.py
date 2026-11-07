"""Static A4 deployment wiring; no image build or live-server qualification."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def services():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]


def test_mork_uses_the_rust_service_image():
    service = services()["mork-server"]
    assert service["build"] == {
        "context": "./docker/mork",
        "dockerfile": "Dockerfile",
    }
    context = ROOT / service["build"]["context"]
    dockerfile = (context / "Dockerfile").read_text()
    assert dockerfile.startswith("FROM rust:")
    assert "https://github.com/trueagi-io/MORK.git" in dockerfile
    assert "USER mork" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "COPY entrypoint.sh /app/entrypoint.sh" in dockerfile
    assert (context / "entrypoint.sh").is_file()
    assert 'ENTRYPOINT ["/app/entrypoint.sh"]' in dockerfile


@pytest.mark.parametrize("name,command", [("app", "bash"), ("test", "pytest")])
def test_python_workloads_keep_the_root_application_image(name, command):
    service = services()[name]
    assert service["build"] == "."
    assert "profiles" not in service
    assert service["command"] == command
    assert "nsllm-hf-cache:/root/.cache/huggingface" in service["volumes"]
    application = (ROOT / "Dockerfile").read_text()
    assert "FROM python:3.11-slim" in application
    assert 'pip install -e ".[dev]"' in application
    assert not (ROOT / "Dockerfile.app").exists()


def test_make_build_still_builds_the_python_application():
    assert "docker build -t neuro-symbolic-llm ." in (ROOT / "Makefile").read_text()


def test_mork_ports_environment_healthcheck_and_mount_agree():
    service = services()["mork-server"]
    assert service["ports"] == ["8000:8000"]
    environment = dict(item.split("=", 1) for item in service["environment"])
    assert environment["MORK_SERVER_ADDR"] == "0.0.0.0"
    assert environment["MORK_SERVER_PORT"] == "8000"
    assert "./data/mork_data:/app/data" in service["volumes"]
    kind, command = service["healthcheck"]["test"]
    assert kind == "CMD-SHELL"
    assert "http://127.0.0.1:8000/" in command
    assert "200|404" in command
    assert (
        "MORK_SERVER_URL=http://127.0.0.1:8000" in (ROOT / ".env.example").read_text()
    )


def test_fixed_bank_mork_service_does_not_require_redis_or_application():
    configured = services()
    assert not configured["mork-server"].get("depends_on")
    assert configured["redis-queue"]["image"] == "redis:7-alpine"


def test_entrypoint_has_posix_line_endings_and_executes_the_server():
    content = (ROOT / "docker/mork/entrypoint.sh").read_bytes()
    assert content.startswith(b"#!/bin/sh\n")
    assert b"\r" not in content
    assert b"exec /app/MORK_SERVER/target/release/mork-server" in content
    assert b"exec /app/MORK_SERVER/target/release/mork" in content
    assert (
        "docker/mork/entrypoint.sh text eol=lf" in (ROOT / ".gitattributes").read_text()
    )
    assert not (ROOT / "entrypoint.sh").exists()


def test_docker_context_excludes_local_artifacts_and_credentials():
    excluded = set((ROOT / ".dockerignore").read_text().splitlines())
    assert excluded >= {".git", ".env", "data", "runs", "models", "checkpoints"}
