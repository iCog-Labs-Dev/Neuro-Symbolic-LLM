"""Canonical benchmark identities and immutable artifact validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENERATED_REPORT_PREFIX = "report_and_visualization/"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def source_fingerprint(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    diff = git("diff", "--binary", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard")
    untracked_paths = (
        []
        if not untracked
        else sorted(
            path
            for path in untracked.splitlines()
            if path != GENERATED_REPORT_PREFIX.rstrip("/")
            and not path.startswith(GENERATED_REPORT_PREFIX)
        )
    )
    untracked_source_hashes = {
        relative: sha256_file(root / relative)
        for relative in untracked_paths
        if Path(relative).suffix.lower() in {".py", ".yaml", ".yml", ".toml"}
        and (root / relative).is_file()
    }
    payload = {
        "git_commit": commit,
        "tracked_diff_sha256": (
            None if diff is None else sha256_bytes(diff.encode("utf-8"))
        ),
        "untracked_paths": untracked_paths,
        "untracked_source_hashes": untracked_source_hashes,
    }
    payload["source_sha256"] = fingerprint(payload)
    return payload


def effective_condition(
    *,
    method: str,
    seed: int,
    rank: int,
    k: int | None,
    protocol: str,
    task_order: list[str] | tuple[str, ...],
    heldout_tasks: list[str] | tuple[str, ...],
    a2_arm: str | None,
    stage_config_sha256: str,
    runtime_config_sha256: str,
    runtime_source: str,
    epochs: int | None,
    source_sha256: str,
) -> dict[str, Any]:
    condition = {
        "schema": "nsllm.condition/v2",
        "method": method,
        "seed": int(seed),
        "rank": int(rank),
        "training_k": k,
        "deployment_k": 0 if method == "a3" else None,
        "protocol": protocol,
        "task_order": list(task_order),
        "heldout_tasks": sorted(heldout_tasks),
        "a2_arm": a2_arm,
        "stage_config_sha256": stage_config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "runtime_source": runtime_source,
        "epochs": epochs,
        "source_sha256": source_sha256,
    }
    condition["condition_sha256"] = fingerprint(condition)
    return condition


def next_attempt_name(condition_dir: Path) -> str:
    condition_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        name = f"attempt-{attempt:04d}"
        if (condition_dir / name).exists():
            attempt += 1
            continue
        claim = condition_dir / f"{name}.claim"
        try:
            claim.open("x", encoding="utf-8").close()
        except FileExistsError:
            attempt += 1
            continue
        return name


def finalize_attempt(run_dir: Path, spec: dict[str, Any]) -> None:
    """Write immutable run identity, hash every artifact, and mark completion."""
    (run_dir / "spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ignored = {"artifacts.json", "COMPLETE", "FAILED"}
    artifacts = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_file() and relative not in ignored:
            artifacts[relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {"version": 1, "artifacts": artifacts},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "COMPLETE").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "artifact_index": "artifacts.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_complete_attempt(
    run_dir: Path,
    *,
    condition_sha256: str,
    config_sha256: str,
) -> tuple[bool, str | None]:
    required = (
        run_dir / "COMPLETE",
        run_dir / "results.json",
        run_dir / "config.yaml",
        run_dir / "spec.json",
        run_dir / "artifacts.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return False, f"missing artifacts: {missing}"
    try:
        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        marker = json.loads((run_dir / "COMPLETE").read_text(encoding="utf-8"))
        index = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        return False, f"invalid completion metadata: {error}"
    if spec.get("condition_sha256") != condition_sha256:
        return False, "condition fingerprint mismatch"
    if spec.get("config_sha256") != config_sha256:
        return False, "effective config fingerprint mismatch"
    materialized_config = run_dir / "provenance" / "materialized_stage.yaml"
    if (
        not materialized_config.is_file()
        or sha256_file(materialized_config) != config_sha256
    ):
        return False, "materialized stage config artifact mismatch"
    runtime_config = run_dir / "provenance" / "materialized_runtime.yaml"
    expected_runtime = spec.get("runtime_config_sha256")
    if (
        not runtime_config.is_file()
        or not isinstance(expected_runtime, str)
        or sha256_file(runtime_config) != expected_runtime
    ):
        return False, "materialized runtime config artifact mismatch"
    if marker.get("status") != "COMPLETE":
        return False, "completion marker is not successful"
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, "artifact index is invalid"
    actual_paths = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix()
        not in {"artifacts.json", "COMPLETE", "FAILED"}
    }
    if actual_paths != set(artifacts):
        return False, "artifact index coverage mismatch"
    for relative, metadata in artifacts.items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            return False, f"artifact hash mismatch: {relative}"
    return True, None


__all__ = [
    "canonical_json",
    "effective_condition",
    "fingerprint",
    "finalize_attempt",
    "next_attempt_name",
    "sha256_bytes",
    "sha256_file",
    "source_fingerprint",
    "validate_complete_attempt",
]
