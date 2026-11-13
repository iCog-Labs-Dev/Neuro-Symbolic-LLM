"""Run artifacts shared by the stage runners; one fresh directory per run.
JSON and NumPy archives are written atomically and ``metrics.jsonl`` is flushed
per line, so a crashed run never leaves a truncated file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from types import TracebackType
from typing import IO, Any

import jax
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_INDEX_NAME = "artifacts.json"
COMPLETE_MARKER = "COMPLETE"
FAILED_MARKER = "FAILED"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    # jax.Array and np.ndarray both support np.asarray.
    arr = np.asarray(value)
    converted = arr.item() if arr.ndim == 0 else arr.tolist()
    if type(converted) is type(value):
        raise TypeError(
            f"Value of type {type(value).__name__} is not JSON serializable"
        )
    return _to_json(converted)


def _json_dumps(
    value: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        _to_json(value),
        allow_nan=False,
        indent=indent,
        sort_keys=sort_keys,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def file_sha256(path: Path | str) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hash_path(path: Path | str) -> Path:
    """Return the sidecar path used for a checkpoint's SHA-256 digest."""
    return Path(f"{Path(path)}.sha256")


def write_checkpoint_hash(path: Path | str) -> Path:
    """Atomically write a SHA-256 sidecar for a checkpoint."""
    checkpoint = Path(path)
    sidecar = checkpoint_hash_path(checkpoint)
    _atomic_write_text(sidecar, file_sha256(checkpoint) + "\n")
    return sidecar


def verify_checkpoint_hash(path: Path | str) -> bool:
    """Return whether a checkpoint matches its SHA-256 sidecar."""
    checkpoint = Path(path)
    try:
        fields = checkpoint_hash_path(checkpoint).read_text(encoding="utf-8").split()
        expected = fields[0].lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            return False
        return hmac.compare_digest(file_sha256(checkpoint), expected)
    except (IndexError, OSError):
        return False


def artifact_index(run_dir: Path | str) -> dict[str, dict[str, int | str]]:
    """Index regular run artifacts by relative path, size, and SHA-256."""
    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")

    ignored = {ARTIFACT_INDEX_NAME, COMPLETE_MARKER, FAILED_MARKER}
    artifacts: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if len(relative.parts) == 1 and relative.name in ignored:
            continue
        if path.is_symlink():
            raise ValueError(f"Run artifacts must not be symlinks: {path}")
        if path.is_file():
            artifacts[relative.as_posix()] = {
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return artifacts


def write_artifact_index(run_dir: Path | str) -> Path:
    """Write the current run artifact index and return its path."""
    root = Path(run_dir)
    path = root / ARTIFACT_INDEX_NAME
    payload = {"version": 1, "artifacts": artifact_index(root)}
    _atomic_write_text(
        path,
        _json_dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return path


def verify_artifact_index(run_dir: Path | str) -> bool:
    """Return whether every indexed artifact still has the recorded hash."""
    root = Path(run_dir)
    try:
        payload = json.loads((root / ARTIFACT_INDEX_NAME).read_text(encoding="utf-8"))
        if type(payload.get("version")) is not int or payload["version"] != 1:
            return False
        recorded = payload.get("artifacts")
        if not isinstance(recorded, dict):
            return False
        return recorded == artifact_index(root)
    except (OSError, TypeError, ValueError):
        return False


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _resolve_archive_path(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_file():
        candidate = p if p.suffix == ".npz" else p.with_suffix(".npz")
        if candidate.is_file():
            return candidate
        # Account for an interrupted pair of writes (checkpoint and companion params):
        if candidate.parent.name == "params":
            companion = candidate.parent.parent / "checkpoints" / candidate.name
            if companion.is_file():
                return companion
        elif candidate.parent.name == "checkpoints":
            companion = candidate.parent.parent / "params" / candidate.name
            if companion.is_file():
                return companion
    return p


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def run_metadata() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("jax", "jaxlib", "optax", "numpy", "torch", "transformers"):
        module = sys.modules.get(name)
        version = getattr(module, "__version__", None)
        if version is not None:
            versions[name] = str(version)
    status = _git("status", "--porcelain")
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "versions": versions,
    }


def default_run_name(prefix: str) -> str:
    """Return ``<prefix>-YYYYmmdd-HHMMSS`` in UTC."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


class RunWriter:
    def __init__(
        self,
        root: Path | str,
        run_name: str,
        *,
        exact: bool = False,
    ) -> None:
        if (
            not run_name
            or run_name in {".", ".."}
            or "/" in run_name
            or "\\" in run_name
            or Path(run_name).name != run_name
        ):
            raise ValueError(
                f"run_name must be a plain directory name, got {run_name!r}"
            )

        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        if not root_path.is_dir():
            raise ValueError(f"Run root is not a directory: {root_path}")

        self.requested_dir = root_path / run_name
        self.dir = self._claim_directory(self.requested_dir, exact=exact)
        (self.dir / "checkpoints").mkdir()
        (self.dir / "params").mkdir()
        self._metrics: IO[str] | None = (self.dir / "metrics.jsonl").open(
            "x", encoding="utf-8"
        )

    @staticmethod
    def _claim_directory(requested: Path, *, exact: bool) -> Path:
        suffix = 0
        while True:
            candidate = (
                requested
                if suffix == 0
                else requested.with_name(f"{requested.name}-{suffix}")
            )
            try:
                candidate.mkdir()
            except FileExistsError:
                if exact:
                    raise
                suffix += 1
                continue
            return candidate

    @classmethod
    def for_directory(cls, directory: Path | str) -> RunWriter:
        """Create exactly ``directory``, failing if any path already occupies it."""
        path = Path(directory)
        if not path.name:
            raise ValueError(f"Run directory must have a name, got {path}")
        return cls(path.parent, path.name, exact=True)

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.mark_complete()
            return
        try:
            self.mark_failed(exc)
        except Exception as marker_error:
            if exc is not None:
                exc.add_note(f"Could not write FAILED marker: {marker_error}")

    def close(self) -> None:
        if self._metrics is not None:
            self._metrics.close()
            self._metrics = None

    def write_config(self, config: Mapping[str, Any]) -> Path:
        path = self.dir / "config.yaml"
        _atomic_write_text(path, yaml.safe_dump(_to_json(config), sort_keys=False))
        return path

    def write_json(self, name: str, data: Mapping[str, Any]) -> Path:
        relative = Path(name)
        if (
            not name
            or relative.is_absolute()
            or relative == Path(".")
            or ".." in relative.parts
        ):
            raise ValueError(
                f"Artifact name must stay within the run directory: {name!r}"
            )
        path = self.dir / relative
        _atomic_write_text(path, _json_dumps(data, indent=2) + "\n")
        return path

    def log(self, event: str, **fields: Any) -> None:
        if self._metrics is None:
            raise RuntimeError("RunWriter is closed.")
        record = {"event": event, **_to_json(fields)}
        self._metrics.write(_json_dumps(record) + "\n")
        self._metrics.flush()

    def save_params(self, tag: str, params: Mapping[str, Mapping[str, Any]]) -> Path:
        self._validate_tag(tag)
        flat = {
            f"{outer}/{inner}": np.asarray(leaf)
            for outer, leaves in params.items()
            for inner, leaf in leaves.items()
        }
        path = self.dir / "params" / f"{tag}.npz"
        _atomic_savez(path, **flat)
        return path

    def save_checkpoint(
        self,
        tag: str,
        params: Mapping[str, Mapping[str, Any]],
        opt_state: Any | None = None,
        **metadata: Any,
    ) -> Path:
        """Save adapter params, optax optimizer state, and training metadata."""
        self._validate_tag(tag)
        flat: dict[str, Any] = {
            f"params/{outer}/{inner}": np.asarray(leaf)
            for outer, leaves in params.items()
            for inner, leaf in leaves.items()
        }
        if opt_state is not None:
            leaves, treedef = jax.tree_util.tree_flatten(opt_state)
            for i, leaf in enumerate(leaves):
                flat[f"opt/leaf_{i}"] = np.asarray(leaf)
            flat["__opt_treedef__"] = np.frombuffer(
                pickle.dumps(treedef), dtype=np.uint8
            )
        if metadata:
            flat["__metadata__"] = np.array(_json_dumps(metadata))

        path = self.dir / "checkpoints" / f"{tag}.npz"
        _atomic_savez(path, **flat)
        write_checkpoint_hash(path)
        # Also save standalone weights for backwards compatibility.
        self.save_params(tag, params)
        return path

    @staticmethod
    def _validate_tag(tag: str) -> None:
        if (
            not tag
            or tag in {".", ".."}
            or "/" in tag
            or "\\" in tag
            or Path(tag).name != tag
        ):
            raise ValueError(f"tag must be a plain file name, got {tag!r}")

    def artifact_index(self) -> dict[str, dict[str, int | str]]:
        """Return hashes for this run's current artifacts."""
        if self._metrics is not None:
            self._metrics.flush()
        return artifact_index(self.dir)

    def write_artifact_index(self) -> Path:
        """Write hashes for this run's current artifacts."""
        if self._metrics is not None:
            self._metrics.flush()
        return write_artifact_index(self.dir)

    def _mark(self, status: str, error: BaseException | str | None = None) -> Path:
        self.close()
        marker = self.dir / status
        opposite_name = FAILED_MARKER if status == COMPLETE_MARKER else COMPLETE_MARKER
        opposite = self.dir / opposite_name
        if opposite.exists() or opposite.is_symlink():
            raise RuntimeError(f"Run is already marked {opposite_name}: {self.dir}")
        if marker.exists():
            return marker

        index_path = self.write_artifact_index()
        payload: dict[str, Any] = {
            "status": status,
            "finished_at": _utc_now(),
            "artifact_index": index_path.name,
        }
        if isinstance(error, BaseException):
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        elif error is not None:
            payload["error"] = {"message": str(error)}
        _atomic_write_text(marker, _json_dumps(payload, indent=2) + "\n")
        return marker

    def mark_complete(self) -> Path:
        """Finalize the artifact index and mark the run successful."""
        return self._mark(COMPLETE_MARKER)

    def mark_failed(self, error: BaseException | str | None = None) -> Path:
        """Finalize the artifact index and mark the run failed."""
        return self._mark(FAILED_MARKER, error)


def load_params(path: Path | str) -> dict[str, dict[str, np.ndarray]]:
    resolved_path = _resolve_archive_path(path)
    params: dict[str, dict[str, np.ndarray]] = {}
    with np.load(resolved_path) as data:
        for key in data.files:
            if key.startswith("__"):
                continue
            if key.startswith("params/"):
                _, outer, inner = key.split("/", 2)
                params.setdefault(outer, {})[inner] = data[key]
            elif "/" in key and not key.startswith("opt/"):
                outer, inner = key.split("/", 1)
                params.setdefault(outer, {})[inner] = data[key]
    return params


def load_checkpoint(
    path: Path | str,
    *,
    trusted: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], Any | None, dict[str, Any]]:
    """Load adapter params, optax optimizer state, and training metadata.

    Parameters
    ----------
    path:
        Path to the ``.npz`` checkpoint file.
    trusted:
        Whether the checkpoint is from a trusted source. Defaults to ``False``.
        When ``False``, refusing to deserialize ``__opt_treedef__`` prevents
        arbitrary code execution from untrusted pickle payloads. Set to ``True``
        only when loading checkpoints from trusted sources. For untrusted
        checkpoints where only weights are needed, use :func:`load_params`.

    Raises
    ------
    ValueError:
        If the checkpoint contains pickled optimizer state (``__opt_treedef__``)
        and ``trusted`` is ``False``.
    """
    resolved_path = _resolve_archive_path(path)
    params: dict[str, dict[str, np.ndarray]] = {}
    opt_leaves: dict[int, np.ndarray] = {}
    treedef = None
    metadata: dict[str, Any] = {}

    with np.load(resolved_path) as data:
        for key in data.files:
            if key == "__opt_treedef__":
                if not trusted:
                    raise ValueError(
                        f"Refusing to deserialize untrusted '__opt_treedef__' from {path}. "
                        "Unpickling untrusted checkpoints can lead to arbitrary code execution. "
                        "Pass `trusted=True` to `load_checkpoint` only if this checkpoint is from "
                        "a trusted source. To load parameter weights safely, use `load_params`."
                    )
                treedef = pickle.loads(data[key].tobytes())
            elif key == "__metadata__":
                metadata = json.loads(str(data[key]))
            elif key.startswith("params/"):
                _, outer, inner = key.split("/", 2)
                params.setdefault(outer, {})[inner] = data[key]
            elif key.startswith("opt/leaf_"):
                idx = int(key.split("opt/leaf_")[1])
                opt_leaves[idx] = data[key]
            elif "/" in key and not key.startswith("opt/"):
                outer, inner = key.split("/", 1)
                params.setdefault(outer, {})[inner] = data[key]

    opt_state = None
    if treedef is not None and opt_leaves:
        ordered_leaves = [opt_leaves[i] for i in range(len(opt_leaves))]
        opt_state = jax.tree_util.tree_unflatten(treedef, ordered_leaves)

    return params, opt_state, metadata


__all__ = [
    "ARTIFACT_INDEX_NAME",
    "COMPLETE_MARKER",
    "FAILED_MARKER",
    "RunWriter",
    "artifact_index",
    "checkpoint_hash_path",
    "default_run_name",
    "file_sha256",
    "load_checkpoint",
    "load_params",
    "run_metadata",
    "verify_artifact_index",
    "verify_checkpoint_hash",
    "write_artifact_index",
    "write_checkpoint_hash",
]
