"""Run artifacts shared by the stage runners; one fresh directory per run.
JSON is written atomically and ``metrics.jsonl`` is flushed per line, so a
crashed run never leaves a truncated file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import IO, Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _to_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json(v) for v in value]
    if isinstance(value, str | bool | int | float) or value is None:
        return value
    # jax.Array and np.ndarray both support np.asarray.
    arr = np.asarray(value)
    return arr.item() if arr.ndim == 0 else arr.tolist()


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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
    def __init__(self, root: Path | str, run_name: str) -> None:
        if not run_name or Path(run_name).name != run_name:
            raise ValueError(
                f"run_name must be a plain directory name, got {run_name!r}"
            )
        self.dir = Path(root) / run_name
        # Never mix artifacts from two runs.
        self.dir.mkdir(parents=True, exist_ok=False)
        self._metrics: IO[str] | None = (self.dir / "metrics.jsonl").open("a")

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._metrics is not None:
            self._metrics.close()
            self._metrics = None

    def write_config(self, config: Mapping[str, Any]) -> Path:
        path = self.dir / "config.yaml"
        _atomic_write_text(path, yaml.safe_dump(_to_json(config), sort_keys=False))
        return path

    def write_json(self, name: str, data: Mapping[str, Any]) -> Path:
        path = self.dir / name
        _atomic_write_text(path, json.dumps(_to_json(data), indent=2) + "\n")
        return path

    def log(self, event: str, **fields: Any) -> None:
        if self._metrics is None:
            raise RuntimeError("RunWriter is closed.")
        record = {"event": event, **_to_json(fields)}
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()

    def save_params(self, tag: str, params: Mapping[str, Mapping[str, Any]]) -> Path:
        flat = {
            f"{outer}/{inner}": np.asarray(leaf)
            for outer, leaves in params.items()
            for inner, leaf in leaves.items()
        }
        path = self.dir / "params" / f"{tag}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **flat)
        return path

    def save_checkpoint(
        self,
        tag: str,
        params: Mapping[str, Mapping[str, Any]],
        opt_state: Any | None = None,
        **metadata: Any,
    ) -> Path:
        """Save adapter params, optax optimizer state, and training metadata."""
        import pickle

        import jax

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
            flat["__metadata__"] = np.array(json.dumps(_to_json(metadata)))

        path = self.dir / "checkpoints" / f"{tag}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **flat)
        # Also save standalone weights for backwards compatibility.
        self.save_params(tag, params)
        return path


def load_params(path: Path | str) -> dict[str, dict[str, np.ndarray]]:
    params: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path) as data:
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
) -> tuple[dict[str, dict[str, np.ndarray]], Any | None, dict[str, Any]]:
    """Load adapter params, optax optimizer state, and metadata from a checkpoint."""
    import pickle

    import jax

    params: dict[str, dict[str, np.ndarray]] = {}
    opt_leaves: dict[int, np.ndarray] = {}
    treedef = None
    metadata: dict[str, Any] = {}

    with np.load(path) as data:
        for key in data.files:
            if key == "__opt_treedef__":
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
    "RunWriter",
    "default_run_name",
    "load_checkpoint",
    "load_params",
    "run_metadata",
]
