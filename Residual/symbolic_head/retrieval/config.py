"""Validated Q1 configuration for the CPU template-retrieval index."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RetrievalConfig:
    key_dim: int
    backend: str
    top_m: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int

    def __post_init__(self) -> None:
        if self.backend not in {"hnsw", "flat"}:
            raise ValueError("index_backend must be 'hnsw' or 'flat'")
        for name in (
            "key_dim",
            "top_m",
            "hnsw_m",
            "hnsw_ef_construction",
            "hnsw_ef_search",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.backend == "hnsw" and self.hnsw_ef_search < self.top_m:
            raise ValueError("hnsw_ef_search must be at least top_m")


def load_retrieval_config(path: Path | None = None) -> RetrievalConfig:
    """Read the repository's Tier 2 retrieval settings; reject incomplete config."""
    if path is None:
        path = Path(__file__).resolve().parents[3] / "configs" / "tiers.yaml"
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("tier2_cpu"), dict):
        raise ValueError("tier2_cpu configuration is required")
    tier = raw["tier2_cpu"]
    hnsw = tier.get("hnsw")
    if not isinstance(hnsw, dict):
        raise ValueError("tier2_cpu.hnsw configuration is required")
    if tier.get("space") != "cosine":
        raise ValueError("Tier 2 retrieval requires cosine space")
    try:
        return RetrievalConfig(
            key_dim=tier["dim"],
            backend=tier["index_backend"],
            top_m=tier["top_m"],
            hnsw_m=hnsw["M"],
            hnsw_ef_construction=hnsw["ef_construction"],
            hnsw_ef_search=hnsw["ef_query"],
        )
    except KeyError as error:
        raise ValueError(
            f"Missing Tier 2 retrieval setting: {error.args[0]}"
        ) from error
