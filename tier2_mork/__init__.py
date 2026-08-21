"""Tier 2 MORK Sparse Symbolic Engine Package.

Provides pluggable MorkClient interface supporting Dockerized MORK server (via HTTP/gRPC API)
and local embedded snapshot backends (PathMap / HNSW), alongside a clean SymbolicHeadBridge contract
for Tier 1 colleagues.
"""

from tier2_mork.bridge import SymbolicHeadBridge
from tier2_mork.client import (
    DockerMorkClient,
    LocalHNSWClient,
    MmapPathMapClient,
    MorkClient,
    MorkQueryResult,
    get_mork_client,
)

__all__ = [
    "MorkClient",
    "MorkQueryResult",
    "DockerMorkClient",
    "LocalHNSWClient",
    "MmapPathMapClient",
    "get_mork_client",
    "SymbolicHeadBridge",
]
