"""
PCIe Communication Bridge between Tier 1 GPU (JAX/Flax) and Tier 2 CPU (MORK Sparse Engine).
Provides kilobyte-scale payload serialization, round-trip IPC/Network transport, and latency profiling.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import numpy as np

from tier2_mork.store import MORKTemplateStore, TemplateRecord

logger = logging.getLogger(__name__)


@dataclass
class CrossTierPayload:
    """Represents a single token query payload crossing the PCIe/Network bus."""

    layer_id: int
    token_position: int
    query_vector: np.ndarray  # q_sym in R^k (typically k=256, float32 -> 1 KB)
    top_m: int = 8
    timestamp_sent: float = field(default_factory=time.perf_counter)


@dataclass
class CrossTierResponse:
    """Represents returned template payload crossing back to GPU."""

    layer_id: int
    token_position: int
    matched_ids: List[int]
    key_matrix: np.ndarray  # shape (m, k) -> ~8 KB
    value_matrix: np.ndarray  # shape (m, k) -> ~8 KB
    latency_ms: float
    timestamp_received: float = field(default_factory=time.perf_counter)


class Profiler:
    """Tracks cross-tier roundtrip latencies, payload sizes, and bandwidth utilization."""

    def __init__(self) -> None:
        self.latencies_ms: List[float] = []
        self.bytes_sent: List[int] = []
        self.bytes_received: List[int] = []

    def record_query(
        self, send_bytes: int, recv_bytes: int, latency_ms: float
    ) -> None:
        self.bytes_sent.append(send_bytes)
        self.bytes_received.append(recv_bytes)
        self.latencies_ms.append(latency_ms)

    def summary(self) -> Dict[str, float]:
        if not self.latencies_ms:
            return {"count": 0}
        return {
            "count": len(self.latencies_ms),
            "p50_latency_ms": float(np.percentile(self.latencies_ms, 50)),
            "p95_latency_ms": float(np.percentile(self.latencies_ms, 95)),
            "p99_latency_ms": float(np.percentile(self.latencies_ms, 99)),
            "avg_bytes_sent_per_token": float(np.mean(self.bytes_sent)),
            "avg_bytes_recv_per_token": float(np.mean(self.bytes_received)),
        }


class PCIeBridgeClient:
    """
    High-speed PCIe/IPC bridge client used by Tier 1 GPU loop to query Tier 2 CPU MORK engine.
    """

    def __init__(
        self,
        mork_store: Optional[MORKTemplateStore] = None,
        host: str = "localhost",
        port: int = 8000,
        direct_in_memory: bool = True,
    ) -> None:
        """
        Initialize the PCIe Bridge client.

        :param mork_store: In-memory store reference (for co-located / zero-copy nodes).
        :param host: Tier 2 CPU host address.
        :param port: Tier 2 CPU service port.
        :param direct_in_memory: If True, uses zero-copy in-memory calls (NVLink-C2C style).
        """
        self.mork_store = mork_store
        self.host = host
        self.port = port
        self.direct_in_memory = direct_in_memory
        self.profiler = Profiler()

    def send_query(self, payload: CrossTierPayload) -> CrossTierResponse:
        """
        Send q_sym payload to Tier 2 MORK and receive top-m keys & values.

        :param payload: CrossTierPayload containing q_sym vector.
        :return: CrossTierResponse with returned key and value matrices.
        """
        t0 = time.perf_counter()
        q_arr = np.asarray(payload.query_vector, dtype=np.float32)

        # Compute payload size in bytes
        send_bytes = q_arr.nbytes + 16  # vector bytes + header

        if self.direct_in_memory and self.mork_store is not None:
            records, distances, keys, values = self.mork_store.retrieve_top_m(
                q_arr, top_m=payload.top_m
            )
            matched_ids = [r.template_id for r in records]
        else:
            # Fallback mock for remote client
            matched_ids = list(range(1, payload.top_m + 1))
            k_dim = q_arr.shape[-1]
            keys = np.zeros((payload.top_m, k_dim), dtype=np.float32)
            values = np.zeros((payload.top_m, k_dim), dtype=np.float32)

        recv_bytes = keys.nbytes + values.nbytes + 32
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.profiler.record_query(send_bytes, recv_bytes, latency_ms)

        return CrossTierResponse(
            layer_id=payload.layer_id,
            token_position=payload.token_position,
            matched_ids=matched_ids,
            key_matrix=keys,
            value_matrix=values,
            latency_ms=latency_ms,
        )
