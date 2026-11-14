"""Eager CUDA transfer boundary for the TCP retrieval client."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import TYPE_CHECKING

from symbolic_memory.service import RetrievalClient

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class CudaQueryResult:
    """Retrieved arrays on the query device and synchronized wall-clock stages."""

    keys: Tensor
    values: Tensor
    scores: Tensor
    template_ids: list[list[str]]
    device_to_host_ms: float
    retrieval_round_trip_ms: float
    host_to_device_ms: float
    total_ms: float


def retrieve_cuda(
    client: RetrievalClient, queries: Tensor, top_m: int = 8
) -> CudaQueryResult:
    """Retrieve from a detached host query and return results to its CUDA device.

    Retrieval is outside autograd; the original query remains available for
    differentiable attention. Timings include allocation and conversion.
    Initial synchronization excludes pending producer work from total_ms.
    """
    import torch

    if not isinstance(queries, torch.Tensor) or queries.device.type != "cuda":
        raise ValueError("queries must be a CUDA tensor; CPU fallback is not supported")
    if queries.dtype != torch.float32 or queries.ndim not in (1, 2):
        raise ValueError("queries must be a rank-1 or rank-2 float32 CUDA tensor")
    device = queries.device
    torch.cuda.synchronize(device)
    start = perf_counter_ns()
    host_queries = queries.detach().cpu().numpy()
    copied_to_host = perf_counter_ns()
    result = client.query_top_k(host_queries, top_m=top_m)
    received = perf_counter_ns()
    keys = torch.from_numpy(result.keys).to(device)
    values = torch.from_numpy(result.values).to(device)
    scores = torch.from_numpy(result.scores).to(device)
    torch.cuda.synchronize(device)
    finished = perf_counter_ns()
    return CudaQueryResult(
        keys=keys,
        values=values,
        scores=scores,
        template_ids=result.template_ids,
        device_to_host_ms=(copied_to_host - start) / 1_000_000,
        retrieval_round_trip_ms=(received - copied_to_host) / 1_000_000,
        host_to_device_ms=(finished - received) / 1_000_000,
        total_ms=(finished - start) / 1_000_000,
    )
