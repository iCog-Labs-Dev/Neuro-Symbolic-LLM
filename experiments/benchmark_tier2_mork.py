"""Tier 2 MORK Sparse Symbolic Engine Benchmark Suite.

Evaluates:
1. Indexing Throughput (templates / sec).
2. HNSW Vector Similarity Search Latency (p50, p95, p99, mean in ms) & QPS.
3. Symbolic Head Bridge Forward Pass Scaling (per-token latency across batch sizes).
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass

import numpy as np

from tier2_mork.bridge import SymbolicHeadBridge
from tier2_mork.client import DockerMorkClient, MorkClient, get_mork_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VectorRetrievalMetrics:
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float


@dataclass
class BenchmarkSummary:
    client_backend: str
    num_templates: int
    num_queries: int
    key_dim: int
    hidden_dim: int
    top_m: int
    indexing_time_sec: float
    indexing_qps: float
    retrieval_metrics: VectorRetrievalMetrics
    bridge_batch_latencies_ms: dict[int, float]


def run_benchmark_suite(
    num_templates: int = 10000,
    num_queries: int = 1000,
    key_dim: int = 256,
    hidden_dim: int = 768,
    top_m: int = 8,
    use_docker: bool = False,
    server_url: str = "http://127.0.0.1:8000",
    warmup_queries: int = 20,
) -> BenchmarkSummary:
    """Execute performance benchmark sweep across indexing, search, and bridge components."""
    client: MorkClient = (
        DockerMorkClient(server_url=server_url, key_dim=key_dim)
        if use_docker
        else get_mork_client(key_dim=key_dim)
    )

    client_name = client.__class__.__name__
    rng = np.random.default_rng(42)

    logger.info("Starting Tier 2 MORK Performance Benchmark Suite")
    logger.info("Backend: %s | Templates: %d | Queries: %d | KeyDim: %d", client_name, num_templates, num_queries, key_dim)

    # 1. Indexing Throughput Benchmark
    start_insert = time.perf_counter()
    for i in range(num_templates):
        k_vec = rng.normal(0, 1, key_dim).astype(np.float32)
        v_vec = rng.normal(0, 1, key_dim).astype(np.float32)
        client.add_template(f"tpl_{i:06d}", f"(Concept (Node_{i}))", k_vec, v_vec)
    t_insert_sec = time.perf_counter() - start_insert
    indexing_qps = num_templates / t_insert_sec
    logger.info("Seeded %d templates in %.4f sec (%.1f templates/sec)", num_templates, t_insert_sec, indexing_qps)

    # 2. HNSW Retrieval Latency Benchmark
    query_vectors = rng.normal(0, 1, (num_queries + warmup_queries, key_dim)).astype(np.float32)

    # Warm-up phase
    for i in range(warmup_queries):
        _ = client.query_top_k(query_vectors[i : i + 1], top_m=top_m)

    latencies_ms: list[float] = []
    for i in range(warmup_queries, warmup_queries + num_queries):
        q = query_vectors[i : i + 1]
        t0 = time.perf_counter()
        _ = client.query_top_k(q, top_m=top_m)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    ret_metrics = VectorRetrievalMetrics(
        mean_latency_ms=float(np.mean(latencies_ms)),
        p50_latency_ms=float(np.percentile(latencies_ms, 50)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        p99_latency_ms=float(np.percentile(latencies_ms, 99)),
        throughput_qps=float(num_queries / (sum(latencies_ms) / 1000.0)),
    )

    logger.info(
        "Vector Search Latency: Mean=%.3f ms | p50=%.3f ms | p95=%.3f ms | QPS=%.1f",
        ret_metrics.mean_latency_ms,
        ret_metrics.p50_latency_ms,
        ret_metrics.p95_latency_ms,
        ret_metrics.throughput_qps,
    )

    # 3. Symbolic Head Bridge Forward Pass Latency Benchmark
    bridge = SymbolicHeadBridge(
        hidden_dim=hidden_dim, key_dim=key_dim, top_m=top_m, mork_client=client
    )

    batch_latencies: dict[int, float] = {}
    batch_sizes = [1, 4, 8, 16]
    for b in batch_sizes:
        h_in = rng.normal(0, 1, (b, 8, hidden_dim)).astype(np.float32)
        t0 = time.perf_counter()
        _ = bridge.forward(h_in)
        t_ms = (time.perf_counter() - t0) * 1000.0
        batch_latencies[b] = t_ms
        tokens = b * 8
        per_token_us = (t_ms * 1000.0) / tokens
        logger.info("Bridge Pass (Batch=%d, Tokens=%d): Total=%.3f ms | Per-token=%.2f µs", b, tokens, t_ms, per_token_us)

    return BenchmarkSummary(
        client_backend=client_name,
        num_templates=num_templates,
        num_queries=num_queries,
        key_dim=key_dim,
        hidden_dim=hidden_dim,
        top_m=top_m,
        indexing_time_sec=t_insert_sec,
        indexing_qps=indexing_qps,
        retrieval_metrics=ret_metrics,
        bridge_batch_latencies_ms=batch_latencies,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2 MORK Sparse Symbolic Engine Benchmark CLI")
    parser.add_argument("--num-templates", type=int, default=10000, help="Number of templates to index")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of query evaluations")
    parser.add_argument("--key-dim", type=int, default=256, help="Symbolic key/value dimension")
    parser.add_argument("--hidden-dim", type=int, default=768, help="Continuous hidden dimension")
    parser.add_argument("--top-m", type=int, default=8, help="Top-m retrieval count")
    parser.add_argument("--use-docker", action="store_true", help="Benchmark Docker container REST API")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000", help="MORK server URL")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON benchmark report")

    args = parser.parse_args()

    summary = run_benchmark_suite(
        num_templates=args.num_templates,
        num_queries=args.num_queries,
        key_dim=args.key_dim,
        hidden_dim=args.hidden_dim,
        top_m=args.top_m,
        use_docker=args.use_docker,
        server_url=args.server_url,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        logger.info("Saved benchmark report to %s", args.output_json)


if __name__ == "__main__":
    main()
