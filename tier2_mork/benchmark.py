"""
Holistic Benchmarking Suite for Tier 2 MORK HNSW Indexing & Retrieval Engine.
Measures latency (p50/p95/p99), throughput (QPS), index construction rates, and Recall@m.
"""

from typing import Any, Dict, List, Tuple
import logging
import time
import numpy as np

from tier2_mork.store import MORKTemplateStore, TemplateRecord

logger = logging.getLogger(__name__)


class MORKBenchmarkSuite:
    """
    Benchmarking harness for evaluating Tier 2 MORK CPU performance.
    """

    def __init__(self, dim: int = 256, metric: str = "cosine") -> None:
        self.dim = dim
        self.metric = metric

    def generate_synthetic_templates(
        self, num_templates: int
    ) -> List[TemplateRecord]:
        """Generate synthetic Atomese template records for indexing benchmarks."""
        np.random.seed(42)
        records = []
        for i in range(num_templates):
            key = np.random.randn(self.dim).astype(np.float32)
            value = np.random.randn(self.dim).astype(np.float32)
            if self.metric == "cosine":
                key /= np.linalg.norm(key)
                value /= np.linalg.norm(value)

            rec = TemplateRecord(
                template_id=i + 1,
                metta_ast=f"(EvaluationLink (Predicate 'p_{i}') (List (Concept '$X')))",
                key_embedding=key,
                value_embedding=value,
                category=f"cat_{i % 5}",
            )
            records.append(rec)
        return records

    def run_full_benchmark(
        self,
        num_templates: int = 10000,
        num_queries: int = 1000,
        top_m: int = 8,
        ef_search_list: List[int] = [16, 32, 50, 100],
    ) -> Dict[str, Any]:
        """
        Run end-to-end index construction, latency, throughput, and Recall@m sweep.
        """
        logger.info(f"Starting MORK benchmark: {num_templates} templates, {num_queries} queries")

        # 1. Benchmark Index Construction
        templates = self.generate_synthetic_templates(num_templates)
        store = MORKTemplateStore(
            dim=self.dim, space=self.metric, max_capacity=num_templates + 1000
        )

        build_start = time.perf_counter()
        store.insert_batch(templates)
        build_time = time.perf_counter() - build_start
        build_rate = num_templates / build_time

        # 2. Benchmark Query Latency & Throughput Sweep across ef_search settings
        queries = np.random.randn(num_queries, self.dim).astype(np.float32)
        if self.metric == "cosine":
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        ef_results = {}
        for ef in ef_search_list:
            store.hnsw_index.set_query_ef(ef)
            latencies_ms = []

            start_sweep = time.perf_counter()
            for i in range(num_queries):
                q = queries[i : i + 1]
                t0 = time.perf_counter()
                store.retrieve_top_m(q, top_m=top_m)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            total_sweep_time = time.perf_counter() - start_sweep
            qps = num_queries / total_sweep_time

            ef_results[f"ef_{ef}"] = {
                "qps": qps,
                "mean_latency_ms": float(np.mean(latencies_ms)),
                "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
                "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
                "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
            }

        # 3. Calculate Recall@m against Brute Force Exact Search
        all_keys = np.vstack([t.key_embedding for t in templates])
        all_ids = np.array([t.template_id for t in templates])

        recall_samples = min(100, num_queries)
        exact_top_m_ids = []
        for i in range(recall_samples):
            q = queries[i]
            sims = np.dot(all_keys, q)
            top_exact_indices = np.argsort(-sims)[:top_m]
            exact_top_m_ids.append(set(all_ids[top_exact_indices]))

        store.hnsw_index.set_query_ef(50)
        hnsw_hits = 0
        for i in range(recall_samples):
            matched_recs, _, _, _ = store.retrieve_top_m(queries[i : i + 1], top_m=top_m)
            hnsw_ids = set([r.template_id for r in matched_recs])
            hnsw_hits += len(exact_top_m_ids[i].intersection(hnsw_ids))

        recall_at_m = hnsw_hits / (recall_samples * top_m)

        results = {
            "num_templates": num_templates,
            "num_queries": num_queries,
            "dim": self.dim,
            "metric": self.metric,
            "build_time_sec": build_time,
            "build_templates_per_sec": build_rate,
            "recall_at_m": recall_at_m,
            "query_benchmarks": ef_results,
        }

        logger.info(f"MORK Benchmark complete: Recall@{top_m}={recall_at_m:.4f}, p95_latency={ef_results['ef_50']['p95_latency_ms']:.3f} ms")
        return results
