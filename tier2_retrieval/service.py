"""
Fast Top-m Service API & Service Engine for Tier 2 CPU Node Deployment.
Provides both direct in-memory calls and REST/gRPC endpoints for cross-tier PCIe communication.
"""

from typing import Any, Dict, List, Optional
import logging
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field

from tier2_retrieval.store import TemplateStore, TemplateRecord

logger = logging.getLogger(__name__)


class QueryPayload(BaseModel):
    """Pydantic model for cross-tier query request from Tier 1 GPU."""

    query_vector: List[float] = Field(
        ..., description="Projected query vector q_sym in R^k"
    )
    top_m: int = Field(8, description="Number of templates to retrieve")
    layer_id: Optional[int] = Field(None, description="Requesting layer ID")


class TemplateInsertPayload(BaseModel):
    """Pydantic model for inserting new templates from Tier 3 Async Consolidation."""

    template_id: int
    metta_ast: str
    key_embedding: List[float]
    value_embedding: List[float]
    category: str = "general"


class QueryResponse(BaseModel):
    """Pydantic model for top-m retrieval response sent back to Tier 1 GPU."""

    matched_ids: List[int]
    distances: List[float]
    keys: List[List[float]]  # top_m x k key vectors
    values: List[List[float]]  # top_m x k value vectors
    latency_ms: float


class Tier2Service:
    """Service wrapper for hosting template store in memory or serving over network."""

    def __init__(self, store: Optional[TemplateStore] = None, dim: int = 256):
        self.store = store or TemplateStore(dim=dim)

    def query_direct(
        self, query_vector: np.ndarray, top_m: int = 8
    ) -> Dict[str, Any]:
        """In-memory direct query execution for ultra-low latency within same node."""
        import time

        start_time = time.perf_counter()
        records, distances, keys, values = self.store.retrieve_top_m(
            query_vector, top_m=top_m
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "matched_ids": [r.template_id for r in records],
            "distances": distances.tolist(),
            "keys": keys.tolist(),
            "values": values.tolist(),
            "latency_ms": latency_ms,
        }


def create_app(service: Optional[Tier2Service] = None) -> FastAPI:
    """Factory creating FastAPI application for Tier 2 CPU node deployment."""
    app = FastAPI(
        title="Tier 2 Symbolic Retrieval Engine API",
        version="0.1.0",
        description="Sub-millisecond Atomese template retrieval engine for SingularityNET PC-Residual LLM.",
    )
    t2_service = service or Tier2Service()

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "indexed_templates": t2_service.store.size(),
            "dim": t2_service.store.dim,
        }

    @app.post("/query", response_model=QueryResponse)
    def query_templates(payload: QueryPayload):
        q_arr = np.array(payload.query_vector, dtype=np.float32)
        if q_arr.shape[0] != t2_service.store.dim:
            raise HTTPException(
                status_code=400,
                detail=f"Dimension mismatch: expected {t2_service.store.dim}, got {q_arr.shape[0]}",
            )

        res = t2_service.query_direct(q_arr, top_m=payload.top_m)
        return QueryResponse(**res)

    @app.post("/insert")
    def insert_template(payload: TemplateInsertPayload):
        record = TemplateRecord(
            template_id=payload.template_id,
            metta_ast=payload.metta_ast,
            key_embedding=np.array(payload.key_embedding, dtype=np.float32),
            value_embedding=np.array(payload.value_embedding, dtype=np.float32),
            category=payload.category,
        )
        t2_service.store.insert_template(record)
        return {"status": "success", "template_id": payload.template_id}

    return app

