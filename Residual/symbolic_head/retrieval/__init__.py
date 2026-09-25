# ruff: noqa: N999 - The repository uses the capitalized ``Residual`` namespace.
"""CPU retrieval configuration and derived FAISS indexing."""

from Residual.symbolic_head.retrieval.config import (
    RetrievalConfig,
    load_retrieval_config,
)
from Residual.symbolic_head.retrieval.faiss_index import FaissTemplateIndex

__all__ = ["FaissTemplateIndex", "RetrievalConfig", "load_retrieval_config"]
