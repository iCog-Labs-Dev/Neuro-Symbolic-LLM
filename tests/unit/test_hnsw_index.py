"""
Unit tests for GenericHNSWIndex.
"""

import numpy as np
import pytest
from tier2_retrieval.index import GenericHNSWIndex


def test_hnsw_index_basic_query():
    dim = 64
    index = GenericHNSWIndex(dim=dim, space="cosine", max_elements=1000)

    np.random.seed(42)
    keys = np.random.randn(100, dim).astype(np.float32)
    ids = list(range(1, 101))

    index.add_items(keys, ids)
    assert index.count() == 100

    query_vec = keys[0]
    labels, distances = index.query(query_vec, top_m=5)

    assert labels.shape == (1, 5)
    assert labels[0, 0] == 1  # The top match for keys[0] should be itself (ID 1)
    assert distances[0, 0] < 1e-4  # Cosine distance to itself should be near zero


def test_hnsw_index_resize():
    dim = 32
    index = GenericHNSWIndex(dim=dim, max_elements=10)
    keys = np.random.randn(25, dim).astype(np.float32)
    ids = list(range(1, 26))

    index.add_items(keys, ids)
    assert index.count() == 25
