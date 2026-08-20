"""
Unit tests for TemplateStore.
"""

import numpy as np
import pytest
from tier2_retrieval.store import TemplateStore, TemplateRecord


def test_template_store_insert_and_retrieve():
    dim = 128
    store = TemplateStore(dim=dim, max_capacity=500)

    t1 = TemplateRecord(
        template_id=1,
        metta_ast="(EvaluationLink (Predicate 'p1'))",
        key_embedding=np.random.randn(dim).astype(np.float32),
        value_embedding=np.random.randn(dim).astype(np.float32),
    )
    t2 = TemplateRecord(
        template_id=2,
        metta_ast="(EvaluationLink (Predicate 'p2'))",
        key_embedding=np.random.randn(dim).astype(np.float32),
        value_embedding=np.random.randn(dim).astype(np.float32),
    )

    store.insert_batch([t1, t2])
    assert store.size() == 2

    records, dists, keys, values = store.retrieve_top_m(t1.key_embedding, top_m=2)
    assert len(records) == 2
    assert records[0].template_id == 1
    assert keys.shape == (2, dim)
    assert values.shape == (2, dim)
