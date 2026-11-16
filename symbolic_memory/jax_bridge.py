"""Ordered host retrieval for compiled JAX execution with fixed bank values."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

from symbolic_memory.service import RetrievalClient


class RetrievedTemplates(NamedTuple):
    """Static-shaped arrays; indices refer to the retriever's full ID registry."""

    keys: jax.Array
    values: jax.Array
    scores: jax.Array
    template_indices: jax.Array


class JaxRetriever:
    """Execute runtime TCP retrieval through an ordered JAX I/O callback.

    Selection receives a stopped-gradient query; attention can differentiate
    through the original query. Retrieved vectors are constant bank data.
    Service failures propagate to the compiled caller.

    Construct the retriever and persistent client outside JIT. Query batches
    are passed directly; ordered callbacks do not support vmap. Shapes and
    top_m are static compilation inputs. Device placement follows the caller's
    JAX execution context.
    """

    def __init__(
        self,
        client: RetrievalClient,
        template_ids: Sequence[str],
        *,
        top_m: int = 8,
    ) -> None:
        identifiers = tuple(template_ids)
        if (
            len(identifiers) != client.info["record_count"]
            or any(not isinstance(item, str) or not item for item in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("Provide one unique full ID for every bank record")
        if (
            isinstance(top_m, bool)
            or not isinstance(top_m, int)
            or not 1 <= top_m <= client.info["max_top_m"]
        ):
            raise ValueError("top_m must be within the service limits")
        self.template_ids = identifiers
        self._indices = {identifier: i for i, identifier in enumerate(identifiers)}
        self._client = client
        self._top_m = top_m
        self._matches = min(top_m, len(identifiers))
        self._dimension = client.info["vector_dimension"]

    def _retrieve(
        self, queries: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        result = self._client.query_top_k(np.asarray(queries), self._top_m)
        try:
            indices = np.asarray(
                [
                    [self._indices[identifier] for identifier in row]
                    for row in result.template_ids
                ],
                dtype=np.int32,
            )
        except KeyError as error:
            raise ValueError(
                "Service returned an ID absent from the bank registry"
            ) from error
        return result.keys, result.values, result.scores, indices

    def __call__(self, queries: jax.Array) -> RetrievedTemplates:
        """Return [batch, matches, dimension] K/V and [batch, matches] metadata."""
        if queries.dtype != jnp.float32 or queries.ndim not in (1, 2):
            raise ValueError("Queries must be rank-1 or rank-2 float32 JAX arrays")
        if queries.ndim == 1:
            queries = queries[None, :]
        count, dimension = queries.shape
        if dimension != self._dimension:
            raise ValueError(f"Query dimension must be {self._dimension}")
        if not 1 <= count <= self._client.info["max_queries"]:
            raise ValueError("Query batch size must be within the service limits")
        vectors = jax.ShapeDtypeStruct((count, self._matches, dimension), jnp.float32)
        scores = jax.ShapeDtypeStruct((count, self._matches), jnp.float32)
        indices = jax.ShapeDtypeStruct((count, self._matches), jnp.int32)
        result = io_callback(
            self._retrieve,
            (vectors, vectors, scores, indices),
            jax.lax.stop_gradient(queries),
            ordered=True,
        )
        return RetrievedTemplates(*result)
