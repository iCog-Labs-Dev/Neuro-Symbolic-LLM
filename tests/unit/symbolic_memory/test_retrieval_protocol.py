"""TCP retrieval message framing and resource limits."""

import json

import numpy as np
import pytest

from residual.symbolic_head.contracts.template_record import MorkQueryResult
from symbolic_memory.protocol import (
    HEADER,
    LENGTH,
    MAX_MESSAGE_BYTES,
    encode_info,
    encode_query,
    encode_result,
    query_shape,
)


def test_query_carries_requested_top_m_and_little_endian_vectors():
    queries = np.array([[1, -0.0, 2], [3, 4, 5]], dtype=np.float32)
    packet = encode_query(queries, top_m=7)
    assert HEADER.unpack(packet[: HEADER.size]) == (b"QSYM", 2, 2, 3, 7)
    assert query_shape(packet[: HEADER.size], 3, 16, 8) == (2, 3, 7)
    assert packet[HEADER.size :] == queries.astype("<f4").tobytes()


@pytest.mark.parametrize(
    "version,count,width,top_m",
    [
        (1, 1, 3, 8),
        (2, 0, 3, 8),
        (2, 17, 3, 8),
        (2, 1, 4, 8),
        (2, 1, 3, 0),
        (2, 1, 3, 9),
    ],
)
def test_invalid_headers_are_rejected_before_payload_read(version, count, width, top_m):
    with pytest.raises(ValueError):
        query_shape(HEADER.pack(b"QSYM", version, count, width, top_m), 3, 16, 8)


def test_payload_size_is_bounded_from_header():
    with pytest.raises(ValueError, match="transport limit"):
        query_shape(
            HEADER.pack(b"QSYM", 2, MAX_MESSAGE_BYTES, 3, 1), 3, MAX_MESSAGE_BYTES, 1
        )


@pytest.mark.parametrize("query", [[], [[1], [2, 3]], [float("nan")], [float("inf")]])
def test_invalid_queries_are_rejected(query):
    with pytest.raises(ValueError):
        encode_query(query, 1)


@pytest.mark.parametrize("top_m", [True, 0, -1, 1.5])
def test_invalid_top_m_is_rejected(top_m):
    with pytest.raises(ValueError):
        encode_query(np.ones(3), top_m)


def test_result_layout_preserves_original_vectors_scores_and_full_unicode_ids():
    result = MorkQueryResult(
        keys=np.array([[[3, -0.0]]], dtype=np.float32),
        values=np.array([[[0.1, 4]]], dtype=np.float32),
        scores=np.array([[0.75]], dtype=np.float32),
        template_ids=[["pattern-" + "a" * 64 + "-réseau"]],
    )
    packet = encode_result(result)
    assert HEADER.unpack(packet[: HEADER.size]) == (b"TMPL", 2, 1, 1, 2)
    numeric = result.keys.tobytes() + result.values.tobytes() + result.scores.tobytes()
    assert packet[HEADER.size : HEADER.size + len(numeric)] == numeric
    offset = HEADER.size + len(numeric)
    size = LENGTH.unpack(packet[offset : offset + LENGTH.size])[0]
    assert packet[offset + LENGTH.size :] == result.template_ids[0][0].encode("utf-8")
    assert size == len(packet) - offset - LENGTH.size


def test_bank_descriptor_is_length_prefixed_and_versioned():
    info = {"protocol_version": 2, "input_snapshot_sha256": "a" * 64}
    packet = encode_info(info)
    assert packet[:4] == b"BANK"
    assert LENGTH.unpack(packet[4:8])[0] == len(packet) - 8
    assert json.loads(packet[8:]) == info
