"""Bounded float32 messages for CPU template retrieval over TCP."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np

from residual.symbolic_head.contracts.template_record import MorkQueryResult

VERSION = 2
HEADER = struct.Struct("<4sIIII")
LENGTH = struct.Struct("<I")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_INFO_BYTES = 16 * 1024


def receive(connection: socket.socket, size: int) -> bytes:
    """Read exactly one bounded field, including across partial TCP reads."""
    if not 0 <= size <= MAX_MESSAGE_BYTES:
        raise ValueError("Message field exceeds the transport limit")
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = connection.recv_into(view[offset:])
        if count == 0:
            raise EOFError("Retrieval connection closed before the field completed")
        offset += count
    return bytes(data)


def encode_info(info: dict[str, Any]) -> bytes:
    payload = json.dumps(info, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_INFO_BYTES:
        raise ValueError("Bank descriptor exceeds the transport limit")
    return b"BANK" + LENGTH.pack(len(payload)) + payload


def read_info(connection: socket.socket) -> dict[str, Any]:
    if receive(connection, 4) != b"BANK":
        raise ValueError("Expected a retrieval bank descriptor")
    size = LENGTH.unpack(receive(connection, LENGTH.size))[0]
    if not 0 < size <= MAX_INFO_BYTES:
        raise ValueError("Invalid bank descriptor length")
    info = json.loads(receive(connection, size))
    if not isinstance(info, dict) or info.get("protocol_version") != VERSION:
        raise ValueError("Unsupported retrieval protocol version")
    return info


def query_shape(
    header: bytes, dimension: int, max_queries: int, max_top_m: int
) -> tuple[int, int, int]:
    magic, version, count, width, top_m = HEADER.unpack(header)
    if magic != b"QSYM" or version != VERSION:
        raise ValueError("Expected a QSYM version 2 request")
    if width != dimension:
        raise ValueError(f"Query dimension must be {dimension}")
    if not 1 <= count <= max_queries or not 1 <= top_m <= max_top_m:
        raise ValueError("Query batch size or top_m exceeds service limits")
    if count * width * 4 > MAX_MESSAGE_BYTES:
        raise ValueError("Query payload exceeds the transport limit")
    return count, width, top_m


def encode_query(queries: np.ndarray, top_m: int) -> bytes:
    with np.errstate(over="raise", invalid="raise"):
        array = np.asarray(queries, dtype="<f4")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError("Queries must be a nonempty finite float32 matrix")
    if isinstance(top_m, bool) or not isinstance(top_m, int) or not 1 <= top_m < 2**32:
        raise ValueError("top_m must be a positive uint32 integer")
    if array.nbytes > MAX_MESSAGE_BYTES:
        raise ValueError("Query payload exceeds the transport limit")
    payload: bytes = array.tobytes()
    return HEADER.pack(b"QSYM", VERSION, *array.shape, top_m) + payload


def encode_result(result: MorkQueryResult) -> bytes:
    count, matches, width = result.keys.shape
    if result.values.shape != result.keys.shape or result.scores.shape != (
        count,
        matches,
    ):
        raise ValueError("Inconsistent retrieval result shapes")
    if len(result.template_ids) != count or any(
        len(row) != matches for row in result.template_ids
    ):
        raise ValueError("Inconsistent retrieval ID counts")
    if any(
        not identifier or any(char in identifier for char in "\t\r\n")
        for row in result.template_ids
        for identifier in row
    ):
        raise ValueError("TCP template IDs must be nonempty and exclude tabs/newlines")
    arrays = [
        np.asarray(value, dtype="<f4")
        for value in (result.keys, result.values, result.scores)
    ]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Retrieval result contains nonfinite values")
    identifiers = "\n".join("\t".join(row) for row in result.template_ids).encode(
        "utf-8"
    )
    size = (
        HEADER.size
        + sum(array.nbytes for array in arrays)
        + LENGTH.size
        + len(identifiers)
    )
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("Retrieval result exceeds the transport limit")
    return b"".join(
        [
            HEADER.pack(b"TMPL", VERSION, count, matches, width),
            *(array.tobytes() for array in arrays),
            LENGTH.pack(len(identifiers)),
            identifiers,
        ]
    )


def read_result(
    connection: socket.socket, count: int, matches: int, width: int
) -> MorkQueryResult:
    magic, version, returned_count, returned_matches, returned_width = HEADER.unpack(
        receive(connection, HEADER.size)
    )
    if (magic, version, returned_count, returned_matches, returned_width) != (
        b"TMPL",
        VERSION,
        count,
        matches,
        width,
    ):
        raise ValueError("Retrieval response does not match the requested shape")
    size = count * matches * (2 * width + 1) * 4
    if HEADER.size + size + LENGTH.size > MAX_MESSAGE_BYTES:
        raise ValueError("Retrieval response exceeds the transport limit")
    arrays = np.frombuffer(receive(connection, size), dtype="<f4")
    if not np.isfinite(arrays).all():
        raise ValueError("Retrieval response contains nonfinite values")
    vector_size = count * matches * width
    id_size = LENGTH.unpack(receive(connection, LENGTH.size))[0]
    if HEADER.size + size + LENGTH.size + id_size > MAX_MESSAGE_BYTES:
        raise ValueError("Retrieval IDs exceed the transport limit")
    rows = receive(connection, id_size).decode("utf-8").split("\n")
    ids = [row.split("\t") if row else [] for row in rows]
    if len(ids) != count or any(
        len(row) != matches or any(not item or "\r" in item for item in row)
        for row in ids
    ):
        raise ValueError("Retrieval response has inconsistent template IDs")
    shape = (count, matches, width)
    return MorkQueryResult(
        keys=arrays[:vector_size].reshape(shape).copy(),
        values=arrays[vector_size : 2 * vector_size].reshape(shape).copy(),
        scores=arrays[2 * vector_size :].reshape(count, matches).copy(),
        template_ids=ids,
    )
