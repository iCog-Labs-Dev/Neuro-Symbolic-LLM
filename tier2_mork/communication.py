"""Binary wire format for Tier 2 query and template packets."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from tier2_mork.client import MorkClient, MorkQueryResult

_QUERY_MAGIC = b"QSYM"
_TEMPL_MAGIC = b"TMPL"
_VERSION = 1


@dataclass(frozen=True)
class QueryPacket:
    n: int
    k: int
    payload: bytes

    @property
    def nbytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class TemplatePacket:
    n: int
    m: int
    k: int
    payload: bytes

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def encode_query(q_sym: np.ndarray) -> QueryPacket:
    q = np.asarray(q_sym, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.ndim != 2:
        raise ValueError(f"q_sym must be (N, k), got {q.shape}")
    n, k = int(q.shape[0]), int(q.shape[1])
    header = _QUERY_MAGIC + struct.pack("<III", _VERSION, n, k)
    body = np.ascontiguousarray(q).tobytes()
    return QueryPacket(n=n, k=k, payload=header + body)


def decode_query(packet: QueryPacket | bytes) -> np.ndarray:
    raw = packet.payload if isinstance(packet, QueryPacket) else packet
    if raw[:4] != _QUERY_MAGIC:
        raise ValueError("invalid query packet magic")
    version, n, k = struct.unpack_from("<III", raw, 4)
    if version != _VERSION:
        raise ValueError(f"unsupported query packet version {version}")
    expected = 16 + n * k * 4
    if len(raw) != expected:
        raise ValueError(f"query packet size {len(raw)} != {expected}")
    return np.frombuffer(raw, dtype=np.float32, offset=16).reshape(n, k).copy()


def encode_templates(result: MorkQueryResult) -> TemplatePacket:
    keys = np.ascontiguousarray(result.keys, dtype=np.float32)
    values = np.ascontiguousarray(result.values, dtype=np.float32)
    scores = np.ascontiguousarray(result.scores, dtype=np.float32)
    if keys.ndim != 3:
        raise ValueError(f"keys must be (N, m, k), got {keys.shape}")
    n, m, k = (int(keys.shape[0]), int(keys.shape[1]), int(keys.shape[2]))
    if values.shape != keys.shape or scores.shape != (n, m):
        raise ValueError("keys/values/scores shape mismatch")
    id_blob = ("\n".join("\t".join(row) for row in result.template_ids)).encode("utf-8")
    header = _TEMPL_MAGIC + struct.pack("<IIII", _VERSION, n, m, k)
    body = keys.tobytes() + values.tobytes() + scores.tobytes() + struct.pack("<I", len(id_blob)) + id_blob
    return TemplatePacket(n=n, m=m, k=k, payload=header + body)


def decode_templates(packet: TemplatePacket | bytes) -> MorkQueryResult:
    raw = packet.payload if isinstance(packet, TemplatePacket) else packet
    if raw[:4] != _TEMPL_MAGIC:
        raise ValueError("invalid template packet magic")
    version, n, m, k = struct.unpack_from("<IIII", raw, 4)
    if version != _VERSION:
        raise ValueError(f"unsupported template packet version {version}")
    off = 20
    kv = n * m * k * 4
    keys = np.frombuffer(raw, dtype=np.float32, offset=off, count=n * m * k).reshape(n, m, k).copy()
    off += kv
    values = np.frombuffer(raw, dtype=np.float32, offset=off, count=n * m * k).reshape(n, m, k).copy()
    off += kv
    scores = np.frombuffer(raw, dtype=np.float32, offset=off, count=n * m).reshape(n, m).copy()
    off += n * m * 4
    (id_len,) = struct.unpack_from("<I", raw, off)
    off += 4
    id_text = raw[off : off + id_len].decode("utf-8")
    template_ids = [row.split("\t") for row in id_text.split("\n")] if id_text else []
    if len(template_ids) != n:
        raise ValueError("template id rows != N")
    return MorkQueryResult(keys=keys, values=values, template_ids=template_ids, scores=scores)


def retrieve_on_wire(client: MorkClient, q_sym: np.ndarray, top_m: int) -> tuple[MorkQueryResult, QueryPacket, TemplatePacket]:
    """Pack q, top-m on CPU store, pack (p, v). This is the Tier 2 hop."""
    query_pkt = encode_query(q_sym)
    q = decode_query(query_pkt)
    result = client.query_top_k(q, top_m=top_m)
    templ_pkt = encode_templates(result)
    return decode_templates(templ_pkt), query_pkt, templ_pkt
