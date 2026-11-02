"""Native MORK template structures with linked, bounded vector/metadata storage."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx
import numpy as np
from hybrid_miner.mining.canonicalize_patterns import (
    Expression,
    parse_expression,
    render_expression,
)

from residual.symbolic_head.contracts.template_record import (
    TemplateRecord,
    validate_template_record,
)
from symbolic_memory.bank import TemplateBank

if TYPE_CHECKING:
    from residual.symbolic_head.retrieval import FaissTemplateIndex

_PART_HEX_LENGTH = 60
_RECORD_HEAD = "BankRecordV1"
_PART_HEAD = "BankPartV1"
_NATIVE_HEAD = "BankTemplateV1"
_HANDLE = re.compile(r"r[a-z2-7]{52}\Z")
_DIGEST = re.compile(r"h[a-z2-7]{52}\Z")
_HEX = re.compile(r"(?:[0-9a-f]{2}){1,30}\Z")


def _token(prefix: str, payload: bytes) -> str:
    return (
        prefix
        + base64.b32encode(hashlib.sha256(payload).digest())
        .decode()
        .rstrip("=")
        .lower()
    )


def _encode(
    record: TemplateRecord, dimension: int, pattern_metadata: dict | None = None
) -> bytes:
    identifier, expression, key, value = validate_template_record(
        record.template_id,
        record.metta_expr,
        record.key_vector,
        record.val_vector,
        dimension,
    )
    metadata = json.dumps(
        [identifier, expression, dimension, pattern_metadata],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    key_bytes: bytes = key.astype("<f4").tobytes()
    value_bytes: bytes = value.astype("<f4").tobytes()
    return metadata + b"\n" + key_bytes + value_bytes


def _decode(payload: bytes, dimension: int) -> TemplateRecord:
    metadata, separator, vectors = payload.partition(b"\n")
    if not separator or len(vectors) != dimension * 8:
        raise ValueError("Stored record has invalid vector byte lengths")
    fields = json.loads(metadata)
    if (
        not isinstance(fields, list)
        or len(fields) != 4
        or not isinstance(fields[0], str)
        or not isinstance(fields[1], str)
        or not isinstance(fields[2], int)
        or isinstance(fields[2], bool)
        or fields[2] != dimension
        or (fields[3] is not None and not isinstance(fields[3], dict))
    ):
        raise ValueError("Stored record has invalid metadata")
    key = np.frombuffer(vectors[: dimension * 4], dtype="<f4").copy()
    value = np.frombuffer(vectors[dimension * 4 :], dtype="<f4").copy()
    identifier, expression, key, value = validate_template_record(
        fields[0],
        fields[1],
        key,
        value,
        dimension,
    )
    key.setflags(write=False)
    value.setflags(write=False)
    return TemplateRecord(identifier, expression, key, value)


def _native_expression(source: str) -> Expression:
    """Parse a native expression and enforce the server's wire-size limits."""
    expression = parse_expression(source)
    if not isinstance(expression, tuple):
        raise ValueError("A native template must be a structured expression")

    def size(node: Expression) -> int:
        if isinstance(node, tuple):
            if len(node) > 63:
                raise ValueError(
                    "Native expression arity exceeds this MORK build's limit"
                )
            return 1 + sum(size(child) for child in node)
        if node.startswith("$"):
            return 1
        if re.fullmatch(r"_[0-9]+", node):
            raise ValueError(
                "Native input must use named variables, not MORK wire references"
            )
        symbol = json.loads(node) if node.startswith('"') else node
        length = len(symbol.encode("utf-8"))
        if length > 63:
            raise ValueError(
                "Native symbol exceeds 63 bytes; a compatible MORK build is required"
            )
        return 1 + length

    # Include the native wrapper's constants and leave room below the 4 KiB buffer.
    if size(expression) + 128 > 3500:
        raise ValueError("Native template exceeds this MORK build's safe export size")
    return expression


def _alpha_shape(expression: Expression, *, wire: bool = False) -> Expression:
    """Preserve variable sharing while accounting for MORK's $ / _N notation."""
    names: dict[str, str] = {}
    count = 0

    def visit(node: Expression) -> Expression:
        nonlocal count
        if isinstance(node, tuple):
            return tuple(visit(child) for child in node)
        if wire and re.fullmatch(r"_[0-9]+", node):
            index = int(node[1:]) - 1
            if not 0 <= index < count:
                raise ValueError("Invalid native variable reference returned by MORK")
            return f"$v{index}"
        if node.startswith("$"):
            if node == "$" or node not in names:
                result = f"$v{count}"
                count += 1
                if node != "$":
                    names[node] = result
                return result
            return names[node]
        return node

    return visit(expression)


def _rows(text: str, head: str, namespace: str) -> list[list[str]]:
    """Parse flat ground rows in the bank storage format."""
    result = []
    position = 0
    for match in re.finditer(r"\(([^()]*)\)", text):
        if text[position : match.start()].strip():
            raise ValueError("Unexpected data in MORK bank response")
        fields = match.group(1).split()
        if len(fields) != 5 or fields[:2] != [head, namespace]:
            raise ValueError("Unexpected MORK bank row")
        if not _HANDLE.fullmatch(fields[2]):
            raise ValueError("Invalid stored record handle")
        result.append(fields)
        position = match.end()
    if text[position:].strip():
        raise ValueError("Incomplete MORK bank response")
    return result


class MorkTemplateStore:
    """Store native MORK expressions with linked metadata and vector payloads.

    Native expressions support structural queries. Linked payloads preserve
    full IDs, source PatternRecord metadata and float32 vector bytes.
    """

    def __init__(self, server_url: str, timeout_sec: float = 30.0) -> None:
        url = httpx.URL(server_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("MORK URL must use http or https")
        if url.query or url.fragment or url.userinfo:
            raise ValueError("MORK URL must not contain credentials, query or fragment")
        if timeout_sec <= 0:
            raise ValueError("timeout must be positive")
        self.server_url = server_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout_sec)

    def __enter__(self) -> MorkTemplateStore:
        return self

    def __exit__(self, *args: object) -> None:
        self._http.close()

    @staticmethod
    def _namespace(bank: TemplateBank) -> str:
        return _token("b", b"native-v1:" + bytes.fromhex(bank.input_snapshot_sha256))

    def _query(self, pattern: str, template: str) -> str:
        response = self._http.get(
            f"{self.server_url}/export/{quote(pattern, safe='')}/{quote(template, safe='')}/"
        )
        response.raise_for_status()
        text: str = response.text
        return text

    def _download(self, head: str, namespace: str) -> list[list[str]]:
        pattern = f"({head} {namespace} $r $a $b)"
        return _rows(self._query(pattern, pattern), head, namespace)

    def _upload(self, rows: list[str]) -> None:
        if rows:
            response = self._http.post(
                f"{self.server_url}/upload/%24x/%24x/",
                content=("\n".join(rows) + "\n").encode("utf-8"),
                headers={"Content-Type": "text/plain"},
            )
            response.raise_for_status()

    def _headers(self, bank: TemplateBank) -> dict[str, tuple[int, str]]:
        namespace = self._namespace(bank)
        headers: dict[str, tuple[int, str]] = {}
        for _, _, handle, count, digest in self._download(_RECORD_HEAD, namespace):
            if (
                handle in headers
                or not count.isdecimal()
                or int(count) < 1
                or not _DIGEST.fullmatch(digest)
            ):
                raise ValueError("Conflicting or invalid stored record header")
            headers[handle] = (int(count), digest)
        return headers

    def _snapshot(
        self, bank: TemplateBank
    ) -> tuple[dict[str, tuple[int, str]], dict[str, dict[int, str]]]:
        namespace = self._namespace(bank)
        headers = self._headers(bank)
        parts: dict[str, dict[int, str]] = defaultdict(dict)
        for _, _, handle, index, data in self._download(_PART_HEAD, namespace):
            if not index.isdecimal() or not _HEX.fullmatch(data):
                raise ValueError("Invalid stored payload part")
            number = int(index)
            if number in parts[handle]:
                raise ValueError("Duplicate or conflicting stored payload part")
            parts[handle][number] = data
        return headers, parts

    @staticmethod
    def _expected(bank: TemplateBank) -> dict[str, bytes]:
        result = {}
        metadata = {
            pattern.id: pattern.model_dump(mode="json") for pattern in bank.patterns
        }
        for record in bank.records:
            _native_expression(record.metta_expr)
            handle = _token("r", record.template_id.encode("utf-8"))
            if handle in result:
                raise ValueError("Duplicate record ID or storage-handle collision")
            result[handle] = _encode(
                record, bank.vector_dim, metadata.get(record.template_id)
            )
        if not result:
            raise ValueError("No templates supplied")
        return result

    @staticmethod
    def _payload(handle: str, header: tuple[int, str], parts: dict[int, str]) -> bytes:
        count, digest = header
        if len(parts) != count or any(
            index != number for number, index in enumerate(sorted(parts))
        ):
            raise ValueError(f"Incomplete stored record: {handle}")
        payload = bytes.fromhex("".join(parts[index] for index in range(count)))
        if _token("h", payload) != digest:
            raise ValueError(f"Stored record checksum mismatch: {handle}")
        return payload

    def _native_records(self, bank: TemplateBank) -> dict[str, Expression]:
        namespace = self._namespace(bank)
        pattern = f"({_NATIVE_HEAD} {namespace} $id $term)"
        tree = parse_expression("(results " + self._query(pattern, pattern) + ")")
        if not isinstance(tree, tuple):
            raise ValueError("Invalid native template response")
        records: dict[str, Expression] = {}
        for row in tree[1:]:
            if (
                not isinstance(row, tuple)
                or len(row) != 4
                or row[:2] != (_NATIVE_HEAD, namespace)
                or not isinstance(row[2], str)
                or not _HANDLE.fullmatch(row[2])
                or row[2] in records
            ):
                raise ValueError("Conflicting or invalid native template row")
            records[row[2]] = _alpha_shape(row[3], wire=True)
        return records

    def match_templates(self, bank: TemplateBank, expression: str) -> list[str]:
        """Match native MORK structures and return their public template IDs."""
        term = _native_expression(expression)
        source = render_expression(term)
        identifier = "$__bank_id"
        while identifier in source:
            identifier += "_"
        pattern = f"({_NATIVE_HEAD} {self._namespace(bank)} {identifier} {source})"
        response = self._query(pattern, f"(match {identifier})")
        tree = parse_expression("(results " + response + ")")
        if not isinstance(tree, tuple):
            raise ValueError("Invalid native match response")
        ids = {
            _token("r", record.template_id.encode("utf-8")): record.template_id
            for record in bank.records
        }
        committed = self._headers(bank)
        expected = self._expected(bank)
        matches = set()
        for row in tree[1:]:
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or row[0] != "match"
                or not isinstance(row[1], str)
                or row[1] not in ids
            ):
                raise ValueError("Unknown native match handle")
            if row[1] not in committed:
                continue
            payload = expected[row[1]]
            count = (len(payload) * 2 + _PART_HEX_LENGTH - 1) // _PART_HEX_LENGTH
            if committed[row[1]] != (count, _token("h", payload)):
                raise ValueError(
                    "Matched template metadata does not belong to this bank"
                )
            matches.add(ids[row[1]])
        return sorted(matches)

    def check_bank(self, bank: TemplateBank) -> dict[str, object]:
        expected = self._expected(bank)
        headers, parts = self._snapshot(bank)
        native = self._native_records(bank)
        missing, conflicting = [], []
        for record in bank.records:
            handle = _token("r", record.template_id.encode("utf-8"))
            if handle not in headers or handle not in native:
                missing.append(record.template_id)
            elif self._payload(
                handle, headers[handle], parts.get(handle, {})
            ) != expected[handle] or native[handle] != _alpha_shape(
                _native_expression(record.metta_expr)
            ):
                conflicting.append(record.template_id)
        extra = sorted(
            (headers.keys() | parts.keys() | native.keys()) - expected.keys()
        )
        return {
            "matches": not missing and not conflicting and not extra,
            "expected_count": len(expected),
            "matched_count": len(expected) - len(missing) - len(conflicting),
            "missing_ids": missing,
            "conflicting_ids": conflicting,
            "extra_handles": extra,
            "native_count": len(native),
            "namespace": self._namespace(bank),
        }

    def install_bank(
        self, bank: TemplateBank, batch_size: int = 64
    ) -> dict[str, object]:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        expected = self._expected(bank)
        headers, stored = self._snapshot(bank)
        native = self._native_records(bank)
        terms = {
            _token("r", r.template_id.encode("utf-8")): _native_expression(r.metta_expr)
            for r in bank.records
        }
        if any(
            handle not in terms or shape != _alpha_shape(terms[handle])
            for handle, shape in native.items()
        ):
            raise ValueError(
                "Native structure conflicts with supplied bank; nothing uploaded"
            )
        if (headers.keys() | stored.keys()) - expected.keys():
            raise ValueError("Unexpected records in this immutable bank namespace")
        namespace = self._namespace(bank)
        encoded = {}
        for handle, payload in expected.items():
            text = payload.hex()
            parts = [
                text[i : i + _PART_HEX_LENGTH]
                for i in range(0, len(text), _PART_HEX_LENGTH)
            ]
            header = (len(parts), _token("h", payload))
            if handle in headers and headers[handle] != header:
                raise ValueError(
                    "Stored header conflicts with the supplied bank; nothing uploaded"
                )
            if any(
                index >= len(parts) or parts[index] != value
                for index, value in stored.get(handle, {}).items()
            ):
                raise ValueError(
                    "Stored payload conflicts with the supplied bank; nothing uploaded"
                )
            encoded[handle] = (parts, header)
        missing = [
            handle
            for handle in expected
            if handle not in headers or handle not in native
        ]
        try:
            handles = list(expected)
            for start in range(0, len(handles), batch_size):
                batch = handles[start : start + batch_size]
                rows = [
                    f"({_PART_HEAD} {namespace} {handle} {index} {value})"
                    for handle in batch
                    for index, value in enumerate(encoded[handle][0])
                    if index not in stored.get(handle, {})
                ]
                self._upload(rows)
                self._upload(
                    [
                        f"({_RECORD_HEAD} {namespace} {handle} {encoded[handle][1][0]} {encoded[handle][1][1]})"
                        for handle in batch
                        if handle not in headers
                    ]
                )
                self._upload(
                    [
                        f"({_NATIVE_HEAD} {namespace} {handle} {render_expression(terms[handle])})"
                        for handle in batch
                        if handle not in native
                    ]
                )
            report = self.check_bank(bank)
            if not report["matches"]:
                raise ValueError(f"Bank readback mismatch: {report}")
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                "Installation did not verify; partial writes may exist. "
                f"No rollback was attempted. Run check-bank before retrying: {error}"
            ) from error
        return {
            "expected_count": len(expected),
            "already_present_count": len(expected) - len(missing),
            "uploaded_count": len(missing),
            "verified_count": len(expected),
            "native_count": len(terms),
            "index_updated": False,
            "namespace": namespace,
        }

    def load_index(self, bank: TemplateBank) -> FaissTemplateIndex:
        """Verify stored payloads and native structures, then build a FAISS index."""
        from residual.symbolic_head.retrieval import (
            FaissTemplateIndex,
            load_retrieval_config,
        )

        expected = self._expected(bank)
        headers, parts = self._snapshot(bank)
        native = self._native_records(bank)
        if set(native) != set(expected):
            raise ValueError("Native template structures are missing")
        if set(headers) != set(expected) or (parts.keys() - expected.keys()):
            raise ValueError("The requested bank is not completely installed")
        records = []
        for handle, original in expected.items():
            payload = self._payload(handle, headers[handle], parts.get(handle, {}))
            if payload != original:
                raise ValueError("Stored record differs from the requested bank")
            record = _decode(payload, bank.vector_dim)
            if native[handle] != _alpha_shape(_native_expression(record.metta_expr)):
                raise ValueError("Native structure differs from the requested template")
            if _token("r", record.template_id.encode("utf-8")) != handle:
                raise ValueError("Stored record ID does not match its handle")
            records.append(record)
        config = load_retrieval_config()
        index = FaissTemplateIndex(
            key_dim=bank.vector_dim,
            backend=config.backend,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construction=config.hnsw_ef_construction,
            hnsw_ef_search=config.hnsw_ef_search,
        )
        index.rebuild(records)
        return index
