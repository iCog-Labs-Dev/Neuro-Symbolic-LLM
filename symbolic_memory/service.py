"""Snapshot-based CPU retrieval server and persistent TCP client."""

from __future__ import annotations

import logging
import socket
import socketserver
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from residual.symbolic_head.contracts.template_record import MorkQueryResult
from symbolic_memory.protocol import (
    HEADER,
    MAX_MESSAGE_BYTES,
    VERSION,
    encode_info,
    encode_query,
    encode_result,
    query_shape,
    read_info,
    read_result,
    receive,
)

if TYPE_CHECKING:
    from residual.symbolic_head.retrieval import FaissTemplateIndex
    from symbolic_memory.bank import TemplateBank
    from symbolic_memory.storage import MorkTemplateStore

logger = logging.getLogger(__name__)


class RetrievalServer(socketserver.ThreadingTCPServer):
    """Serve an immutable FAISS snapshot loaded from a verified MORK bank."""

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(
        self,
        address: tuple[str, int],
        bank: TemplateBank,
        store: MorkTemplateStore,
        *,
        max_queries: int = 1024,
        max_top_m: int = 64,
        max_connections: int = 8,
        timeout_sec: float = 30.0,
    ) -> None:
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (max_queries, max_top_m, max_connections)
            )
            or not np.isfinite(timeout_sec)
            or timeout_sec <= 0
        ):
            raise ValueError("Service limits and timeout must be positive")
        if any(
            any(char in row.template_id for char in "\t\r\n") for row in bank.records
        ):
            raise ValueError("TCP template IDs cannot contain tabs or newlines")
        self.index: FaissTemplateIndex = store.load_index(bank)
        self.timeout_sec = timeout_sec
        self._slots = threading.BoundedSemaphore(max_connections)
        self.info: dict[str, Any] = {
            "protocol_version": VERSION,
            "input_snapshot_sha256": bank.input_snapshot_sha256,
            "record_count": len(bank.records),
            "vector_dimension": bank.vector_dim,
            "index_backend": self.index.index_backend,
            "max_queries": max_queries,
            "max_top_m": max_top_m,
            "max_message_bytes": MAX_MESSAGE_BYTES,
        }
        super().__init__(address, _RetrievalHandler)

    def process_request(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
    ) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class _RetrievalHandler(socketserver.BaseRequestHandler):
    server: RetrievalServer

    def handle(self) -> None:
        connection = self.request
        connection.settimeout(self.server.timeout_sec)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            connection.sendall(encode_info(self.server.info))
            while True:
                count, width, top_m = query_shape(
                    receive(connection, HEADER.size),
                    self.server.index.key_dim,
                    self.server.info["max_queries"],
                    self.server.info["max_top_m"],
                )
                matches = min(top_m, self.server.info["record_count"])
                if (
                    HEADER.size + count * matches * (2 * width + 1) * 4
                    > MAX_MESSAGE_BYTES
                ):
                    raise ValueError("Requested response exceeds the transport limit")
                queries = np.frombuffer(
                    receive(connection, count * width * 4), dtype="<f4"
                )
                result = self.server.index.query_top_k(
                    queries.reshape(count, width), top_m
                )
                connection.sendall(encode_result(result))
        except (EOFError, ConnectionError, TimeoutError):
            return
        except (ValueError, OSError) as error:
            logger.warning("Retrieval connection rejected: %s", error)


class RetrievalClient:
    """Persistent, serialized TCP connection bound to an expected bank snapshot."""

    def __init__(
        self,
        host: str,
        port: int,
        expected_snapshot: str,
        timeout_sec: float = 30.0,
    ) -> None:
        self._lock = threading.Lock()
        self._address = (host, port)
        self._expected_snapshot = expected_snapshot
        self._timeout_sec = timeout_sec
        self._connection, self.info = self._connect()

    def _connect(self) -> tuple[socket.socket, dict[str, Any]]:
        connection = socket.create_connection(self._address, self._timeout_sec)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            info = read_info(connection)
            if info.get("input_snapshot_sha256") != self._expected_snapshot:
                raise ValueError("Retrieval service loaded a different bank snapshot")
            for field in (
                "record_count",
                "vector_dimension",
                "max_queries",
                "max_top_m",
            ):
                value = info.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Invalid service descriptor field: {field}")
        except BaseException:
            connection.close()
            raise
        return connection, info

    def reconnect(self) -> None:
        """Open a new connection without retrying a query.

        Bank identity, dimensions and request limits must match the original
        descriptor to preserve compiled retrieval shapes.
        """
        with self._lock:
            self.close()
            connection, info = self._connect()
            if info != self.info:
                connection.close()
                raise ValueError("Reconnected retrieval service descriptor changed")
            self._connection = connection

    def query_top_k(self, queries: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        packet = encode_query(queries, top_m)
        count, width, top_m = query_shape(
            packet[: HEADER.size],
            self.info["vector_dimension"],
            self.info["max_queries"],
            self.info["max_top_m"],
        )
        with self._lock:
            try:
                self._connection.sendall(packet)
                return read_result(
                    self._connection,
                    count,
                    min(top_m, self.info["record_count"]),
                    width,
                )
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> RetrievalClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
