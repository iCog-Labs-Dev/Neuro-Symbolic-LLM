"""Pytest configuration — MORK Docker is mandatory for all tests."""

import pytest

from tier2_mork.client import DockerMorkClient, get_mork_client


@pytest.fixture
def mork_client():
    """Provide a DockerMorkClient connected to the MORK server.

    Tests FAIL (not skip) when MORK is unreachable. Start the server with:
        docker compose up -d --build
    """
    client = get_mork_client(key_dim=16)
    assert isinstance(client, DockerMorkClient)
    if not client.is_connected():
        raise ConnectionError(
            "MORK server is not reachable at "
            f"{client.server_url}. "
            "All tests require a live MORK Docker container. "
            "Start it with: docker compose up -d --build"
        )
    return client


@pytest.fixture
def mork_client_256():
    """Provide a DockerMorkClient with key_dim=256 for full-dimension tests."""
    client = get_mork_client(key_dim=256)
    assert isinstance(client, DockerMorkClient)
    if not client.is_connected():
        raise ConnectionError(
            "MORK server is not reachable at "
            f"{client.server_url}. "
            "All tests require a live MORK Docker container. "
            "Start it with: docker compose up -d --build"
        )
    return client
