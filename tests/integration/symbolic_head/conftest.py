"""Fixtures for A4 tests requiring a live MORK HTTP service."""

import pytest


@pytest.fixture
def mork_client():
    """Provide a DockerMorkClient connected to the MORK server.

    Tests fail when the configured external MORK server is unreachable.
    """
    from residual.symbolic_head.mork_client import DockerMorkClient, get_mork_client

    client = get_mork_client(key_dim=16)
    assert isinstance(client, DockerMorkClient)
    if not client.is_connected():
        raise ConnectionError(
            "MORK server is not reachable at "
            f"{client.server_url}. "
            "MORK integration tests require a live Docker container. "
            "Start a compatible MORK HTTP server at the configured endpoint."
        )
    return client


@pytest.fixture
def mork_client_256():
    """Provide a DockerMorkClient with key_dim=256 for full-dimension tests."""
    from residual.symbolic_head.mork_client import DockerMorkClient, get_mork_client

    client = get_mork_client(key_dim=256)
    assert isinstance(client, DockerMorkClient)
    if not client.is_connected():
        raise ConnectionError(
            "MORK server is not reachable at "
            f"{client.server_url}. "
            "These tests require a compatible MORK HTTP server."
        )
    return client
