"""Tier 2 retrieval bridge."""

import numpy as np

from tier2_mork.client import MorkClient, MorkQueryResult, get_mork_client
from tier2_mork.comm import QueryPacket, TemplatePacket, retrieve_on_wire


class Tier2Retrieve:
    def __init__(self, mork_client: MorkClient | None = None, key_dim: int = 256) -> None:
        self.mork_client = (
            mork_client if mork_client is not None else get_mork_client(key_dim=key_dim)
        )

    def retrieve(
        self, q_sym: np.ndarray, top_m: int = 8
    ) -> tuple[MorkQueryResult, QueryPacket, TemplatePacket]:
        return retrieve_on_wire(self.mork_client, q_sym, top_m=top_m)


# Kept so older imports do not break; retrieve-only.
SymbolicHeadBridge = Tier2Retrieve
