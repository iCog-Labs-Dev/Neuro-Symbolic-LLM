"""Stage A symbolic head and Tier 2 retrieval integration."""

from residual.symbolic_head.head import SymbolicHead
from residual.symbolic_head.losses import (
    combined_symbolic_loss,
    key_space_alignment_loss,
    value_space_regression_loss,
)
from residual.symbolic_head.mork_client import (
    DockerMorkClient,
    MorkClient,
    MorkQueryResult,
    TemplateRecord,
    get_mork_client,
)
from residual.symbolic_head.miner_adapter import (
    PublishedTemplate,
    adapt_miner_payload,
    publish_miner_payload,
)
from residual.symbolic_head.retrieval_bridge import Tier2Retrieve

__all__ = [
    "SymbolicHead",
    "combined_symbolic_loss",
    "key_space_alignment_loss",
    "value_space_regression_loss",
    "DockerMorkClient",
    "MorkClient",
    "MorkQueryResult",
    "TemplateRecord",
    "Tier2Retrieve",
    "get_mork_client",
    "PublishedTemplate",
    "adapt_miner_payload",
    "publish_miner_payload",
]
