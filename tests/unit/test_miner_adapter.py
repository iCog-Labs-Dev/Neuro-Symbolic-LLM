from dataclasses import dataclass

import numpy as np
import pytest

from Residual.symbolic_head.miner_adapter import adapt_miner_payload


@dataclass
class MinerPayload:
    template_id: str
    sexpr: str
    key: tuple[float, ...]
    value: tuple[float, ...]


def test_adapt_miner_payload_matches_tier2_record_contract():
    payload = MinerPayload(
        template_id="T1",
        sexpr="(and (Inheritance $V0 mammal))",
        key=(1.0, 2.0),
        value=(3.0, 4.0),
    )

    published = adapt_miner_payload(payload, key_dim=2)

    assert published.record.template_id == "T1"
    assert published.record.metta_expr == payload.sexpr
    np.testing.assert_array_equal(published.record.key_vector, [1.0, 2.0])
    np.testing.assert_array_equal(published.record.val_vector, [3.0, 4.0])


@pytest.mark.parametrize(
    "payload",
    [
        MinerPayload("", "(P x)", (1.0, 2.0), (3.0, 4.0)),
        MinerPayload("T1", "", (1.0, 2.0), (3.0, 4.0)),
        MinerPayload("T1", "(P x)", (1.0,), (3.0, 4.0)),
        MinerPayload("T1", "(P x)", (float("nan"), 2.0), (3.0, 4.0)),
    ],
)
def test_adapt_miner_payload_rejects_invalid_data(payload):
    with pytest.raises(ValueError):
        adapt_miner_payload(payload, key_dim=2)