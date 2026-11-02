"""Native expression bounds, storage codec integrity and variable sharing."""

import numpy as np
import pytest

from residual.symbolic_head.contracts.template_record import TemplateRecord
from symbolic_memory.storage import (
    MorkTemplateStore,
    _alpha_shape,
    _decode,
    _encode,
    _native_expression,
    _rows,
    _token,
)


def _record():
    return TemplateRecord(
        "pattern-" + "a" * 64 + "-réseau-版",
        '(Agent $event $person)\n(Note "quoted (text)")',
        np.array([3, -0.0, 4], dtype=np.float32),
        np.array([0.1, -0.0, -2], dtype=np.float32),
    )


def test_codec_preserves_full_unicode_text_and_float32_bits():
    record = _record()
    decoded = _decode(_encode(record, 3), 3)
    assert decoded.template_id == record.template_id
    assert decoded.metta_expr == record.metta_expr
    assert decoded.key_vector.tobytes() == record.key_vector.tobytes()
    assert decoded.val_vector.tobytes() == record.val_vector.tobytes()


def test_handles_fit_noninterned_symbol_limit():
    handle = _token("r", _record().template_id.encode("utf-8"))
    assert len(handle) == 53
    assert handle == _token("r", _record().template_id.encode("utf-8"))
    assert handle != _token("r", b"different")


def test_missing_parts_are_not_complete_records():
    payload = _encode(_record(), 3)
    with pytest.raises(ValueError, match="Incomplete"):
        MorkTemplateStore._payload("record", (2, _token("h", payload)), {0: "ab"})


def test_changed_payload_fails_checksum():
    payload = _encode(_record(), 3)
    with pytest.raises(ValueError, match="checksum"):
        MorkTemplateStore._payload("record", (1, _token("h", payload)), {0: "ab"})


def test_wrong_vector_byte_length_is_rejected():
    with pytest.raises(ValueError, match="byte lengths"):
        _decode(_encode(_record(), 3)[:-1], 3)


def test_wrong_dimension_is_rejected():
    with pytest.raises(ValueError):
        _decode(_encode(_record(), 3), 2)


@pytest.mark.parametrize("text", ["garbage", "(BankPartV1 b r 0 ab", "((nested))"])
def test_malformed_flat_response_is_rejected(text):
    with pytest.raises(ValueError):
        _rows(text, "BankPartV1", "b")


def test_rows_accept_reordered_ground_expressions():
    handle = _token("r", b"record")
    text = f"(BankPartV1 b {handle} 1 cd)\n(BankPartV1 b {handle} 0 ab)\n"
    assert [row[3] for row in _rows(text, "BankPartV1", "b")] == ["1", "0"]


def test_native_shape_preserves_shared_variables():
    from hybrid_miner.mining.canonicalize_patterns import parse_expression

    expected = _alpha_shape(_native_expression("(, (Agent $e $p) (Event $e))"))
    actual = _alpha_shape(parse_expression("(, (Agent $ $) (Event _1))"), wire=True)
    different = _alpha_shape(parse_expression("(, (Agent $ $) (Event _2))"), wire=True)
    assert actual == expected
    assert different != expected


def test_unsupported_native_symbol_is_not_silently_encoded_or_truncated():
    with pytest.raises(ValueError, match="63 bytes"):
        _native_expression("(" + "a" * 64 + " $e)")


def test_native_export_size_is_checked_before_upload():
    from hybrid_miner.mining.canonicalize_patterns import render_expression

    clause = tuple(["Predicate"] + ["constant"] * 60)
    expression = tuple([","] + [clause] * 10)
    with pytest.raises(ValueError, match="safe export size"):
        _native_expression(render_expression(expression))
