"""Response-envelope cleanup for semantic extraction output."""

import json

import pytest

from parser.semantic.output_cleanup import clean_model_output


def test_unwraps_one_current_extraction_object() -> None:
    payload = {
        "entities": [],
        "assertions": [
            {
                "predicate": "Evaluation",
                "relation": "rain",
                "arguments": [],
                "fallback": True,
                "confidence": 0.9,
                "evidence": [{"text": "It rains."}],
            }
        ],
        "rules": [],
    }
    output = "Here is the extraction:\n" + json.dumps(payload) + "\nDone."
    assert json.loads(clean_model_output(output)) == payload


def test_preserves_unicode_quotes_and_nested_source_content() -> None:
    payload = {
        "entities": [],
        "assertions": [
            {
                "predicate": "Evaluation",
                "relation": "say",
                "arguments": [],
                "fallback": True,
                "confidence": 0.9,
                "evidence": [{"text": 'Café {x} says "yes".\n🙂', "occurrence": 0}],
            }
        ],
        "rules": [],
    }
    output = "Result:\n" + json.dumps(payload, ensure_ascii=False)
    assert json.loads(clean_model_output(output)) == payload


@pytest.mark.parametrize(
    "output",
    [
        '{"entities": [], "assertions": []}\n{"entities": [], "assertions": []}',
        'Result: {"entities": [], "assertions": []}\nAlternative: {"assertions": []}',
    ],
)
def test_multiple_objects_remain_invalid(output: str) -> None:
    cleaned = clean_model_output(output)
    assert cleaned == output
    with pytest.raises(json.JSONDecodeError):
        json.loads(cleaned)


@pytest.mark.parametrize(
    "output",
    [
        "not JSON",
        "{broken}",
        '{"assertions": [',
        '{"broken": invalid, "nested": {"entities": [], "assertions": []}}',
        'Result: [{"entities": [], "assertions": []}]',
        '{"entities": [], "assertions": []}}',
    ],
)
def test_malformed_output_is_not_repaired_into_a_document(output: str) -> None:
    assert clean_model_output(output) == output


@pytest.mark.parametrize(
    "output",
    [
        '{"entities": [], "assertions": []}',
        '[{"assertions": []}]',
    ],
)
def test_valid_json_is_not_reserialized(output: str) -> None:
    assert clean_model_output("  " + output + "  ") == output


def test_unwraps_one_optional_code_fence() -> None:
    assert (
        clean_model_output('```json\n{"assertions": []}\n```') == '{"assertions": []}'
    )
