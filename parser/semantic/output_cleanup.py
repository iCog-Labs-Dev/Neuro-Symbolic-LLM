"""Shared cleanup of JSON text returned by semantic-parser backends."""

from __future__ import annotations

import json
import re

_CODE_FENCE_RE = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_model_output(output: str) -> str:
    """Unwrap one JSON response without changing decoded semantic values.

    Tolerate explanatory text around one complete JSON object. Leave multiple
    objects invalid rather than selecting among competing extractions.
    Contract and ontology validation remain the caller's responsibility.
    """

    cleaned = output.strip()
    match = _CODE_FENCE_RE.fullmatch(cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1 or any(token in cleaned[:start] for token in "{}[]"):
        return cleaned
    try:
        candidate, end = json.JSONDecoder().raw_decode(cleaned, start)
    except json.JSONDecodeError:
        return cleaned
    if not isinstance(candidate, dict) or any(
        token in cleaned[end:] for token in "{}[]"
    ):
        return cleaned
    return json.dumps(candidate, ensure_ascii=False)
