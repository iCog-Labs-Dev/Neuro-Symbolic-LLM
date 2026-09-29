"""Shared cleanup of JSON text returned by semantic-parser backends."""

from __future__ import annotations

import re

_CODE_FENCE_RE = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_model_output(output: str) -> str:
    """Remove whitespace and one optional JSON Markdown code fence."""

    cleaned = output.strip()
    match = _CODE_FENCE_RE.fullmatch(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned
