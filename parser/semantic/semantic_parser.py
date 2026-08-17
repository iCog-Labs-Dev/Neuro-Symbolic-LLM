"""parser/semantic/semantic_parser.py

Natural-language to Atomese semantic parsers.

ReferenceSemanticParser uses a large model for accurate offline annotation and
dataset generation. DistilledSemanticParser uses the local 1B-3B model in the
deployed system. Both share one prompt and Atomese validation pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import yaml

from parser.grammar.atomese import (
    PREDICATES,
    LinkAtom,
    canonical,
    parse_atom,
    validate_metta_string,
)
from parser.semantic.backends import ModelBackend


class SemanticParseError(ValueError):
    """Raised when model output cannot be converted into valid Atomese."""


class ModelGenerationError(RuntimeError):
    """Raised when a model backend cannot generate usable output."""


# ── Configuration ──────────────────────────────────────────────────────────────

ALLOWED_PREDICATES = ", ".join(sorted(PREDICATES))
_PARSER_PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "parser_prompt.yaml"
)

_CODE_FENCE_RE = re.compile(
    r"^```(?:metta|scheme|lisp)?\s*(.*?)\s*```$",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class SemanticParserConfig:
    """Configuration shared by both semantic parser roles."""

    model_name: str
    prompt_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Reject incomplete configuration before any model request."""
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version cannot be empty")


# ── Shared semantic parser ─────────────────────────────────────────────────────


class _BaseSemanticParser:
    """Shared prompting, generation, and Atomese validation pipeline."""

    parser_role = "semantic"

    def __init__(
        self,
        *,
        backend: ModelBackend,
        config: SemanticParserConfig,
    ) -> None:
        """Create a parser with a replaceable model backend."""
        self._backend = backend
        self._config = config

    @property
    def provider_name(self) -> str:
        """Return the active backend provider identifier."""
        return self._backend.provider_name

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._config.model_name

    @property
    def prompt_version(self) -> str:
        """Return the prompt version used by this parser."""
        return self._config.prompt_version

    @staticmethod
    def build_prompt(sentence: str) -> str:
        """Build the shared text-to-MeTTa conversion prompt."""

        with _PARSER_PROMPT_CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        system_prompt = config["parser"]["system_prompt"].format(
            allowed_predicates=ALLOWED_PREDICATES,
        )

        prompt = config["parser"]["prompt_template"].format(
            system_prompt=system_prompt,
            context=config["parser"]["context"],
            sentence=sentence,
        )
        return dedent(prompt).strip()

    @staticmethod
    def clean_model_output(output: str) -> str:
        """Remove whitespace and one optional Markdown code fence."""
        cleaned = output.strip()
        match = _CODE_FENCE_RE.fullmatch(cleaned)
        if match:
            cleaned = match.group(1).strip()
        return cleaned

    def generate_metta(self, sentence: str) -> str:
        """Generate one cleaned candidate MeTTa expression."""
        normalized_sentence = sentence.strip()
        if not normalized_sentence:
            raise ValueError("The sentence cannot be empty")

        try:
            output = self._backend.generate(
                prompt=self.build_prompt(normalized_sentence),
                model=self.model_name,
            )
        except Exception as error:
            raise ModelGenerationError(
                f"{self.parser_role} parser backend {self.provider_name!r} "
                f"with model {self.model_name!r} failed"
            ) from error

        cleaned = self.clean_model_output(output)
        if not cleaned:
            raise ModelGenerationError("The model returned an empty response")
        return cleaned

    def parse(self, sentence: str) -> list[LinkAtom]:
        """Translate a sentence into one or more validated Atomese links.

        Model output may contain several adjacent top-level expressions.  For
        example, ``(StateOf cat sleeping) (On cat chair)`` is represented as
        two LinkAtom instances, while nested links such as an Evaluation/List
        expression remain a single atom.
        """
        generated_metta = self.generate_metta(sentence)

        if generated_metta.upper() == "UNSUPPORTED":
            raise SemanticParseError(f"Unsupported sentence: {sentence.strip()!r}")

        expressions = self._split_top_level_atoms(generated_metta)
        atoms: list[LinkAtom] = []

        for expression in expressions:
            normalized_metta = canonical(expression)
            is_valid, validation_error = validate_metta_string(normalized_metta)

            if not is_valid:
                raise SemanticParseError(
                    f"The model returned invalid MeTTa: {validation_error}"
                )

            try:
                atom = parse_atom(normalized_metta)
            except ValueError as error:
                raise SemanticParseError(
                    f"The model returned malformed MeTTa: {expression!r}"
                ) from error

            if not isinstance(atom, LinkAtom):
                raise SemanticParseError(
                    "The model response is not a top-level LinkAtom"
                )
            atoms.append(atom)

        return atoms

    @staticmethod
    def _split_top_level_atoms(output: str) -> list[str]:
        """Split whitespace-separated top-level links without splitting nesting."""
        atoms: list[str] = []
        depth = 0
        start: int | None = None

        index = 0
        while index < len(output):
            character = output[index]

            if character.isspace() and depth == 0:
                index += 1
                continue
            if character == "(":
                if depth == 0:
                    start = index
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    raise SemanticParseError(
                        "The model returned unbalanced parentheses"
                    )
                if depth == 0 and start is not None:
                    atoms.append(output[start : index + 1])
                    start = None
            elif depth == 0:
                raise SemanticParseError(
                    "The model response must contain top-level LinkAtom expressions"
                )
            index += 1

        if depth != 0:
            raise SemanticParseError("The model returned unbalanced parentheses")
        if not atoms:
            raise SemanticParseError("The model returned no MeTTa expressions")
        return atoms


# ── Public parser roles ────────────────────────────────────────────────────────


class ReferenceSemanticParser(_BaseSemanticParser):
    """High-accuracy parser used for offline annotation and dataset generation."""

    parser_role = "reference"


class DistilledSemanticParser(_BaseSemanticParser):
    """Local 1B-3B parser used by the final deployed system."""

    parser_role = "distilled"


def main() -> None:
    """Run a small demonstration using the configured model profile."""
    from parser.semantic.model_config import build_reference_semantic_parser

    parser = build_reference_semantic_parser()
    for atom in parser.parse("Ethiopia is inside Africa."):
        print(atom)
