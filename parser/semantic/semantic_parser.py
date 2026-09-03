"""parser/semantic/semantic_parser.py

Structured natural-language semantic parsers.

ReferenceSemanticParser uses a large model (teacher) for offline annotation.
DistilledSemanticParser uses a saved fine-tuned model (student) for inference.

Both share the same validation, normalization, and rendering pipeline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, cast

import torch
import yaml
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from parser.grammar.atomese import LinkAtom
from parser.semantic.backends import ModelBackend
from parser.semantic.metta_renderer import (
    MettaRenderError,
)
from parser.semantic.metta_renderer import (
    render_metta as render_semantic_result,
)
from parser.semantic.metta_renderer import (
    validate_rendered_metta as parse_rendered_metta,
)
from parser.semantic.normalization import (
    SemanticNormalizationError,
    normalize_semantic_result,
)
from parser.semantic.schema import SemanticParseResult


class SemanticParseError(ValueError):
    """Raised when model output cannot be converted into valid semantics."""


class ModelGenerationError(RuntimeError):
    """Raised when a model backend cannot generate usable output."""


# ── Configuration ──────────────────────────────────────────────────────────────

_PARSER_PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "parser_prompt.yaml"
)

_PREDICATE_SCHEMA_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "predicate_schema.yaml"
)


def _load_predicate_schemas() -> dict[str, dict[str, object]]:
    """Load predicate schemas and reject invalid configuration early."""
    with _PREDICATE_SCHEMA_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Predicate schema must be a YAML mapping")
    predicates = config.get("predicates")
    if not isinstance(predicates, dict) or not predicates:
        raise ValueError("Predicate schema must define a nonempty 'predicates' mapping")

    for name, schema in predicates.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Predicate names must be nonempty strings")
        if not isinstance(schema, dict):
            raise ValueError(f"Schema for {name!r} must be a mapping")
        description = schema.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Schema for {name!r} requires a description")

        if schema.get("variable_arity") is True:
            minimum = schema.get("min_arity")
            maximum = schema.get("max_arity")
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
                raise ValueError(f"Invalid min_arity for {name!r}")
            if maximum is not None and (
                not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or maximum < minimum
            ):
                raise ValueError(f"Invalid max_arity for {name!r}")
            continue

        arity = schema.get("arity")
        roles = schema.get("roles")
        if not isinstance(arity, int) or isinstance(arity, bool) or arity < 1:
            raise ValueError(f"Invalid arity for {name!r}")
        if (
            not isinstance(roles, list)
            or len(roles) != arity
            or any(not isinstance(role, str) or not role.strip() for role in roles)
        ):
            raise ValueError(f"Schema roles for {name!r} must match its arity")

    evaluation = predicates.get("Evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("fallback") is not True:
        raise ValueError("Evaluation must be defined with fallback=true")
    return predicates


PREDICATE_SCHEMAS = _load_predicate_schemas()

# Only semantic predicates from YAML are exposed to the LLM.
ALLOWED_PREDICATES = ", ".join(sorted(PREDICATE_SCHEMAS))


def _build_predicate_guide() -> str:
    """Describe predicate roles directly from the validated YAML schema."""
    lines = []
    for name, schema in PREDICATE_SCHEMAS.items():
        description = schema["description"]
        if schema.get("variable_arity") is True:
            roles = "semantic roles appropriate to the relation"
        else:
            roles = ", ".join(cast(list[str], schema["roles"]))
        lines.append(f"- {name}({roles}): {description}")
    return "\n".join(lines)


PREDICATE_GUIDE = _build_predicate_guide()

# Prompt format used by both teacher and student
PROMPT = "Extract triples from this sentence as JSON:\nSentence: {text}\nJSON:"

_CODE_FENCE_RE = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class SemanticParserConfig:
    """Configuration shared by both semantic parser roles."""

    model_name: str
    prompt_version: str = "2.0.0"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version cannot be empty")


# ── Base Parser ────────────────────────────────────────────────────────────────


class _BaseSemanticParser:
    """Shared structured semantic extraction pipeline.

    Supports both:
    - Reference (teacher): uses backend (external API) to generate
    - Distilled (student): uses saved model from disk
    """

    parser_role = "semantic"

    def __init__(
        self,
        *,
        backend: ModelBackend | None = None,
        config: SemanticParserConfig,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """Create a parser with either a backend or a loaded model."""
        self._backend = backend
        self._config = config
        self._model = model
        self._tokenizer = tokenizer

    @property
    def provider_name(self) -> str:
        if self._backend is not None:
            return self._backend.provider_name
        return "distilled-local"

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def prompt_version(self) -> str:
        return self._config.prompt_version

    @staticmethod
    def build_prompt(sentence: str, context: str = "") -> str:
        """Build the structured semantic extraction prompt."""
        with _PARSER_PROMPT_CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        system_prompt = config["parser"]["system_prompt"].replace(
            "{allowed_predicates}", ALLOWED_PREDICATES
        )
        system_prompt = system_prompt.replace("{predicate_guide}", PREDICATE_GUIDE)
        prompt = config["parser"]["prompt_template"].format(
            system_prompt=system_prompt,
            context=context,
            sentence=sentence,
        )
        return dedent(prompt).strip()

    @staticmethod
    def clean_model_output(output: str) -> str:
        """Remove whitespace and one optional JSON Markdown code fence."""
        cleaned = output.strip()
        match = _CODE_FENCE_RE.fullmatch(cleaned)
        if match:
            cleaned = match.group(1).strip()
        return cleaned

    @staticmethod
    def validate_predicates(result: SemanticParseResult) -> SemanticParseResult:
        """Validate predicate and fallback consistency."""
        for assertion in result.assertions:
            predicate = assertion.predicate
            if predicate not in PREDICATE_SCHEMAS:
                raise SemanticParseError(f"Unknown semantic predicate: {predicate!r}")
            if predicate == "Evaluation":
                if not assertion.fallback:
                    raise SemanticParseError("Evaluation must have fallback=true")
                if not assertion.relation:
                    raise SemanticParseError("Evaluation requires a relation")
            else:
                if assertion.fallback:
                    raise SemanticParseError(
                        f"{predicate} is a known predicate, so fallback must be false"
                    )
                if assertion.relation is not None:
                    raise SemanticParseError(f"{predicate} should have relation=null")
        return result

    @staticmethod
    def validate_arguments(result: SemanticParseResult) -> SemanticParseResult:
        """Validate assertion argument counts against the YAML schema."""
        for assertion in result.assertions:
            schema = PREDICATE_SCHEMAS.get(assertion.predicate)
            if not isinstance(schema, dict):
                raise SemanticParseError(
                    f"No schema defined for predicate {assertion.predicate!r}"
                )
            actual_arity = len(assertion.arguments)
            if schema.get("variable_arity") is True:
                minimum, maximum = schema.get("min_arity"), schema.get("max_arity")
                if not isinstance(minimum, int) or minimum < 1:
                    raise ValueError(f"Invalid min_arity for {assertion.predicate!r}")
                if maximum is not None and (
                    not isinstance(maximum, int) or maximum < minimum
                ):
                    raise ValueError(f"Invalid max_arity for {assertion.predicate!r}")
                if actual_arity < minimum:
                    raise SemanticParseError(
                        f"{assertion.predicate} requires at least {minimum} arguments, "
                        f"got {actual_arity}"
                    )
                if isinstance(maximum, int) and actual_arity > maximum:
                    raise SemanticParseError(
                        f"{assertion.predicate} allows at most {maximum} arguments, "
                        f"got {actual_arity}"
                    )
                continue

            expected_arity = schema.get("arity")
            if not isinstance(expected_arity, int):
                raise ValueError(f"Invalid arity schema for {assertion.predicate!r}")
            if actual_arity != expected_arity:
                raise SemanticParseError(
                    f"{assertion.predicate} requires {expected_arity} arguments, "
                    f"got {actual_arity}"
                )
            expected_roles = schema.get("roles")
            actual_roles = [argument.role for argument in assertion.arguments]
            if actual_roles != expected_roles:
                raise SemanticParseError(
                    f"{assertion.predicate} requires argument roles "
                    f"{expected_roles!r}, got {actual_roles!r}"
                )
        return result

    @staticmethod
    def render_metta(result: SemanticParseResult) -> list[str]:
        """Convert structured semantics into deterministic MeTTa expressions."""
        try:
            return render_semantic_result(result)
        except MettaRenderError as error:
            raise SemanticParseError(str(error)) from error

    @staticmethod
    def validate_rendered_metta(expressions: list[str]) -> list[LinkAtom]:
        """Validate rendered MeTTa and convert it into LinkAtom objects."""
        try:
            return parse_rendered_metta(expressions)
        except MettaRenderError as error:
            raise SemanticParseError(str(error)) from error

    @staticmethod
    def split_top_level_atoms(output: str) -> list[str]:
        """Split whitespace-separated top-level links without splitting nesting."""
        atoms: list[str] = []
        depth = 0
        start: int | None = None

        for idx, ch in enumerate(output):
            if ch.isspace() and depth == 0:
                continue
            if ch == "(":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    raise SemanticParseError("Unbalanced parentheses")
                if depth == 0 and start is not None:
                    atoms.append(output[start : idx + 1])
                    start = None
            elif depth == 0:
                raise SemanticParseError("Top-level LinkAtom expected")

        if depth != 0:
            raise SemanticParseError("Unbalanced parentheses")
        if not atoms:
            raise SemanticParseError("No MeTTa expressions")
        return atoms

    def _generate_text(self, prompt: str) -> str:
        """Generate text using either backend or saved model."""
        if self._model is not None and self._tokenizer is not None:
            # Distilled: use saved model
            enc = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out_ids = self._model.generate(
                    **enc,
                    max_new_tokens=256,
                    num_beams=4,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            new_ids = out_ids[0][prompt_len:]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        elif self._backend is not None:
            return self._backend.generate(prompt=prompt, model=self.model_name)

        else:
            raise ValueError("No backend or model available")

    def _parse_json_to_metta(self, json_output: str) -> list[LinkAtom]:
        """Parse JSON, normalize, validate, render to MeTTa."""
        # 1. Parse JSON
        try:
            data = json.loads(json_output)
            result = SemanticParseResult.model_validate(data)
        except json.JSONDecodeError as error:
            raise SemanticParseError(f"Invalid JSON output: {error}") from error
        except ValidationError as error:
            raise SemanticParseError(f"Invalid structured output: {error}") from error

        # 2. Normalize
        try:
            result = normalize_semantic_result(result)
        except SemanticNormalizationError as error:
            raise SemanticParseError(f"Normalization failed: {error}") from error

        # 3. Validate predicates and arguments
        result = self.validate_predicates(result)
        result = self.validate_arguments(result)

        # 4. Render to MeTTa
        try:
            expressions = self.render_metta(result)
        except MettaRenderError as error:
            raise SemanticParseError(f"Render failed: {error}") from error

        # 5. Validate and return
        return self.validate_rendered_metta(expressions)

    def generate_structured(
        self,
        sentence: str,
        context: str = "",
        *,
        aliases: Mapping[str, str] | None = None,
        alias_types: Mapping[str, str] | None = None,
    ) -> SemanticParseResult:
        """Generate, normalize, and validate one structured model response."""
        normalized_sentence = sentence.strip()
        if not normalized_sentence:
            raise ValueError("The sentence cannot be empty")

        prompt = self.build_prompt(
            sentence=normalized_sentence,
            context=context.strip(),
        )

        try:
            output = self._generate_text(prompt)
        except Exception as error:
            raise ModelGenerationError(
                f"{self.parser_role} parser failed with model {self.model_name!r}"
            ) from error

        cleaned = self.clean_model_output(output)
        if not cleaned:
            raise ModelGenerationError("The model returned an empty response")

        try:
            result = SemanticParseResult.model_validate_json(cleaned)
        except ValidationError as error:
            raise SemanticParseError(
                "The model returned invalid structured semantic output"
            ) from error

        try:
            result = normalize_semantic_result(
                result,
                aliases=aliases,
                alias_types=alias_types,
            )
        except SemanticNormalizationError as error:
            raise SemanticParseError(str(error)) from error

        result = self.validate_predicates(result)
        result = self.validate_arguments(result)
        return result

    def parse(
        self,
        sentence: str,
        context: str = "",
        *,
        aliases: Mapping[str, str] | None = None,
        alias_types: Mapping[str, str] | None = None,
    ) -> list[LinkAtom]:
        """Convert text into validated Atomese LinkAtom objects."""
        result = self.generate_structured(
            sentence=sentence,
            context=context,
            aliases=aliases,
            alias_types=alias_types,
        )
        expressions = self.render_metta(result)
        return self.validate_rendered_metta(expressions)


# ── Reference Parser (Teacher) ───────────────────────────────────────────────


class ReferenceSemanticParser(_BaseSemanticParser):
    """High-accuracy parser used for offline annotation and dataset generation."""

    parser_role = "reference"

    def __init__(self, *, backend: ModelBackend, config: SemanticParserConfig):
        super().__init__(backend=backend, config=config)


# ── Distilled Parser (Student) ──────────────────────────────────────────────


class DistilledSemanticParser(_BaseSemanticParser):
    """Local parser that loads a saved fine-tuned student model."""

    parser_role = "distilled"

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ):
        self.model_dir = Path(model_dir)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        # Load tokenizer from disk
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model from disk
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        # Load metadata
        metadata_path = self.model_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"base_model": "unknown"}

        # Create config
        config = SemanticParserConfig(
            model_name=self.metadata.get("base_model", "unknown"),
            prompt_version=self.metadata.get("prompt_version", "2.0.0"),
        )

        # Initialize base class with model and tokenizer
        super().__init__(
            backend=None,
            config=config,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        print(f"DistilledSemanticParser loaded from {model_dir}")
        print(f"  Base model: {self.metadata.get('base_model', 'unknown')}")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> DistilledSemanticParser:
        """Load a saved model for inference."""
        return cls(model_dir, device, max_new_tokens, num_beams)

    def parse(
        self,
        sentence: str,
        context: str = "",
        *,
        aliases: Mapping[str, str] | None = None,
        alias_types: Mapping[str, str] | None = None,
    ) -> list[LinkAtom]:
        """Parse a sentence using the loaded student model."""
        prompt = self.build_prompt(sentence.strip(), context)
        json_output = self._generate_text(prompt)
        return self._parse_json_to_metta(json_output)


# ── Builder functions ─────────────────────────────────────────────────────────


def build_distilled_parser(
    model_dir: str = "models/distil_parser",
    device: str = "auto",
    max_new_tokens: int = 256,
    num_beams: int = 4,
) -> DistilledSemanticParser:
    """Build a distilled parser from a saved model directory."""
    return DistilledSemanticParser.from_pretrained(
        model_dir=model_dir,
        device=device,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
