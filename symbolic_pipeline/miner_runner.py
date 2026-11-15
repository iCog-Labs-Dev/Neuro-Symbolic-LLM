"""Run the installed hybrid-miner pipeline from Neuro-Symbolic-LLM."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from parser.ontology.catalog import DEFAULT_BUNDLE_MANIFEST
from symbolic_pipeline.document_export import export_semantic_documents
from symbolic_pipeline.miner_bundle import load_catalog_bundle


@dataclass(frozen=True, slots=True)
class MinerRunPaths:
    """Artifacts produced by one mining run."""

    documents: Path
    rejected_documents: Path
    skipped_rules: Path
    bundle: Path
    emitted_atoms: Path
    metadata: Path
    provenance: Path
    admitted_atoms: Path
    quarantine: Path
    admission: Path
    support: Path
    exact_matches: Path
    rejected_exact_matches: Path
    pattern_records: Path
    rejected_pattern_records: Path

    @classmethod
    def in_directory(cls, output_dir: str | Path):
        root = Path(output_dir)
        return cls(
            documents=root / "documents.jsonl",
            rejected_documents=root / "documents.rejected.jsonl",
            skipped_rules=root / "documents.skipped.jsonl",
            bundle=root / "ontology.bundle.json",
            emitted_atoms=root / "facts.metta",
            metadata=root / "facts.metadata.metta",
            provenance=root / "facts.provenance.jsonl",
            admitted_atoms=root / "admitted.metta",
            quarantine=root / "quarantined.jsonl",
            admission=root / "admission.json",
            support=root / "support.jsonl",
            exact_matches=root / "exact-matches.jsonl",
            rejected_exact_matches=root / "exact-matches.rejected.jsonl",
            pattern_records=root / "pattern-records.jsonl",
            rejected_pattern_records=root / "pattern-records.rejected.jsonl",
        )

    def outputs(self):
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class MinerRunSummary:
    """Summary of the complete mining workflow."""

    accepted_documents: int
    rejected_documents: int
    skipped_rules: int
    admitted_documents: int
    quarantined_documents: int
    admitted_facts: int
    support_patterns: int
    exact_patterns: int
    rejected_exact_patterns: int
    candidate_records: int
    rejected_candidate_records: int
    paths: MinerRunPaths


def _validate_runtime_paths(petta_root, miner_root):
    petta = Path(petta_root).expanduser().resolve()
    miner = Path(miner_root).expanduser().resolve()

    if not (petta / "src" / "main.pl").is_file():
        raise FileNotFoundError(f"PeTTa runtime not found at {petta}")

    if not (miner / "experiments" / "pattern-miner" / "pattern-miner.metta").is_file():
        raise FileNotFoundError(f"Hyperon Miner checkout not found at {miner}")

    return petta, miner


def run_hybrid_miner(
    *,
    parser_results_path: str | Path,
    output_dir: str | Path,
    petta_root: str | Path,
    miner_root: str | Path,
    ontology_manifest: str | Path = DEFAULT_BUNDLE_MANIFEST,
    min_support: int = 2,
    swipl: str = "swipl",
    timeout_seconds: int = 120,
    skip_unsupported_rules: bool = False,
) -> MinerRunSummary:
    """Run parser export, PeTTa admission, mining and candidate construction."""

    if (
        isinstance(min_support, bool)
        or not isinstance(min_support, int)
        or min_support < 1
    ):
        raise ValueError("min_support must be a positive integer")

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
        raise ValueError("timeout_seconds must be positive")

    source = Path(parser_results_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Parser results not found: {source}")

    petta, miner = _validate_runtime_paths(petta_root, miner_root)

    paths = MinerRunPaths.in_directory(output_dir)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    if source in {p.resolve() for p in paths.outputs()}:
        raise ValueError("Parser input must differ from output artifacts")

    existing = [p for p in paths.outputs() if p.exists()]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing artifacts: " + ", ".join(map(str, existing))
        )

    # Import the installed hybrid-miner package.
    try:
        from hybrid_miner.emission.emit_petta import emit_petta
        from hybrid_miner.mining.admitted_support import (
            mine_admitted_support,
        )
        from hybrid_miner.mining.build_pattern_records import (
            build_pattern_records,
        )
        from hybrid_miner.mining.exact_matches import (
            recover_exact_matches,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install hybrid-miner using "
            '`python -m pip install -e ".[miner]"` '
            "with a compatible miner revision."
        ) from error

    created: list[Path] = []

    try:
        # 1. Load shared ontology catalog and miner bundle.
        _, bundle = load_catalog_bundle(Path(ontology_manifest))

        # 2. Convert parser results into SemanticDocuments.
        export = export_semantic_documents(
            source,
            paths.documents,
            paths.rejected_documents,
            bundle=bundle,
            bundle_path=paths.bundle,
            skipped_path=paths.skipped_rules if skip_unsupported_rules else None,
        )

        # The rejected and skipped files explain a failed run, so cleanup keeps them.
        created.extend([paths.documents, paths.bundle])

        if export.accepted == 0:
            raise ValueError(
                "No parser documents were accepted. "
                f"Inspect {paths.rejected_documents}"
            )

        # 3. Emit PeTTa facts and provenance.
        emit_petta(
            bundle=bundle,
            input_path=paths.documents,
            atoms_path=paths.emitted_atoms,
            metadata_path=paths.metadata,
            provenance_path=paths.provenance,
        )

        created.extend(
            [
                paths.emitted_atoms,
                paths.metadata,
                paths.provenance,
            ]
        )

        # 4. Run admission and pattern mining.
        admitted = mine_admitted_support(
            bundle=bundle,
            documents_path=paths.documents,
            emitted_atoms_path=paths.emitted_atoms,
            metadata_path=paths.metadata,
            provenance_path=paths.provenance,
            petta_root=petta,
            miner_root=miner,
            admitted_atoms_path=paths.admitted_atoms,
            quarantine_path=paths.quarantine,
            support_path=paths.support,
            admission_path=paths.admission,
            min_support=min_support,
            swipl=swipl,
            timeout_seconds=timeout_seconds,
        )

        created.extend(
            [
                paths.admitted_atoms,
                paths.quarantine,
                paths.admission,
                paths.support,
            ]
        )

        # 5. Recover exact matches and source evidence.
        exact_count, rejected_exact_count = recover_exact_matches(
            bundle=bundle,
            documents_path=paths.documents,
            atoms_path=admitted.admission.atoms_path,
            provenance_path=paths.provenance,
            support_path=paths.support,
            petta_root=petta,
            miner_root=miner,
            output_path=paths.exact_matches,
            rejected_path=paths.rejected_exact_matches,
            admission_path=admitted.admission.admission_path,
            swipl=swipl,
            timeout_seconds=timeout_seconds,
            min_support=min_support,
        )

        created.extend(
            [
                paths.exact_matches,
                paths.rejected_exact_matches,
            ]
        )

        # 6. Build typed candidate PatternRecords.
        candidate_count, rejected_candidate_count = build_pattern_records(
            bundle=bundle,
            documents_path=paths.documents,
            atoms_path=admitted.admission.atoms_path,
            provenance_path=paths.provenance,
            exact_path=paths.exact_matches,
            output_path=paths.pattern_records,
            rejected_path=paths.rejected_pattern_records,
            admission_path=admitted.admission.admission_path,
        )

        created.extend(
            [
                paths.pattern_records,
                paths.rejected_pattern_records,
            ]
        )

        return MinerRunSummary(
            accepted_documents=export.accepted,
            rejected_documents=export.rejected,
            skipped_rules=export.skipped_rules,
            admitted_documents=admitted.admission.accepted_documents,
            quarantined_documents=admitted.admission.rejected_documents,
            admitted_facts=admitted.admission.facts,
            support_patterns=len(admitted.mining.records),
            exact_patterns=exact_count,
            rejected_exact_patterns=rejected_exact_count,
            candidate_records=candidate_count,
            rejected_candidate_records=rejected_candidate_count,
            paths=paths,
        )

    except BaseException:
        # Remove partial outputs if the workflow fails.
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m symbolic_pipeline.miner_runner",
        description="Run the complete hybrid-miner workflow.",
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Parser SemanticParseResult JSONL",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--petta-root",
        type=Path,
        default=os.getenv("PETTA_ROOT"),
    )
    parser.add_argument(
        "--miner-root",
        type=Path,
        default=os.getenv("MINER_ROOT"),
    )
    parser.add_argument(
        "--ontology-manifest",
        type=Path,
        default=DEFAULT_BUNDLE_MANIFEST,
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--swipl",
        default="swipl",
    )
    parser.add_argument(
        "--skip-unsupported-rules",
        action="store_true",
        help="leave out universal rules the miner cannot hold instead of "
        "rejecting their documents; they are listed in documents.skipped.jsonl",
    )

    options = parser.parse_args(argv)

    if options.petta_root is None or options.miner_root is None:
        parser.error(
            "Provide --petta-root and --miner-root " "or set PETTA_ROOT and MINER_ROOT."
        )

    try:
        result = run_hybrid_miner(
            parser_results_path=options.input,
            output_dir=options.output_dir,
            petta_root=options.petta_root,
            miner_root=options.miner_root,
            ontology_manifest=options.ontology_manifest,
            min_support=options.min_support,
            timeout_seconds=options.timeout_seconds,
            swipl=options.swipl,
            skip_unsupported_rules=options.skip_unsupported_rules,
        )

    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Parser documents: {result.accepted_documents} accepted, "
        f"{result.rejected_documents} rejected"
    )
    if result.skipped_rules:
        print(f"Skipped rules: {result.skipped_rules} -> {result.paths.skipped_rules}")
    print(
        f"PeTTa admission: {result.admitted_documents} accepted, "
        f"{result.quarantined_documents} quarantined"
    )
    print(f"Admitted facts: {result.admitted_facts}")
    print(
        f"Raw support patterns: {result.support_patterns} " f"-> {result.paths.support}"
    )
    print(
        f"Evidence-backed patterns: {result.exact_patterns}; "
        f"rejected: {result.rejected_exact_patterns}"
    )
    print(
        f"Candidate records: {result.candidate_records}; "
        f"rejected: {result.rejected_candidate_records}"
    )
    print(f"Pattern records: {result.paths.pattern_records}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
