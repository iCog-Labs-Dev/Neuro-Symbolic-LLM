"""Run the installed hybrid-miner pipeline from Neuro-Symbolic-LLM.

The runner orchestrates the existing hybrid-miner interfaces.

Workflow:
    1. Export parser results as SemanticDocuments.
    2. Emit PeTTa facts, metadata, and provenance.
    3. Validate admission and mine admitted facts.
    4. Recover exact matches and source evidence.
    5. Build typed candidate PatternRecords.
    6. Validate artifacts and write a run summary.

Each stage reports progress and elapsed time. If a stage fails,
only that stage's incomplete outputs are removed. Previously
completed stages are preserved.

Use a fresh output directory for each new run. This runner does
not implement automatic checkpointing or stage resumption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parser.ontology.catalog import DEFAULT_BUNDLE_MANIFEST
from symbolic_pipeline.document_export import export_semantic_documents
from symbolic_pipeline.miner_bundle import load_catalog_bundle

# ---------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinerRunPaths:
    """Artifacts produced by one mining run."""

    documents: Path
    rejected_documents: Path
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
    def in_directory(cls, output_dir: str | Path) -> MinerRunPaths:
        root = Path(output_dir)

        return cls(
            documents=root / "documents.jsonl",
            rejected_documents=root / "documents.rejected.jsonl",
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

    def outputs(self) -> tuple[Path, ...]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


# ---------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinerRunSummary:
    """Summary of the complete mining workflow."""

    accepted_documents: int
    rejected_documents: int
    admitted_documents: int
    quarantined_documents: int
    admitted_facts: int
    support_patterns: int
    exact_patterns: int
    rejected_exact_patterns: int
    candidate_records: int
    rejected_candidate_records: int
    paths: MinerRunPaths


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def log(message: str = "") -> None:
    """Print a message immediately."""

    print(message, flush=True)


def format_duration(seconds: float) -> str:
    """Format elapsed execution time."""

    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, remaining = divmod(int(seconds), 60)

    if minutes < 60:
        return f"{minutes}m {remaining}s"

    hours, minutes = divmod(minutes, 60)

    return f"{hours}h {minutes}m {remaining}s"


def count_jsonl(path: Path) -> int:
    """Count nonempty JSONL records."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def directory_size(path: Path) -> int:
    """Return total size of a file or directory."""

    if path.is_file():
        return path.stat().st_size

    if not path.is_dir():
        return 0

    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_bytes(size: int) -> str:
    """Format bytes as a readable size."""

    value = float(size)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"

            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} TB"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file without exposing an incomplete summary."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

        temporary.replace(path)

    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------
# Runtime validation
# ---------------------------------------------------------------------


def _validate_runtime_paths(
    petta_root: str | Path,
    miner_root: str | Path,
) -> tuple[Path, Path]:
    """Validate PeTTa and Hyperon Miner checkouts."""

    petta = Path(petta_root).expanduser().resolve()
    miner = Path(miner_root).expanduser().resolve()

    if not (petta / "src" / "main.pl").is_file():
        raise FileNotFoundError(f"PeTTa runtime not found at {petta}")

    if not (miner / "experiments" / "pattern-miner" / "pattern-miner.metta").is_file():
        raise FileNotFoundError(f"Hyperon Miner checkout not found at {miner}")

    return petta, miner


# ---------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------


def validate_stage_outputs(
    stage_name: str,
    outputs: Sequence[Path],
) -> None:
    """Ensure every expected stage artifact exists."""

    missing = [path for path in outputs if not path.is_file() or path.is_symlink()]

    if missing:
        raise RuntimeError(
            f"Stage '{stage_name}' did not produce all expected artifacts:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def cleanup_stage_outputs(outputs: Sequence[Path]) -> None:
    """Remove only the explicitly assigned outputs of a failed stage."""

    for path in outputs:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)


def run_stage(
    *,
    number: int,
    total: int,
    name: str,
    description: str,
    outputs: Sequence[Path],
    action: Callable[[], Any],
    metrics: Callable[[Any], dict[str, Any]] | None = None,
) -> Any:
    """Execute one stage with progress, timing and safe cleanup."""

    log()
    log("=" * 68)
    log(f"[{number}/{total}] {description}")
    log("=" * 68)

    existing = [path for path in outputs if path.exists()]

    if existing:
        raise FileExistsError(
            f"Outputs already exist for stage '{name}'. "
            "Use a fresh output directory or inspect the previous run:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )

    started = time.monotonic()

    try:
        result = action()

        validate_stage_outputs(name, outputs)

        elapsed = time.monotonic() - started

        log("Status: COMPLETED")
        log(f"Elapsed time: {format_duration(elapsed)}")

        if metrics is not None:
            for key, value in metrics(result).items():
                log(f"  {key}: {value}")

        return result

    except BaseException as error:
        elapsed = time.monotonic() - started

        cleanup_stage_outputs(outputs)

        log("Status: FAILED")
        log(f"Elapsed time: {format_duration(elapsed)}")
        log(f"Error: {type(error).__name__}: {error}")
        log("Incomplete outputs from this stage were cleaned up.")
        log("Outputs from previously completed stages were preserved.")

        raise


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


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
        raise ValueError("timeout_seconds must be a positive integer")

    source = Path(parser_results_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Parser results not found: {source}")

    ontology_path = Path(ontology_manifest).expanduser().resolve()

    if not ontology_path.is_file():
        raise FileNotFoundError(f"Ontology manifest not found: {ontology_path}")

    petta, miner = _validate_runtime_paths(
        petta_root,
        miner_root,
    )

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    paths = MinerRunPaths.in_directory(root)
    summary_path = root / "miner_summary.json"

    # Ensure the input cannot be overwritten by a pipeline output.
    all_outputs = (
        *paths.outputs(),
        summary_path,
    )

    if source in {path.resolve() for path in all_outputs}:
        raise ValueError("Parser input must differ from all miner output artifacts.")

    # Require a clean output directory for a new run.
    existing = [path for path in all_outputs if path.exists()]

    if existing:
        raise FileExistsError(
            "The miner output directory already contains artifacts. "
            "Use a fresh output directory to preserve the previous run:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )

    # Import the installed hybrid-miner interfaces.
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

    log()
    log("=" * 68)
    log("HYBRID MINER PIPELINE")
    log("=" * 68)

    log(f"Parser input: {source}")
    log(f"Output directory: {root}")
    log(f"Ontology manifest: {ontology_path}")
    log(f"Minimum support: {min_support}")
    log(f"Execution timeout: {timeout_seconds}s")
    log(f"PeTTa runtime: {petta}")
    log(f"Hyperon Miner: {miner}")

    # Load the shared ontology catalog and miner bundle.
    log("\nLoading ontology bundle...")

    _, bundle = load_catalog_bundle(ontology_path)

    log(f"Ontology bundle loaded: {bundle.bundle_id}")
    log(f"Ontology version: {bundle.bundle_version}")

    total_stages = 6

    # =============================================================
    # STAGE 1: SEMANTIC DOCUMENT EXPORT
    # =============================================================

    stage1_outputs = (
        paths.documents,
        paths.rejected_documents,
        paths.bundle,
    )

    export = run_stage(
        number=1,
        total=total_stages,
        name="document_export",
        description="Semantic document export",
        outputs=stage1_outputs,
        action=lambda: export_semantic_documents(
            source,
            paths.documents,
            paths.rejected_documents,
            bundle=bundle,
            bundle_path=paths.bundle,
        ),
        metrics=lambda result: {
            "Accepted documents": result.accepted,
            "Rejected documents": result.rejected,
        },
    )

    if export.accepted == 0:
        raise ValueError(
            "No parser documents were accepted. " f"Inspect {paths.rejected_documents}"
        )

    # =============================================================
    # STAGE 2: PETTA FACT EMISSION
    # =============================================================

    stage2_outputs = (
        paths.emitted_atoms,
        paths.metadata,
        paths.provenance,
    )

    run_stage(
        number=2,
        total=total_stages,
        name="fact_emission",
        description="PeTTa fact emission",
        outputs=stage2_outputs,
        action=lambda: emit_petta(
            bundle=bundle,
            input_path=paths.documents,
            atoms_path=paths.emitted_atoms,
            metadata_path=paths.metadata,
            provenance_path=paths.provenance,
        ),
        metrics=lambda _: {
            "Facts file": paths.emitted_atoms,
            "Metadata file": paths.metadata,
            "Provenance file": paths.provenance,
        },
    )

    # =============================================================
    # STAGE 3: ADMISSION AND PATTERN MINING
    # =============================================================

    stage3_outputs = (
        paths.admitted_atoms,
        paths.quarantine,
        paths.admission,
        paths.support,
    )

    admitted = run_stage(
        number=3,
        total=total_stages,
        name="admission_and_mining",
        description="PeTTa admission and pattern mining",
        outputs=stage3_outputs,
        action=lambda: mine_admitted_support(
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
        ),
        metrics=lambda result: {
            "Admitted documents": result.admission.accepted_documents,
            "Quarantined documents": result.admission.rejected_documents,
            "Admitted facts": result.admission.facts,
            "Support patterns": len(result.mining.records),
        },
    )

    # =============================================================
    # STAGE 4: EXACT MATCH RECOVERY
    # =============================================================

    stage4_outputs = (
        paths.exact_matches,
        paths.rejected_exact_matches,
    )

    exact_count, rejected_exact_count = run_stage(
        number=4,
        total=total_stages,
        name="exact_match_recovery",
        description="Exact match recovery and source evidence",
        outputs=stage4_outputs,
        action=lambda: recover_exact_matches(
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
        ),
        metrics=lambda result: {
            "Evidence-backed patterns": result[0],
            "Rejected patterns": result[1],
        },
    )

    # =============================================================
    # STAGE 5: CANDIDATE CONSTRUCTION
    # =============================================================

    stage5_outputs = (
        paths.pattern_records,
        paths.rejected_pattern_records,
    )

    candidate_count, rejected_candidate_count = run_stage(
        number=5,
        total=total_stages,
        name="candidate_construction",
        description="Candidate PatternRecord construction",
        outputs=stage5_outputs,
        action=lambda: build_pattern_records(
            bundle=bundle,
            documents_path=paths.documents,
            atoms_path=admitted.admission.atoms_path,
            provenance_path=paths.provenance,
            exact_path=paths.exact_matches,
            output_path=paths.pattern_records,
            rejected_path=paths.rejected_pattern_records,
            admission_path=admitted.admission.admission_path,
        ),
        metrics=lambda result: {
            "Candidate records": result[0],
            "Rejected candidates": result[1],
        },
    )

    # =============================================================
    # STAGE 6: FINAL VALIDATION
    # =============================================================

    def final_validation() -> dict[str, Any]:
        artifacts = paths.outputs()

        missing = [
            path for path in artifacts if not path.is_file() or path.is_symlink()
        ]

        if missing:
            raise RuntimeError(
                "Missing or invalid final artifacts:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )

        sizes = {path.name: path.stat().st_size for path in artifacts}

        return {
            "artifact_count": len(artifacts),
            "total_bytes": sum(sizes.values()),
            "artifact_sizes": sizes,
        }

    final_metrics = run_stage(
        number=6,
        total=total_stages,
        name="final_validation",
        description="Final artifact validation",
        outputs=(),
        action=final_validation,
    )

    # =============================================================
    # FINAL SUMMARY
    # =============================================================

    summary = MinerRunSummary(
        accepted_documents=export.accepted,
        rejected_documents=export.rejected,
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

    summary_data = {
        "pipeline": "neuro_symbolic_hybrid_miner",
        "status": "completed",
        "input": str(source),
        "output_directory": str(root),
        "configuration": {
            "ontology_manifest": str(ontology_path),
            "min_support": min_support,
            "timeout_seconds": timeout_seconds,
            "swipl": swipl,
            "petta_root": str(petta),
            "miner_root": str(miner),
        },
        "counts": {
            "accepted_documents": summary.accepted_documents,
            "rejected_documents": summary.rejected_documents,
            "admitted_documents": summary.admitted_documents,
            "quarantined_documents": summary.quarantined_documents,
            "admitted_facts": summary.admitted_facts,
            "support_patterns": summary.support_patterns,
            "exact_patterns": summary.exact_patterns,
            "rejected_exact_patterns": summary.rejected_exact_patterns,
            "candidate_records": summary.candidate_records,
            "rejected_candidate_records": summary.rejected_candidate_records,
        },
        "validation": final_metrics,
        "artifacts": {
            name: str(getattr(paths, name)) for name in paths.__dataclass_fields__
        },
        "completed_at": time.time(),
    }

    write_json_atomic(summary_path, summary_data)

    log()
    log("=" * 68)
    log("MINER PIPELINE COMPLETED")
    log("=" * 68)

    for name, value in summary_data["counts"].items():
        log(f"{name}: {value}")

    log(f"\nSummary: {summary_path}")
    log(f"Total artifact size: {format_bytes(final_metrics['total_bytes'])}")

    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


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
        help="Fresh directory for this mining run",
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
        )

    except (OSError, ValueError, RuntimeError) as error:
        print(
            f"error: {error}",
            file=sys.stderr,
        )
        return 1

    print("\nFinal results:")
    print(
        f"Parser documents: {result.accepted_documents} accepted, "
        f"{result.rejected_documents} rejected"
    )
    print(
        f"PeTTa admission: {result.admitted_documents} accepted, "
        f"{result.quarantined_documents} quarantined"
    )
    print(f"Admitted facts: {result.admitted_facts}")
    print(f"Support patterns: {result.support_patterns}")
    print(
        f"Exact patterns: {result.exact_patterns}; "
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
