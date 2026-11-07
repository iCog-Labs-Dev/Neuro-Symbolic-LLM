"""Compile a versioned ontology catalog into the miner's bundle contract."""

from __future__ import annotations

from pathlib import Path

from hybrid_miner.ontology.bundle import OntologyBundle

from parser.ontology.catalog import (
    DEFAULT_BUNDLE_MANIFEST,
    OntologyCatalog,
    load_manifest_catalog,
)


def load_catalog_bundle(
    manifest_path: Path = DEFAULT_BUNDLE_MANIFEST,
    *,
    ontology_directory: Path | None = None,
) -> tuple[OntologyCatalog, OntologyBundle]:
    """Load a manifest's catalog and the miner bundle built from the same modules."""

    catalog, manifest = load_manifest_catalog(
        manifest_path, ontology_directory=ontology_directory
    )
    bundle = OntologyBundle.model_validate(
        {
            "contract_version": "ontology-bundle-v1",
            "bundle_id": manifest.bundle_id,
            "bundle_version": manifest.bundle_version,
            "ontology_versions": catalog.versions,
            "types": sorted(catalog.types | {"String"}),
            "predicates": [
                {
                    "name": relation.name,
                    "ontology_version": relation.ontology_version,
                    "argument_types": [
                        list(argument.allowed_types) for argument in relation.arguments
                    ],
                }
                for relation in sorted(
                    catalog.relations.values(), key=lambda item: item.name
                )
            ],
        }
    )
    return catalog, bundle
