"""The parser and miner must agree on one versioned ontology bundle."""

from pathlib import Path

import pytest

pytest.importorskip("hybrid_miner.ontology.bundle")

from hybrid_miner.ontology.bundle import OntologyBundle  # noqa: E402

from symbolic_pipeline.miner_bundle import load_catalog_bundle  # noqa: E402


def test_default_bundle_round_trips_with_miner_contract(tmp_path: Path) -> None:
    catalog, bundle = load_catalog_bundle()
    path = tmp_path / "bundle.json"
    path.write_text(bundle.canonical_json() + "\n", encoding="utf-8")

    restored = OntologyBundle.load_json(path)
    assert restored == bundle
    assert set(restored.signatures()) == set(catalog.relations)
    assert restored.ontology_versions == catalog.versions
    assert set(restored.types) == catalog.types | {"String"}


def test_manifest_rejects_changed_module_version(tmp_path: Path) -> None:
    manifest = tmp_path / "wrong-version.yaml"
    manifest.write_text(
        'bundle_id: semantic-extraction\nbundle_version: "1"\nmodules:\n  upper: upper-v2\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version mismatch"):
        load_catalog_bundle(manifest)
