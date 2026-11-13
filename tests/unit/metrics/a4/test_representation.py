"""Pure calculation tests for A4 and future symbolic representation metrics."""

import ast
from pathlib import Path

import numpy as np
import pytest

import metrics.a4.representation as representation
from metrics.a4.representation import (
    bank_template_mutual_information,
    capacity_sensitivity,
    column_template_binding_stability,
    cosine_geometry,
    factor_organization,
    human_template_review_summary,
    key_log_probabilities,
    query_silhouette,
    select_factors,
    symbolic_alignment_preservation,
    symbolic_attention_contribution,
    symbolic_commutator,
    symbolic_gradient_conflict,
    symbolic_key_entanglement,
    symbolic_mixed_fisher_coupling,
    value_alignment_error,
)


def test_module_runtime_import_is_numpy_only() -> None:
    tree = ast.parse(Path(representation.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imports.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imports <= {"__future__", "collections", "numpy", "typing"}


def test_fracture_overlap_and_zero_effects() -> None:
    result = factor_organization([[1, 0], [0, 2], [1, 1], [0, 0]])
    assert result["fracture"] == [0.5, 0.5, 1.0, None]
    assert result["valid_factor_indices"] == [0, 1, 2]
    np.testing.assert_allclose(
        result["overlap_matrix"], [[1, 0, 0.5], [0, 1, 0.5], [0.5, 0.5, 1]]
    )
    assert result["mean_off_diagonal_overlap"] == pytest.approx(1 / 3)
    assert factor_organization([[0, 0]])["overlap_matrix"] == []


@pytest.mark.parametrize("value", [[[-1, 1]], [[float("nan"), 0]], [], [1, 2]])
def test_invalid_factor_effects_are_rejected(value) -> None:
    with pytest.raises(ValueError):
        factor_organization(value)


def test_fixed_bank_cosine_geometry_counts_unordered_pairs() -> None:
    result = cosine_geometry([[1, 0], [2, 0], [0, 3]], block_size=1)
    assert result["pairs"] == 3
    assert result["mean_cosine"] == pytest.approx(1 / 3)
    assert result["mean_squared_cosine"] == pytest.approx(1 / 3)
    assert result["norm_max"] == 3
    assert cosine_geometry([[1, 0]])["mean_cosine"] is None
    with pytest.raises(ValueError, match="zero"):
        cosine_geometry([[0, 0]])


def test_key_probability_is_raw_dot_product_and_numerically_stable() -> None:
    result = key_log_probabilities([[1000, 0]], [[1, 0], [2, 0]])
    np.testing.assert_allclose(result, [[-1000, 0]])
    with pytest.raises(ValueError, match="dimensions"):
        key_log_probabilities([[1]], [[1, 2]])


def test_query_silhouette_expands_multilabel_memberships() -> None:
    queries = [[0, 0], [0, 0], [10, 0], [10, 0]]
    assert query_silhouette(queries, [["a"], ["a"], ["b"], ["b"]])[
        "silhouette"
    ] == pytest.approx(1)
    multilabel = query_silhouette([[0], [0]], [["a", "b"], ["a", "b"]])
    assert multilabel["memberships"] == 4
    assert multilabel["silhouette"] == pytest.approx(0)
    assert query_silhouette([[0], [1]], [["a"], ["b"]])["silhouette"] is None
    with pytest.raises(ValueError, match="Unique"):
        query_silhouette([[0]], [["a", "a"]])


def test_factor_selection_requires_cross_document_reference_support() -> None:
    probes = [
        {"corpus_id": "c", "document_id": "one", "relevant_ids": ["a", "b"]},
        {"corpus_id": "c", "document_id": "one", "relevant_ids": ["a", "b"]},
        {"corpus_id": "c", "document_id": "two", "relevant_ids": ["a", "c"]},
    ]
    names, support = select_factors(probes)
    assert names == ["a"]
    assert support == {"a": [0, 1, 2]}


def test_capacity_controls_use_exact_cosine_and_keep_full_dimension() -> None:
    vectors = np.eye(64)[:3]
    result = capacity_sensitivity(
        vectors[:2], vectors, ["a", "b", "c"], [["a"], ["b"]], np.eye(64)
    )
    full = next(row for row in result["controls"] if row["dimensions"] == 64)
    assert full["scores"][0]["all_probe_precision"] == pytest.approx(1)
    assert full["available_reference_pair_fraction"] == pytest.approx(1)
    assert result["ann_exact_id_agreement_at_8"] is None
    with pytest.raises(ValueError, match="reference"):
        capacity_sensitivity(
            vectors[:2],
            vectors,
            ["a", "b", "c"],
            [["missing"], ["b"]],
            np.eye(64),
        )


def test_bank_template_mutual_information_uses_joint_counts() -> None:
    bound = bank_template_mutual_information([[2, 0], [0, 2]])
    independent = bank_template_mutual_information([[1, 1], [1, 1]])
    assert bound["mutual_information"] == pytest.approx(np.log(2))
    assert bound["bank_entropy"] == pytest.approx(np.log(2))
    assert bound["template_entropy"] == pytest.approx(np.log(2))
    assert independent["mutual_information"] == pytest.approx(0)
    np.testing.assert_allclose(bound["joint_probabilities"], [[0.5, 0], [0, 0.5]])


@pytest.mark.parametrize("counts", [[[0, 0]], [[1, -1]], [[1, float("inf")]]])
def test_bank_template_mutual_information_rejects_invalid_counts(counts) -> None:
    with pytest.raises(ValueError):
        bank_template_mutual_information(counts)


def test_column_template_binding_stability_correlates_aligned_matrices() -> None:
    first = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    same_pattern = 0.8 * first + 0.1
    inverse = 1 - first
    result = column_template_binding_stability([first, same_pattern, inverse])
    assert result["pairs"] == 3
    assert result["defined_pairs"] == 3
    np.testing.assert_allclose(
        result["pairwise_correlations"],
        [[1, 1, -1], [1, 1, -1], [-1, -1, 1]],
        atol=1e-12,
    )
    assert result["mean_pairwise_correlation"] == pytest.approx(-1 / 3)


def test_binding_stability_marks_constant_correlations_undefined() -> None:
    result = column_template_binding_stability([np.zeros((2, 2)), np.ones((2, 2))])
    assert result["defined_pairs"] == 0
    assert result["pairwise_correlations"] == [[None, None], [None, None]]
    assert result["mean_pairwise_correlation"] is None


def test_symbolic_alignment_preservation_reports_top_k_set_agreement() -> None:
    before = [[3, 2, 1, 0], [3, 2, 1, 0]]
    after = [[3, 2, 0, 1], [0, 2, 3, 1]]
    result = symbolic_alignment_preservation(before, after, top_k=2)
    assert result["per_probe_top_k_agreement"] == [1, 0.5]
    assert result["top_k_agreement"] == pytest.approx(0.75)
    assert result["exact_top_k_set_agreement_rate"] == pytest.approx(0.5)
    assert result["top_1_agreement"] == pytest.approx(0.5)


def test_symbolic_alignment_preservation_validates_alignment_and_depth() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        symbolic_alignment_preservation([[1, 0]], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="top_k"):
        symbolic_alignment_preservation([[1, 0]], [[1, 0]], top_k=3)


def test_symbolic_key_entanglement_returns_full_cosine_matrix() -> None:
    result = symbolic_key_entanglement([[1, 0], [0, 1], [1, 1]])
    root_half = 1 / np.sqrt(2)
    np.testing.assert_allclose(
        result["cosine_matrix"],
        [[1, 0, root_half], [0, 1, root_half], [root_half, root_half, 1]],
    )
    assert result["pairs"] == 3
    assert result["mean_off_diagonal_cosine"] == pytest.approx(np.sqrt(2) / 3)
    assert result["mean_squared_off_diagonal_cosine"] == pytest.approx(1 / 3)
    assert result["off_diagonal_absolute_mass"] == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="zero"):
        symbolic_key_entanglement([[0, 0]])


def test_symbolic_commutator_reports_parameter_and_function_spaces() -> None:
    result = symbolic_commutator(
        [2, 0],
        [0, 2],
        [[1, 0], [0, 1]],
        [[0, 0], [0, -1]],
        reference_parameters=[1, 1],
        eps=0,
    )
    assert result["parameter_commutator"] == pytest.approx(2)
    assert result["function_commutator"] == pytest.approx(1.5)
    assert result["function_commutator_rms"] == pytest.approx(np.sqrt(2.5))
    assert result["function_commutator_max"] == pytest.approx(2)


def test_symbolic_commutator_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="parameter vectors"):
        symbolic_commutator([1], [1, 2], [[0]], [[0]])
    with pytest.raises(ValueError, match="output arrays"):
        symbolic_commutator([1], [1], [[0]], [[0], [1]])


def test_value_alignment_error_is_mean_squared_l2_per_example() -> None:
    assert value_alignment_error([[1, 2], [3, 4]], [[0, 0], [1, 2]]) == pytest.approx(
        6.5
    )
    assert value_alignment_error([1, 2], [0, 0]) == pytest.approx(5)


@pytest.mark.parametrize(
    ("predicted", "target"),
    [
        ([[1, 2]], [[1], [2]]),
        ([[float("nan")]], [[0]]),
        ([], []),
        (1, 1),
    ],
)
def test_value_alignment_error_rejects_invalid_arrays(predicted, target) -> None:
    with pytest.raises(ValueError):
        value_alignment_error(predicted, target)


def test_symbolic_gradient_conflict_summarizes_nonzero_pairs() -> None:
    result = symbolic_gradient_conflict(
        {
            "forward": [1, 0],
            "reverse": [-1, 0],
            "orthogonal": [0, 2],
            "zero": [0, 0],
        }
    )
    assert result["status"] == "ok"
    assert result["pairs"] == 6
    assert result["defined_pairs"] == 3
    assert result["zero_gradient_names"] == ["zero"]
    assert result["mean_conflict"] == pytest.approx(1 / 3)
    assert result["max_conflict"] == pytest.approx(1)
    assert result["fraction_conflicting"] == pytest.approx(1 / 3)
    assert result["conflict_matrix"][0][1] == pytest.approx(1)
    assert result["conflict_matrix"][0][3] is None


def test_symbolic_gradient_conflict_marks_undefined_summary() -> None:
    result = symbolic_gradient_conflict({"only": [0, 0]})
    assert result["status"] == "unavailable"
    assert result["mean_conflict"] is None
    with pytest.raises(ValueError, match="equal shapes"):
        symbolic_gradient_conflict({"a": [1], "b": [1, 2]})


def test_symbolic_mixed_fisher_coupling_uses_per_example_scores() -> None:
    result = symbolic_mixed_fisher_coupling(
        {
            "first": [[1], [0]],
            "same_support": [[2], [0]],
            "disjoint_support": [[0], [3]],
            "zero": [[0], [0]],
        }
    )
    assert result["status"] == "ok"
    assert result["defined_pairs"] == 3
    assert result["coupling_matrix"][0][1] == pytest.approx(1)
    assert result["coupling_matrix"][0][2] == pytest.approx(0)
    assert result["coupling_matrix"][0][3] is None
    assert result["mean_coupling"] == pytest.approx(1 / 3)
    assert result["zero_score_gradient_groups"] == ["zero"]


def test_symbolic_mixed_fisher_coupling_validates_example_alignment() -> None:
    with pytest.raises(ValueError, match="align"):
        symbolic_mixed_fisher_coupling({"a": [[1], [2]], "b": [[1]]})
    with pytest.raises(ValueError, match="matrix"):
        symbolic_mixed_fisher_coupling({"a": [1, 2]})


def test_symbolic_attention_contribution_orients_each_metric() -> None:
    result = symbolic_attention_contribution(
        {"accuracy": 0.8, "loss": 0.4},
        {"accuracy": 0.7, "loss": 0.5},
        higher_is_better={"accuracy": True, "loss": False},
    )
    assert result["improved_count"] == 2
    assert result["degraded_count"] == 0
    assert result["metrics"]["accuracy"]["difference"] == pytest.approx(0.1)
    assert result["metrics"]["accuracy"]["contribution"] == pytest.approx(0.1)
    assert result["metrics"]["loss"]["difference"] == pytest.approx(-0.1)
    assert result["metrics"]["loss"]["contribution"] == pytest.approx(0.1)


def test_symbolic_attention_contribution_requires_aligned_finite_metrics() -> None:
    with pytest.raises(ValueError, match="match exactly"):
        symbolic_attention_contribution({"a": 1}, {"b": 1}, True)
    with pytest.raises(ValueError, match="every metric"):
        symbolic_attention_contribution({"a": 1, "b": 2}, {"a": 0, "b": 1}, {"a": True})
    with pytest.raises(ValueError, match="finite scalar"):
        symbolic_attention_contribution({"a": True}, {"a": 0}, True)


def test_human_template_review_summary_reports_agreement() -> None:
    result = human_template_review_summary(
        [[True, True], [False, False]],
        [[True, False], [True, True]],
        reviewer_ids=["r1", "r2"],
    )
    assert result["items"] == 2
    assert result["ratings"] == 4
    assert result["reviewers"] == 2
    assert result["relevance_rate"] == pytest.approx(0.5)
    assert result["readability_rate"] == pytest.approx(0.75)
    assert result["relevant_and_readable_rate"] == pytest.approx(0.25)
    assert result["agreement"]["status"] == "ok"
    assert result["agreement"]["relevance"]["pairwise_agreement"] == pytest.approx(1)
    assert result["agreement"]["relevance"]["fleiss_kappa"] == pytest.approx(1)
    assert result["agreement"]["readability"]["pairwise_agreement"] == pytest.approx(
        0.5
    )
    assert result["agreement"]["readability"]["fleiss_kappa"] == pytest.approx(-1 / 3)


def test_human_review_agreement_requires_aligned_repeated_ratings() -> None:
    result = human_template_review_summary(
        [True, False], [True, True], reviewer_ids=["r1", "r2"]
    )
    assert result["agreement"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="boolean"):
        human_template_review_summary([1, 0], [True, False])
    with pytest.raises(ValueError, match="unique"):
        human_template_review_summary(
            [[True, False]], [[True, False]], reviewer_ids=["same", "same"]
        )
