"""Tests for comparison calibration, archetype constraints, and phase gates."""

from __future__ import annotations

import json

import numpy as np
import pytest

import anchorop as ao
from anchorop.anchored import verify_phase2_gate


def _measurement_for_operator(J: np.ndarray, *, seed: int = 1, m: int = 8) -> ao.MeasuredOperator:
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(J.shape[0], m))
    S = -np.linalg.solve(J, U)
    return ao.measure_from_sensitivity(S, U, reg="tsvd", reg_param=J.shape[0])


@pytest.mark.acceptance
def test_ACCEPTANCE_exact_operator_beats_declared_nulls() -> None:
    J = np.array([[-1.1, 0.1, 0.0], [0.0, -0.9, 0.1], [0.0, 0.0, -1.2]])
    measurement = _measurement_for_operator(J)
    result = ao.compare(
        measurement,
        {"exact": J},
        nulls=("shuffled_edges", "random_init"),
        n_null=24,
        seed=5,
    )["exact"]
    assert result.metrics["operator_relative_error"] == pytest.approx(0.0, abs=1e-10)
    assert result.metrics["equation_relative_residual"] == pytest.approx(0.0, abs=1e-10)
    assert result.metrics["spectral_wasserstein"] == pytest.approx(0.0, abs=1e-10)
    assert result.metrics["hyperbolicity_agreement"] == 1.0
    shuffled = result.null_metrics["shuffled_edges:operator_relative_error"]
    random_init = result.null_metrics["random_init:operator_relative_error"]
    assert np.mean(shuffled) > result.metrics["operator_relative_error"]
    assert np.mean(random_init) > result.metrics["operator_relative_error"]


@pytest.mark.acceptance
def test_ACCEPTANCE_partial_measurement_blocks_spectral_metrics() -> None:
    J = np.diag([-1.0, -1.2, -0.8])
    full = _measurement_for_operator(J, m=6)
    partial = ao.measure_from_sensitivity(full.S[:, :2], full.U[:, :2], reg="tsvd", reg_param=2)
    result = ao.compare(partial, {"candidate": J}, n_null=2)["candidate"]
    assert np.isfinite(result.metrics["operator_relative_error"])
    assert np.isnan(result.metrics["spectral_wasserstein"])
    assert "blocked" in result.metadata["spectral_status"]


@pytest.mark.acceptance
def test_ACCEPTANCE_operator_archetypes_obey_simplex_and_select_k_with_heldout_guides() -> None:
    operators = [
        np.array([[-1.0, 0.1, 0.0], [0.1, -1.1, 0.0], [0.0, 0.0, -0.9]]),
        np.array([[-1.8, 0.05, 0.0], [0.05, -1.5, 0.1], [0.0, 0.1, -1.6]]),
        np.array([[-0.7, -0.1, 0.0], [-0.1, -0.8, 0.0], [0.0, 0.0, -0.6]]),
    ]
    measurements = [
        _measurement_for_operator(operator, seed=index + 20, m=10)
        for index, operator in enumerate(operators)
    ]
    result = ao.fit_archetypes(
        measurements,
        mode="operator",
        k="cv",
        max_k=3,
        n_splits=2,
        holdout_fraction=0.25,
        seed=4,
    )
    assert result.mode == "operator"
    assert result.metadata["selection"] == "guide-held-out equation residual"
    assert result.archetypes.shape[1:] == (3, 3)
    assert set(result.candidate_errors) == {1, 2, 3}
    assert np.all(result.weights >= -1e-12)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-8)


def test_spectral_archetypes_and_transfer_test_return_finite_diagnostics() -> None:
    source = [np.diag([-1.0, -1.1]), np.diag([-1.8, -1.7]), np.diag([-0.6, -0.7])]
    target = [np.diag([-1.2, -1.0]), np.diag([-0.8, -0.9])]
    result = ao.fit_archetypes(source, mode="spectral", k=2)
    assert result.archetypes.shape == (2, 4)
    transfer = ao.transfer_test(source, target, mode="spectral", k=2)
    assert np.isfinite(transfer.transfer_error)
    assert np.isfinite(transfer.error_ratio)
    np.testing.assert_allclose(transfer.weights.sum(axis=1), 1.0, atol=1e-8)


def test_phase2_gate_requires_complete_above_null_evidence(
    tmp_path: pytest.TempPathFactory,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ao.AnchorOpError, match="No evidence record"):
        verify_phase2_gate(missing)
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"above_null": True}), encoding="utf-8")
    with pytest.raises(ao.AnchorOpError, match="incomplete"):
        verify_phase2_gate(incomplete)
    complete = tmp_path / "complete.json"
    complete.write_text(
        json.dumps(
            {
                "preregistration_commit": "abc",
                "benchmark_commit": "def",
                "heldout_metric": 0.1,
                "null_metric": 0.5,
                "above_null": True,
            }
        ),
        encoding="utf-8",
    )
    record = verify_phase2_gate(complete)
    assert record["above_null"] is True
