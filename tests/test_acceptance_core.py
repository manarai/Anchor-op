"""Load-bearing synthetic acceptance tests.

Synthetic systems are used only to verify algebraic recovery and safety guards;
they are not presented as biological evidence.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import anchorop as ao
from anchorop.identifiability import regularized_pseudoinverse


def _known_system(
    d: int = 4, m: int = 9, seed: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    J = -np.eye(d) + 0.12 * rng.normal(size=(d, d))
    U = rng.normal(size=(d, m))
    S = -np.linalg.solve(J, U)
    return J, U, S


def _mini_adata() -> SimpleNamespace:
    controls = np.tile(np.array([1.0, 1.0, 0.2]), (4, 1))
    g0 = np.tile(np.array([0.5, 1.0, 0.2]), (3, 1))
    g1 = np.tile(np.array([1.0, 0.5, 0.2]), (3, 1))
    X = np.vstack([controls, g0, g1])
    obs = pd.DataFrame(
        {
            "guide": ["non-targeting"] * 4 + ["guide_g0"] * 3 + ["guide_g1"] * 3,
            "target_gene": [""] * 4 + ["g0"] * 3 + ["g1"] * 3,
            "batch": ["b1", "b1", "b2", "b2"] + ["b1", "b1", "b2"] + ["b1", "b1", "b2"],
        }
    )
    return SimpleNamespace(X=X, obs=obs, var_names=np.array(["g0", "g1", "g2"]))


@pytest.mark.acceptance
def test_ACCEPTANCE_exact_full_rank_recovery_and_report() -> None:
    J, U, S = _known_system()
    measurement = ao.measure_from_sensitivity(
        S,
        U,
        reg="tsvd",
        reg_param=4,
        bootstrap=8,
        guide_efficiencies={f"guide_{i}": 0.8 for i in range(S.shape[1])},
    )
    np.testing.assert_allclose(measurement.J, J, atol=1e-10, rtol=1e-10)
    assert measurement.report is not None
    assert measurement.report.full_domain_identified
    assert measurement.report.effective_response_rank == 4
    assert measurement.report.bootstrap_covariance is not None
    assert measurement.report.bootstrap_actions is not None
    assert measurement.report.bootstrap_actions.shape == (8, 4, 4)
    assert len(measurement.report.regularization_path) == 4


@pytest.mark.acceptance
def test_ACCEPTANCE_partial_domain_has_correct_orientation_and_blocks_full_j() -> None:
    J, U_full, S_full = _known_system(d=4, m=8)
    U = U_full[:, :2]
    S = S_full[:, :2]
    measurement = ao.measure_from_sensitivity(S, U, reg="tsvd", reg_param=2)
    expected = J @ measurement.response_projector
    np.testing.assert_allclose(measurement.identified_action, expected, atol=1e-10, rtol=1e-10)
    assert measurement.report is not None
    assert measurement.report.effective_response_rank == 2
    assert not measurement.report.full_domain_identified
    with pytest.raises(ao.IdentifiabilityError, match="full Jacobian is not identified"):
        _ = measurement.J
    # The input and response subspaces are generally not interchangeable.
    wrong_left_projection = measurement.input_projector @ J
    assert np.linalg.norm(expected - wrong_left_projection, ord="fro") > 1e-4


@pytest.mark.acceptance
def test_ACCEPTANCE_guide_level_measurement_recovers_simple_action() -> None:
    adata = _mini_adata()
    basis = ao.make_program_basis(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        adata.var_names,
        normalize=False,
        control_count=4,
    )
    measurement = ao.measure_operator(
        adata,
        basis,
        guide_key="guide",
        target_key="target_gene",
        control_label="non-targeting",
        batch_key="batch",
        min_cells_per_guide=2,
        min_knockdown_efficiency=0.1,
        reg="tsvd",
        reg_param=2,
        # This synthetic fixture has no zeros — pin mean_ratio so the
        # efficiency has the expected 0.5 value. The detection_rate default
        # is validated separately on dropout-realistic data.
        efficiency_estimator="mean_ratio",
    )
    np.testing.assert_allclose(measurement.J, -np.eye(2), atol=1e-10, rtol=1e-10)
    assert measurement.report is not None
    assert set(measurement.report.retained_guides) == {"guide_g0", "guide_g1"}
    assert measurement.report.guide_efficiencies["guide_g0"] == pytest.approx(0.5)
    assert measurement.report.guide_efficiencies["guide_g1"] == pytest.approx(0.5)


@pytest.mark.acceptance
def test_ACCEPTANCE_regularization_path_and_tikhonov_are_inspectable() -> None:
    S = np.diag([1.0, 1e-5])
    _, selected, path, _ = regularized_pseudoinverse(S, method="tsvd", parameter=1)
    assert selected.effective_rank == 1
    assert [entry.effective_rank for entry in path] == [1, 2]
    _, selected_tikhonov, tikhonov_path, _ = regularized_pseudoinverse(
        S,
        method="tikhonov",
        parameter=1e-4,
    )
    assert selected_tikhonov.parameter == pytest.approx(1e-4)
    assert len(tikhonov_path) >= 18
    assert np.isfinite(selected_tikhonov.filter_factors).all()


@pytest.mark.acceptance
def test_ACCEPTANCE_dimension_guard_and_no_unreported_action() -> None:
    adata = _mini_adata()
    with pytest.raises(ao.DimensionGuardError):
        ao.fit_programs(
            adata,
            d=201,
            control_mask=np.array([True] * len(adata.obs)),
        )
    J, U, S = _known_system()
    unchecked = ao.MeasuredOperator(
        _identified_action=J,
        S=S,
        U=U,
        report=None,
        guide_names=tuple(f"g{i}" for i in range(S.shape[1])),
    )
    with pytest.raises(ao.IdentifiabilityError, match="No matrix action"):
        _ = unchecked.identified_action


def test_missing_controls_are_refused() -> None:
    adata = _mini_adata()
    adata.obs.loc[:, "guide"] = "guide_g0"
    basis = ao.make_program_basis(np.eye(3, 2), adata.var_names, normalize=False)
    with pytest.raises(ao.AnchorOpError, match="No matched controls"):
        ao.measure_operator(
            adata,
            basis,
            guide_key="guide",
            target_key="target_gene",
            control_label="non-targeting",
            min_cells_per_guide=2,
        )
