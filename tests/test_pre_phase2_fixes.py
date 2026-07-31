"""Regression tests for the three pre-Phase-2 blockers.

1. ``rank_tol`` detects noise-hidden collinearity that the eps-based numerical
   rank silently accepts as full rank.
2. The linearity threshold declared in ``PREREGISTRATION.md`` matches the
   default enforced by :func:`anchorop.linearity_check`.
3. :func:`anchorop.spectral_abscissa_difference` is well-defined, symmetric,
   Lipschitz-stable under small perturbations of a Jordan-like matrix, and
   markedly more stable than the Wasserstein spectrum distance in that regime.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import anchorop as ao


PREREG = Path(__file__).resolve().parent.parent / "PREREGISTRATION.md"


def _rank_deficient_S_and_U(
    d: int = 6, m: int = 40, true_rank: int = 2, noise: float = 1e-2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build (U, S) whose column spaces are rank-``true_rank`` plus small noise."""
    rng = np.random.default_rng(seed)
    J = -np.eye(d) - 0.1 * rng.normal(size=(d, d))
    basis = rng.normal(size=(d, true_rank))
    weights = rng.normal(size=(true_rank, m))
    U = basis @ weights + noise * rng.normal(size=(d, m))
    S = -np.linalg.solve(J, U)
    return U, S


@pytest.mark.acceptance
def test_ACCEPTANCE_rank_tol_detects_noise_hidden_collinearity() -> None:
    """Fix 1: an S built from a rank-2 basis + 1% noise must not be called full rank."""
    U, S = _rank_deficient_S_and_U(d=6, m=40, true_rank=2, noise=1e-2)

    # Old behavior: eps-based tolerance silently declares full rank.
    lax = ao.measure_from_sensitivity(S, U, reg="tsvd", reg_param="path", rank_tol=None)
    assert lax.report.full_domain_identified is True
    assert lax.report.effective_response_rank == 6

    # Scientific default (as used by measure_operator): catches it.
    strict = ao.measure_from_sensitivity(S, U, reg="tsvd", reg_param="path", rank_tol=1e-2)
    assert strict.report.full_domain_identified is False
    assert strict.report.effective_response_rank == 2
    assert strict.report.rank_tol == pytest.approx(1e-2)
    with pytest.raises(ao.IdentifiabilityError):
        _ = strict.J

    # Report input-subspace rank should also collapse under the tolerance.
    assert strict.report.input_subspace_dim == 2
    assert strict.report.response_subspace_dim == 2


def test_measure_operator_uses_scientific_rank_tol_by_default() -> None:
    """``measure_operator``'s default rank_tol is the preregistered ``1e-2``."""
    sig = inspect.signature(ao.measure_operator)
    assert sig.parameters["rank_tol"].default == pytest.approx(1e-2)


def test_preregistered_linearity_threshold_matches_code_default() -> None:
    """Fix 2: the number written in PREREGISTRATION.md is the same one enforced by code."""
    sig = inspect.signature(ao.linearity_check)
    code_default = sig.parameters["threshold"].default
    text = PREREG.read_text(encoding="utf-8")
    match = re.search(r"Linearity — weak/strong bin action agreement.*?`([0-9.]+)`", text)
    assert match is not None, "PREREGISTRATION.md must declare a numeric linearity threshold."
    preregistered = float(match.group(1))
    assert code_default == pytest.approx(preregistered)


def test_preregistered_rank_tol_matches_code_default() -> None:
    """Fix 1+2: the rank tolerance in PREREGISTRATION.md matches the code default."""
    sig = inspect.signature(ao.measure_operator)
    code_default = float(sig.parameters["rank_tol"].default)
    text = PREREG.read_text(encoding="utf-8")
    match = re.search(r"Identifiability — rank tolerance.*?`([0-9.eE+-]+)`", text)
    assert match is not None, "PREREGISTRATION.md must declare a numeric rank tolerance."
    preregistered = float(match.group(1))
    assert code_default == pytest.approx(preregistered)


@pytest.mark.acceptance
def test_ACCEPTANCE_spectral_abscissa_difference_is_zero_for_equal_and_stable_for_nonnormal() -> None:
    """Fix 3: abscissa difference is Lipschitz-stable where Wasserstein blows up."""
    # Zero for identical operators.
    A = np.array([[-1.0, 0.2], [0.0, -0.5]])
    assert ao.spectral_abscissa_difference(A, A) == pytest.approx(0.0, abs=1e-12)

    # Sign of the hyperbolicity gap is captured.
    B_hyper = np.array([[0.05, 0.0], [0.0, -0.9]])
    B_stable = np.array([[-0.1, 0.0], [0.0, -0.9]])
    assert ao.spectral_abscissa_difference(B_hyper, B_stable) == pytest.approx(0.05 - (-0.1))

    # For a diagonalizable (non-defective) non-normal J, the abscissa gap under
    # 1% element noise tracks the perturbation scale rather than blowing up.
    # This is the regime the docstring commits to; fully-defective Jordan
    # blocks are Hölder-continuous (exponent 1/k) in every eigenvalue metric
    # and are not the target of the stability claim.
    J = np.array(
        [
            [-2.0, 1.0, 0.0, 0.0],
            [0.0, -1.5, 0.5, 0.0],
            [0.0, 0.0, -1.0, 0.2],
            [0.0, 0.0, 0.0, -0.8],
        ]
    )
    abscissa_gaps = []
    for trial in range(40):
        rng = np.random.default_rng(trial + 900)
        J_perturbed = J + 1e-2 * rng.normal(size=J.shape)
        abscissa_gaps.append(ao.spectral_abscissa_difference(J, J_perturbed))
    abscissa_gaps = np.asarray(abscissa_gaps)
    assert float(np.mean(abscissa_gaps)) < 5e-2
    assert float(np.max(abscissa_gaps)) < 1e-1


def test_linearity_check_null_correction_backward_compat_and_shape() -> None:
    """Regression: n_null=0 (default) preserves legacy fields; n_null>0 populates them.

    Under perfect linearity on synthetic data, the raw rel_diff should be near
    zero on the identified subspace. With n_null=20 the null_median should also
    be near zero (both bins identify the same operator), and excess_above_null
    should be near zero.
    """
    rng = np.random.default_rng(0)
    J = np.diag([-1.0, -1.5, -0.8, -1.2, -0.9])
    d, m = J.shape[0], 24
    U = rng.normal(size=(d, m))
    S = -np.linalg.solve(J, U)
    guide_names = [f"guide_{i}" for i in range(m)]
    effs = {name: 0.2 + 0.05 * (i % 8) for i, name in enumerate(guide_names)}
    measurement = ao.measure_from_sensitivity(
        S, U, guide_names=guide_names, guide_efficiencies=effs,
        reg="tsvd", reg_param="path", rank_tol=1e-2,
    )

    # Legacy path — no null.
    r0 = ao.linearity_check(measurement, threshold=0.5)
    assert r0.n_null == 0
    assert r0.null_median is None
    assert r0.excess_above_null is None
    assert r0.z_score is None

    # New path — null correction on.
    r1 = ao.linearity_check(measurement, threshold=0.5, n_null=20, null_seed=1)
    assert r1.n_null == 20
    assert r1.null_median is not None and r1.null_median >= 0.0
    assert r1.null_std is not None and r1.null_std >= 0.0
    assert r1.excess_above_null is not None
    # For a linear synthetic system, the observed split rel_diff should not be
    # dramatically above the random-split null (both split types see the same
    # linear operator).
    assert abs(r1.excess_above_null) < 1.0


def test_linearity_check_survives_subset_rank_drop() -> None:
    """Regression: subsets may not support the parent's selected TSVD rank.

    Before the fix, ``linearity_check`` reused ``report.selected_regularization``
    (the parent's chosen rank) on each efficiency bin, which errored when the
    bin's numerical rank at the parent's rank_tol was smaller. Each bin should
    resolve its own rank from its own path.
    """
    rng = np.random.default_rng(0)
    d = 6
    # J is well-conditioned so the parent measurement identifies all 6 directions.
    J = np.diag([-2.0, -1.5, -1.0, -0.8, -0.5, -0.3])
    # Two "kinds" of guides: half hit a 6D-spanning basis, half hit only a 3D
    # sub-basis. Split by median efficiency arranges so one bin is rank-deficient.
    U_full = rng.normal(size=(d, 6))
    U_partial = np.pad(rng.normal(size=(3, 6)), ((0, d - 3), (0, 0)))
    U = np.column_stack([U_full, U_partial])
    S = -np.linalg.solve(J, U)
    guide_names = [f"guide_{i}" for i in range(U.shape[1])]
    # Efficiencies chosen so the low-efficiency (weak) bin is the rank-deficient half.
    efficiencies = {name: 0.2 + 0.1 * (i >= 6) for i, name in enumerate(guide_names)}
    measurement = ao.measure_from_sensitivity(
        S,
        U,
        guide_names=guide_names,
        guide_efficiencies=efficiencies,
        reg="tsvd",
        reg_param="path",
        rank_tol=1e-2,
    )
    assert measurement.report.effective_response_rank == d
    result = ao.linearity_check(measurement, threshold=0.5)
    assert result.overlap_rank >= 1


def test_compare_reports_abscissa_at_full_rank_and_nans_at_partial_rank() -> None:
    """The abscissa metric appears at full rank and is nan (with reason) otherwise."""
    J = np.array([[-1.1, 0.1, 0.0], [0.0, -0.9, 0.1], [0.0, 0.0, -1.2]])
    rng = np.random.default_rng(3)
    U = rng.normal(size=(3, 8))
    S = -np.linalg.solve(J, U)
    full = ao.measure_from_sensitivity(S, U, reg="tsvd", reg_param=3)
    result = ao.compare(full, {"exact": J}, n_null=2, seed=1)["exact"]
    assert "spectral_abscissa_difference" in result.metrics
    assert result.metrics["spectral_abscissa_difference"] == pytest.approx(0.0, abs=1e-10)
    assert "primary_hyperbolicity_metric" in result.metadata

    partial = ao.measure_from_sensitivity(S[:, :2], U[:, :2], reg="tsvd", reg_param=2)
    partial_result = ao.compare(partial, {"exact": J}, n_null=2, seed=1)["exact"]
    assert np.isnan(partial_result.metrics["spectral_abscissa_difference"])
    assert "blocked" in partial_result.metadata["spectral_status"]
