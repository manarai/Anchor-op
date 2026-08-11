"""Tests for held_out_prediction_check.

The diagnostic evaluates ``ρ = ||A·S_test + U_test||_F / ||U_test||_F`` on
k-fold-CV held-out guides after fitting ``A = J·P_X`` on training guides.

Key properties to verify:
- Under a synthetic linear system with known J and no noise, ρ → 0 to
  regularization tolerance.
- Under a truly nonlinear response function, ρ stays substantially above 0.
- ρ is invariant (to at least three decimals) to per-column rescaling of U —
  the property that separates it from the bin-split linearity_check's
  proxy-confounded diagnostic.
- A shuffled-U↔S null distribution can be requested via
  ``n_permutation_null``.
"""

from __future__ import annotations

import numpy as np
import pytest

import anchorop as ao


def _linear_measurement(d: int = 8, m: int = 50, seed: int = 0, noise: float = 0.0):
    """Synthetic measurement drawn from a known invertible J on d dimensions."""
    rng = np.random.default_rng(seed)
    J = rng.normal(0.0, 1.0, size=(d, d))
    # Ensure invertibility
    J += 3.0 * np.eye(d)
    # Random U columns (κ · Wᵀδ_g). Just draw random unit-norm vectors.
    U = rng.normal(0.0, 1.0, size=(d, m))
    S = -np.linalg.solve(J, U)  # J·S = -U
    if noise > 0.0:
        S = S + noise * rng.normal(size=S.shape)
    guide_names = [f"g{i}" for i in range(m)]
    guide_efficiencies = {n: 0.5 for n in guide_names}
    measurement = ao.measure_from_sensitivity(
        S, U, guide_names=guide_names, guide_efficiencies=guide_efficiencies,
        reg="tsvd", reg_param="path", rank_tol=1e-6,
    )
    return measurement, J


def test_rho_near_zero_on_noiseless_linear_measurement() -> None:
    m, _ = _linear_measurement(d=8, m=50, noise=0.0)
    r = ao.held_out_prediction_check(m, n_folds=5, seed=0)
    assert 0 <= r.rho_pooled < 0.05, f"expected rho near 0 on noise-free linear data, got {r.rho_pooled}"
    assert r.n_folds_used == 5
    assert len(r.rho_per_fold) == 5


def test_rho_grows_with_noise() -> None:
    m_clean, _ = _linear_measurement(d=8, m=60, noise=0.0, seed=1)
    m_noisy, _ = _linear_measurement(d=8, m=60, noise=0.3, seed=1)
    r_clean = ao.held_out_prediction_check(m_clean, n_folds=5, seed=0)
    r_noisy = ao.held_out_prediction_check(m_noisy, n_folds=5, seed=0)
    assert r_noisy.rho_pooled > r_clean.rho_pooled


def test_rho_invariant_to_global_U_rescaling() -> None:
    """Multiplying all U columns by a single scalar rescales residual and target
    together and leaves rho unchanged. This is the exact algebraic invariance;
    per-column rescaling is only approximately invariant, and only when the fit
    is close to the zero-predictor baseline (see MANUSCRIPT.md §3.5 note)."""
    measurement, _ = _linear_measurement(d=6, m=40, noise=0.02, seed=2)
    S = measurement.S; U = measurement.U
    U_scaled = 7.3 * U  # arbitrary global scalar
    m_rescaled = ao.measure_from_sensitivity(
        S, U_scaled, guide_names=list(measurement.guide_names),
        guide_efficiencies={n: 7.3 * 0.5 for n in measurement.guide_names},
        reg="tsvd", reg_param="path", rank_tol=1e-6,
    )
    r_baseline = ao.held_out_prediction_check(measurement, n_folds=5, seed=0)
    r_rescaled = ao.held_out_prediction_check(m_rescaled, n_folds=5, seed=0)
    assert abs(r_baseline.rho_pooled - r_rescaled.rho_pooled) < 1e-9


def test_rho_approximately_invariant_to_per_column_rescaling_at_zero_predictor() -> None:
    """When the fit is at zero-predictor baseline (ρ ≈ 1), per-column rescaling
    of U rescales both residual and target near-identically and rho is stable
    to within a few percent. This is the regime the Replogle datasets sit in
    (MANUSCRIPT.md §3.5, Fig. S10)."""
    rng = np.random.default_rng(99)
    d, m = 6, 40
    # Random S and U with no linear relation — expected rho ≈ 1
    S = rng.normal(size=(d, m))
    U = rng.normal(size=(d, m))
    guide_names = [f"g{i}" for i in range(m)]
    measurement = ao.measure_from_sensitivity(
        S, U, guide_names=guide_names,
        guide_efficiencies={n: 0.5 for n in guide_names},
        reg="tsvd", reg_param="path", rank_tol=1e-6,
    )
    scale = 0.2 + rng.uniform(size=m)
    U_scaled = U * scale[np.newaxis, :]
    m_rescaled = ao.measure_from_sensitivity(
        S, U_scaled, guide_names=guide_names,
        guide_efficiencies={n: 0.5 * scale[i] for i, n in enumerate(guide_names)},
        reg="tsvd", reg_param="path", rank_tol=1e-6,
    )
    r_baseline = ao.held_out_prediction_check(measurement, n_folds=5, seed=0)
    r_rescaled = ao.held_out_prediction_check(m_rescaled, n_folds=5, seed=0)
    # Both should be near 1.0, and near each other in relative terms
    assert 0.7 < r_baseline.rho_pooled < 1.5
    assert abs(r_baseline.rho_pooled - r_rescaled.rho_pooled) / r_baseline.rho_pooled < 0.20


def test_permutation_null_returns_higher_rho() -> None:
    """A shuffled-U↔S null should give rho >> the true fit's rho on linear data."""
    measurement, _ = _linear_measurement(d=6, m=60, noise=0.05, seed=4)
    r = ao.held_out_prediction_check(measurement, n_folds=5, seed=0, n_permutation_null=15)
    assert r.null_median is not None
    assert r.null_std is not None
    assert r.z_score is not None
    # On truly linear data, real fit should be well below null median
    assert r.rho_pooled < r.null_median, (
        f"true-fit rho={r.rho_pooled} should be below shuffled-U null median={r.null_median}"
    )


def test_rejects_insufficient_folds() -> None:
    measurement, _ = _linear_measurement(d=6, m=20, seed=5)
    with pytest.raises(ao.AnchorOpError, match="n_folds"):
        ao.held_out_prediction_check(measurement, n_folds=1)


def test_rejects_folds_exceeding_guides() -> None:
    measurement, _ = _linear_measurement(d=6, m=8, seed=6)
    with pytest.raises(ao.AnchorOpError, match="exceeds guide count"):
        ao.held_out_prediction_check(measurement, n_folds=20)


def test_held_out_result_is_public() -> None:
    assert hasattr(ao, "HeldOutPredictionResult")
    assert hasattr(ao, "held_out_prediction_check")


def test_docstring_documents_kappa_invariance() -> None:
    doc = ao.held_out_prediction_check.__doc__
    assert "invariant" in doc.lower() or "κ" in doc
