"""Regression: full_domain_identified must consider rank(U), not just rank(S).

Reviewer's finding F5: with 25 targets and 6 sgRNAs per target sharing W^T delta_g,
Jost's U at d=30 has 5 exactly-zero singular values (rank 25). Adding noise to
S can push its numerical rank back to d, and the earlier gate then falsely
declared full identification even though rank(U) = 25 < d = 30.
"""
from __future__ import annotations

import numpy as np
import pytest

import anchorop as ao
from anchorop.identifiability import make_anchor_report, regularized_pseudoinverse


def _build_rank_deficient_U(d: int, m: int, rank: int, seed: int) -> np.ndarray:
    """d x m input matrix whose column span has exactly `rank` (< d) directions."""
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(d, rank))
    R = rng.normal(size=(rank, m))
    return L @ R  # rank <= min(rank, m); we ensure rank <= d


def test_rank_deficient_U_with_full_rank_S_is_not_full_domain():
    d, m, r_true = 30, 60, 25
    rng = np.random.default_rng(20260927)

    U = _build_rank_deficient_U(d, m, r_true, seed=7)
    # Ground-truth J (invertible), so noise-free S has rank(S) = rank(U) = r_true.
    J = rng.normal(size=(d, d)) - 1.5 * np.eye(d)
    S = -np.linalg.solve(J, U)
    # Add noise so numerical rank of S is d, but rank(U) still 25.
    S_obs = S + 0.30 * rng.normal(size=S.shape)

    meas = ao.measure_from_sensitivity(
        S=S_obs, U=U,
        guide_names=[f"g_{i}" for i in range(m)],
        reg="tsvd", reg_param="path", rank_tol=1e-2,
    )
    rep = meas.report
    # After the gate fix, this must be False because rank(U) < d.
    assert rep.input_subspace_dim < d
    assert not rep.full_domain_identified, (
        f"full_domain_identified must be False when rank(U)={rep.input_subspace_dim} < d={d} "
        f"(effective_response_rank={rep.effective_response_rank}, hard_projector={rep.is_hard_projector})"
    )
    # And .J must refuse.
    with pytest.raises(ao.IdentifiabilityError):
        _ = meas.J


def test_full_rank_U_and_S_still_full_domain():
    d, m = 20, 40
    rng = np.random.default_rng(20260927)
    U = rng.normal(size=(d, m))
    J = rng.normal(size=(d, d)) - 1.5 * np.eye(d)
    S = -np.linalg.solve(J, U)
    meas = ao.measure_from_sensitivity(
        S=S, U=U,
        guide_names=[f"g_{i}" for i in range(m)],
        reg="tsvd", reg_param="path", rank_tol=1e-2,
    )
    rep = meas.report
    assert rep.input_subspace_dim == d
    assert rep.effective_response_rank == d
    assert rep.full_domain_identified
    # .J is available in this regime.
    _ = meas.J


def test_heavy_tikhonov_collapses_effective_rank_and_gate_refuses():
    """rank(U) = d but heavy Tikhonov drops retained directions below d."""
    d, m = 12, 40
    rng = np.random.default_rng(20260927)
    U = rng.normal(size=(d, m))
    J = rng.normal(size=(d, d)) - 1.5 * np.eye(d)
    S = -np.linalg.solve(J, U)
    # alpha is large enough that most sigma_i^2 < alpha, dropping effective rank.
    _, sv, _ = np.linalg.svd(S, full_matrices=False)
    alpha = float(sv[3] ** 2 * 1.5)  # keep only 3 directions
    meas = ao.measure_from_sensitivity(
        S=S, U=U,
        guide_names=[f"g_{i}" for i in range(m)],
        reg="tikhonov", reg_param=alpha, rank_tol=1e-2,
    )
    rep = meas.report
    assert rep.input_subspace_dim == d, "U should be full-rank in this construction"
    assert rep.effective_response_rank < d, (
        f"Tikhonov with alpha above the (d-3)th squared singular value should "
        f"drop effective rank below d, got {rep.effective_response_rank}"
    )
    assert not rep.full_domain_identified
    with pytest.raises(ao.IdentifiabilityError):
        _ = meas.J
