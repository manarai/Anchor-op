"""Tests for anchorop.state_space (experiment 1, Tasks 0 + 1).

The four required Task 0 tests from the preregistration plan:
1. encode shape — ``(n_cells, dim)`` on every arm.
2. jacobian shape — ``(n_cells, dim, n_genes)`` on every arm.
3. For linear reps (PCARep, FARep): ``jacobian[0] @ u_gene ==
   decode_direction(u_gene)``.
4. For scGPT: finite-difference check of the autograd Jacobian on
   5 cells. This test runs only when a scGPT checkpoint is on disk
   and the environment variable ``SCGPT_CKPT`` points at it;
   otherwise it is skipped with a clear reason so the rest of the
   suite still runs on a CPU-only development machine.

The multiome stub is also exercised: it must raise
:class:`NotImplementedError` on every call.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from anchorop.state_space import PCARep, FARep, MultiomeRep, ScGPTRep

SEED = 20260930


# ─── fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def control_matrix():
    """Deterministic synthetic control-cell matrix: n_cells × n_genes."""
    rng = np.random.default_rng(SEED)
    n_cells, n_genes = 400, 50
    # Low-rank + noise so PCA / FA have a defined signal.
    rank = 8
    U = rng.standard_normal((n_cells, rank))
    V = rng.standard_normal((rank, n_genes)) * 0.5
    X = U @ V + 0.1 * rng.standard_normal((n_cells, n_genes))
    return X.astype(np.float64)


@pytest.fixture
def u_gene(control_matrix):
    """A representative gene-space perturbation direction (unit norm,
    negative sign on an arbitrary target gene)."""
    _, n_genes = control_matrix.shape
    u = np.zeros(n_genes); u[7] = -1.0
    return u


# ─── encode / jacobian shape ───────────────────────────────────────────
@pytest.mark.parametrize("cls,dim", [(PCARep, 10), (FARep, 10)])
def test_encode_shape(control_matrix, cls, dim):
    rep = cls(dim=dim).fit(control_matrix)
    Z = rep.encode(control_matrix)
    assert Z.shape == (control_matrix.shape[0], dim), (
        f"{rep.name} encode shape {Z.shape} != "
        f"({control_matrix.shape[0]}, {dim})")


@pytest.mark.parametrize("cls,dim", [(PCARep, 10), (FARep, 10)])
def test_jacobian_shape(control_matrix, cls, dim):
    rep = cls(dim=dim).fit(control_matrix)
    n = control_matrix.shape[0]
    g = control_matrix.shape[1]
    J = rep.jacobian(control_matrix)
    assert J.shape == (n, dim, g), (
        f"{rep.name} jacobian shape {J.shape} != ({n}, {dim}, {g})")


# ─── linearity: jacobian @ u_gene == decode_direction ──────────────────
@pytest.mark.parametrize("cls,dim", [(PCARep, 10), (FARep, 10)])
def test_linear_jacobian_matches_decode_direction(control_matrix, u_gene, cls, dim):
    rep = cls(dim=dim).fit(control_matrix)
    J = rep.jacobian(control_matrix)
    u_z_from_J = J[0] @ u_gene
    u_z_from_decode = rep.decode_direction(u_gene)
    np.testing.assert_allclose(u_z_from_J, u_z_from_decode, atol=1e-10,
                               err_msg=(f"{rep.name}: jacobian[0] @ u_gene "
                                        "does not match decode_direction(u_gene) "
                                        "— linear encoders must agree exactly."))


# ─── linearity: every cell gets the same Jacobian ──────────────────────
@pytest.mark.parametrize("cls,dim", [(PCARep, 10), (FARep, 10)])
def test_linear_jacobian_is_constant_across_cells(control_matrix, cls, dim):
    rep = cls(dim=dim).fit(control_matrix)
    J = rep.jacobian(control_matrix)
    np.testing.assert_allclose(J[0], J[-1], atol=1e-12,
                               err_msg=f"{rep.name}: Jacobian varies across cells; "
                                       "it must be constant for a linear encoder.")


# ─── A4 (revised) knockdown-scale closed-form check on linear reps ─────
@pytest.mark.parametrize("cls,dim", [(PCARep, 10), (FARep, 10)])
@pytest.mark.parametrize("kappa", [0.5, 0.7, 0.9])
def test_linear_knockdown_scale_linear_space(
        control_matrix, cls, dim, kappa):
    """A4 revision (2026-09-30), linear-arm consistency, linear input
    space. For a linear encoder E(x) = (x − μ)·J.T with J =
    components_, the knockdown-scale FD under ``input_space='linear'``
    reduces to −κ · mean(X_ctrl[:, g]) · J[:, g]. Enforced to
    atol=1e-10 across (PCARep, FARep) × κ.
    """
    rep = cls(dim=dim).fit(control_matrix)
    g = 7
    u_z_fd = rep.knockdown_scale_difference(
        control_matrix, g, kappa=kappa, input_space="linear")
    J0 = rep.jacobian(control_matrix)[0]
    xbar_g = control_matrix[:, g].mean()
    u_z_closed = -kappa * xbar_g * J0[:, g]
    np.testing.assert_allclose(
        u_z_fd, u_z_closed, atol=1e-10,
        err_msg=(f"{rep.name}: knockdown-scale (linear) does not match "
                 "the closed form −κ · mean(X_ctrl[:, g]) · J @ δ_g."))


# ─── A4 (amendment 2) log1p knockdown-scale on linear reps ─────────────
@pytest.fixture(scope="module")
def lognorm_matrix():
    """Synthetic positive-expression log1p-normalised matrix for the
    log-space knockdown-scale test (the real-data equivalent is the
    Replogle K562 essential raw-counts h5ad once log-normalised in
    the exp1 pipeline)."""
    rng = np.random.default_rng(SEED + 1)
    n_cells, n_genes = 400, 50
    # Positive-mean lognormal-ish counts, then log1p.
    counts = rng.gamma(shape=1.5, scale=2.0, size=(n_cells, n_genes))
    return np.log1p(counts)


@pytest.mark.parametrize("cls,dim", [(PCARep, 10), (FARep, 10)])
@pytest.mark.parametrize("kappa", [0.5, 0.7, 0.9])
def test_linear_knockdown_scale_log1p_space(
        lognorm_matrix, cls, dim, kappa):
    """A4 amendment 2 (2026-09-30), log1p-space linear consistency.

    For a linear encoder on log1p-normalised inputs, the knockdown-
    scale FD equals ``mean_i(δx_i[g]) · J[:, g]`` where
    ``δx_i[g] = log1p((1-κ) · expm1(x_i[g])) − x_i[g]``. Enforced to
    atol=1e-10 across (PCARep, FARep) × κ.
    """
    rep = cls(dim=dim).fit(lognorm_matrix)
    g = 7
    u_z_fd = rep.knockdown_scale_difference(
        lognorm_matrix, g, kappa=kappa, input_space="log1p")
    J0 = rep.jacobian(lognorm_matrix)[0]
    y = np.expm1(lognorm_matrix[:, g])
    delta_g = np.log1p((1.0 - kappa) * np.clip(y, 0.0, None)) - lognorm_matrix[:, g]
    u_z_closed = float(delta_g.mean()) * J0[:, g]
    np.testing.assert_allclose(
        u_z_fd, u_z_closed, atol=1e-10,
        err_msg=(f"{rep.name}: knockdown-scale (log1p) does not match "
                 "mean(log1p((1-κ) expm1(x[g])) - x[g]) · J[:, g]."))


def test_knockdown_scale_zero_on_residuals():
    """Residual h5ad verification (recorded in EXPERIMENT_LOG.md,
    2026-09-30): when the per-gene control-cell mean is 0 (residual
    data), the knockdown-scale FD under ``input_space='linear'`` is
    exactly 0. This is the motivation for amendment 2 (switch all
    arms to log1p-normalised raw counts)."""
    rng = np.random.default_rng(SEED + 2)
    X = rng.standard_normal((300, 50))
    X -= X.mean(axis=0)  # make per-gene control mean exactly 0
    rep = PCARep(dim=10).fit(X)
    for g in range(5):
        u_z = rep.knockdown_scale_difference(
            X, g, kappa=0.7, input_space="linear")
        np.testing.assert_allclose(u_z, 0.0, atol=1e-10,
            err_msg=f"knockdown-scale on residuals should be 0 at gene {g}; got {u_z}")


# ─── PCARep sanity: encode via mean-centred projection equals sklearn ──
def test_pcarep_encode_matches_sklearn(control_matrix):
    from sklearn.decomposition import PCA
    rep = PCARep(dim=10).fit(control_matrix)
    Z_ours = rep.encode(control_matrix)
    pca = PCA(n_components=10, svd_solver="auto", random_state=0).fit(control_matrix)
    Z_sk = pca.transform(control_matrix)
    # Sign of principal components is arbitrary; compare absolute per-column.
    np.testing.assert_allclose(np.abs(Z_ours), np.abs(Z_sk), atol=1e-8)


# ─── fit guard ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("cls", [PCARep, FARep])
def test_use_before_fit_raises(control_matrix, cls):
    rep = cls(dim=10)
    with pytest.raises(RuntimeError, match="must be fit"):
        rep.encode(control_matrix)
    with pytest.raises(RuntimeError, match="must be fit"):
        rep.jacobian(control_matrix)


# ─── multiome stub ─────────────────────────────────────────────────────
def test_multiome_stub_raises(control_matrix, u_gene):
    rep = MultiomeRep(dim=10)
    with pytest.raises(NotImplementedError):
        rep.fit(control_matrix)
    with pytest.raises(NotImplementedError):
        rep.encode(control_matrix)
    with pytest.raises(NotImplementedError):
        rep.jacobian(control_matrix)
    with pytest.raises(NotImplementedError):
        rep.decode_direction(u_gene)


# ─── scGPT: A4 feasibility, skipped unless the checkpoint is present ───
_SCGPT_CKPT = os.environ.get("SCGPT_CKPT")


@pytest.mark.skipif(
    not _SCGPT_CKPT or not os.path.exists(_SCGPT_CKPT),
    reason=(
        "scGPT checkpoint not present. Set $SCGPT_CKPT to the path of a "
        "pretrained scGPT_human checkpoint to run the A4 Jacobian "
        "feasibility test. Fine on CPU; 5 cells × 3 step sizes."))
def test_scgpt_jacobian_feasibility_on_5_cells(control_matrix):
    """A4 feasibility: autograd Jacobian vs finite differences on 5
    cells across several step sizes. Pass → the scGPT arm proceeds.
    Fail → the arm is halted and the report is attached to the test
    failure for inspection."""
    rep = ScGPTRep(d_out=10, checkpoint_path=_SCGPT_CKPT).fit(control_matrix)
    report = rep.check_jacobian_feasibility(
        X_control=control_matrix,
        n_cells=5,
        step_sizes=(1e-3, 1e-2, 1e-1),
        tol_cosine=0.9,
    )
    assert report["halt_reason"] is None, (
        f"A4 halt invoked for scGPT arm: {report['halt_reason']!r}; "
        f"per-cell cosines = {report['per_cell_cosine']}")
    assert report["stable"], (
        f"A4 stability fail; per-cell cosines = {report['per_cell_cosine']}")
    # Also check that the checkpoint provenance was recorded.
    ckpt = report["checkpoint"]
    assert ckpt is not None and ckpt.sha256 and ckpt.native_dim > 0
