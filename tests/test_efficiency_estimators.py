"""Tests for the two efficiency estimators.

The `detection_rate` estimator is designed to resist the "dropout-driven
pseudo-perfect knockdown" pathology documented in
`examples/05_linearity_diagnostics.ipynb`: on the K562 aggregate, 35/72
retained guides had `mean_ratio` efficiency ≈ 1.0 not because target
transcript was really wiped out, but because the target's control mean was
already at the dropout floor (low baseline).

These tests validate that:
- Both estimators exist as public exports.
- On a fixture that mimics the dropout pathology, `detection_rate` gives a
  meaningful low efficiency while `mean_ratio` degenerates to ≈ 1.0.
- On a fixture with real knockdown (baseline well above dropout floor), both
  estimators agree.
- `measure_operator`'s default is `"detection_rate"` and is recorded in the
  report notes for provenance.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import anchorop as ao


def _dropout_dominated_target_fixture(
    n_control: int = 400,
    n_perturbed: int = 100,
    seed: int = 3,
):
    """AnnData-like fixture where the target gene is at the dropout floor.

    Control target: 3/400 cells detect any UMI, all values are 1.
    Perturbed target: 0/100 cells detect (looks like 100% knockdown to mean_ratio
    but is really just dropout).
    Two other genes carry the perturbation signal so the pipeline can run.
    """
    rng = np.random.default_rng(seed)
    X_ctrl = np.zeros((n_control, 3))
    X_ctrl[:, 0] = rng.gamma(2.0, 1.0, n_control) + 1.0
    X_ctrl[:, 1] = rng.gamma(2.0, 1.0, n_control) + 1.0
    # Target gene: sparse detection
    detect_ctrl = rng.random(n_control) < 0.008
    X_ctrl[detect_ctrl, 2] = 1.0

    X_pert = np.zeros((n_perturbed, 3))
    # Shifted expression to give a real Δz signal that anchor-op can use
    X_pert[:, 0] = rng.gamma(2.0, 1.0, n_perturbed) + 0.6
    X_pert[:, 1] = rng.gamma(2.0, 1.0, n_perturbed) + 1.4
    # Target gene: still no detection (dropout floor pathology)
    # (leave all zeros)

    X = np.vstack([X_ctrl, X_pert])
    obs = pd.DataFrame({
        "guide": ["non-targeting"] * n_control + ["guide_g2"] * n_perturbed,
        "target_gene": [""] * n_control + ["g2"] * n_perturbed,
    })
    return SimpleNamespace(X=X, obs=obs, var_names=np.array(["g0", "g1", "g2"]))


def _real_knockdown_target_fixture(
    n_control: int = 200,
    n_perturbed: int = 80,
    seed: int = 4,
):
    """Target gene is well expressed in controls with mild dropout and clearly reduced
    in perturbed cells with heavy dropout — mirrors real Perturb-seq. Both estimators
    should report a substantial (non-degenerate) efficiency."""
    rng = np.random.default_rng(seed)
    X_ctrl = np.zeros((n_control, 3))
    X_ctrl[:, 0] = rng.gamma(2.0, 1.0, n_control) + 1.0
    X_ctrl[:, 1] = rng.gamma(2.0, 1.0, n_control) + 1.0
    # High baseline with 5% dropout
    detect_ctrl = rng.random(n_control) < 0.95
    X_ctrl[detect_ctrl, 2] = rng.gamma(3.0, 1.5, detect_ctrl.sum()) + 2.0

    X_pert = np.zeros((n_perturbed, 3))
    X_pert[:, 0] = rng.gamma(2.0, 1.0, n_perturbed) + 0.6
    X_pert[:, 1] = rng.gamma(2.0, 1.0, n_perturbed) + 1.4
    # ~70% knockdown: only 30% of perturbed cells still detect target
    detect_pert = rng.random(n_perturbed) < 0.30
    X_pert[detect_pert, 2] = 0.3 * (rng.gamma(3.0, 1.5, detect_pert.sum()) + 2.0)

    X = np.vstack([X_ctrl, X_pert])
    obs = pd.DataFrame({
        "guide": ["non-targeting"] * n_control + ["guide_g2"] * n_perturbed,
        "target_gene": [""] * n_control + ["g2"] * n_perturbed,
    })
    return SimpleNamespace(X=X, obs=obs, var_names=np.array(["g0", "g1", "g2"]))


def test_both_estimators_are_public() -> None:
    assert hasattr(ao, "estimate_knockdown_efficiency")
    assert hasattr(ao, "estimate_knockdown_efficiency_detection_rate")


def test_measure_operator_default_is_detection_rate() -> None:
    sig = inspect.signature(ao.measure_operator)
    assert sig.parameters["efficiency_estimator"].default == "detection_rate"


@pytest.mark.acceptance
def test_ACCEPTANCE_detection_rate_resists_dropout_pseudo_perfect_knockdown() -> None:
    adata = _dropout_dominated_target_fixture()
    basis = ao.make_program_basis(
        np.eye(3), adata.var_names, normalize=False, control_count=400
    )
    # mean_ratio: control mean ≈ 0.008, perturbed mean = 0 → efficiency = 1.0
    n_pert = int((adata.obs["guide"] == "guide_g2").sum())
    ctrl_mask = np.zeros(len(adata.obs), dtype=bool)
    ctrl_mask[: 400] = True
    pert_mask = ~ctrl_mask
    eff_mean = ao.estimate_knockdown_efficiency(
        adata.X,
        target_index=2,
        perturbed_mask=pert_mask,
        control_mask=ctrl_mask,
    )
    eff_det = ao.estimate_knockdown_efficiency_detection_rate(
        adata.X,
        target_index=2,
        perturbed_mask=pert_mask,
        control_mask=ctrl_mask,
    )
    # mean_ratio degenerates to ~1.0
    assert eff_mean >= 0.99
    # detection_rate gives a small (real) value corresponding to the ~0.8%
    # detection loss — well below the 0.99 artifact
    assert eff_det < 0.05
    assert eff_det >= 0.0

    # And measure_operator with the default (detection_rate) should filter this
    # dropout-pathology guide out at min_knockdown_efficiency=0.05.
    with pytest.raises(ao.AnchorOpError, match="No informative perturbation"):
        ao.measure_operator(
            adata, basis,
            guide_key="guide", target_key="target_gene",
            control_label="non-targeting",
            min_cells_per_guide=2, min_knockdown_efficiency=0.05,
            reg="tsvd", reg_param="path",
        )

    # With the legacy mean_ratio estimator, the same guide would silently pass.
    # This is exactly the pathology the new default guards against.
    _ = ao.measure_operator(
        adata, basis,
        guide_key="guide", target_key="target_gene",
        control_label="non-targeting",
        min_cells_per_guide=2, min_knockdown_efficiency=0.05,
        reg="tsvd", reg_param="path",
        efficiency_estimator="mean_ratio",
    )


def test_both_estimators_agree_on_real_knockdown() -> None:
    adata = _real_knockdown_target_fixture()
    ctrl_mask = np.zeros(len(adata.obs), dtype=bool)
    ctrl_mask[: 200] = True
    pert_mask = ~ctrl_mask
    eff_mean = ao.estimate_knockdown_efficiency(
        adata.X, target_index=2, perturbed_mask=pert_mask, control_mask=ctrl_mask,
    )
    eff_det = ao.estimate_knockdown_efficiency_detection_rate(
        adata.X, target_index=2, perturbed_mask=pert_mask, control_mask=ctrl_mask,
    )
    # Both should report a substantial knockdown, though not identical values
    # (mean_ratio integrates level + dropout change; detection_rate reports the
    # dropout change alone).
    assert 0.5 < eff_mean < 0.99
    # detection_rate here reflects ~0.95 - ~0.30 = ~0.65
    assert 0.4 < eff_det < 0.85


def test_notes_record_estimator_choice() -> None:
    adata = _real_knockdown_target_fixture()
    basis = ao.make_program_basis(
        np.eye(3), adata.var_names, normalize=False, control_count=200
    )
    m = ao.measure_operator(
        adata, basis,
        guide_key="guide", target_key="target_gene",
        control_label="non-targeting",
        min_cells_per_guide=2, min_knockdown_efficiency=0.01,
        reg="tsvd", reg_param="path",
        efficiency_estimator="detection_rate",
    )
    # There is only one perturbed target; the guide may still be dropped if the
    # detection change is trivial. Accept either outcome as long as the notes
    # reflect the chosen estimator.
    assert any("detection_rate" in n for n in m.report.notes)


def test_invalid_efficiency_estimator_is_rejected() -> None:
    adata = _real_knockdown_target_fixture()
    basis = ao.make_program_basis(
        np.eye(3), adata.var_names, normalize=False, control_count=200
    )
    with pytest.raises(ao.AnchorOpError, match="efficiency_estimator"):
        ao.measure_operator(
            adata, basis,
            guide_key="guide", target_key="target_gene",
            control_label="non-targeting",
            min_cells_per_guide=2, min_knockdown_efficiency=0.05,
            efficiency_estimator="not_a_real_method",
        )
