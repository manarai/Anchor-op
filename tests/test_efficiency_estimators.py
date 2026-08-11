"""Tests for the three efficiency estimators.

Previous versions of anchor-op defaulted to ``detection_rate`` to "rescue"
low-baseline targets from the ``mean_ratio`` 1.0-spike. A simulation
(``examples/06_estimator_simulation.ipynb``) subsequently showed that
``detection_rate`` is not an unbiased estimator of ``κ = 1 − E[X_pert]/
E[X_ctrl]`` — it is a bounded detection-shift proxy that scales with baseline
expression and systematically biases everywhere except the low-λ limit.

The corrected design defaults to ``mean_ratio`` (unbiased under both Poisson
and Poisson-with-dropout) plus a ``min_control_detection_rate`` filter that
drops information-limited targets outright. The 1.0-spike guides on the K562
aggregate reappear as filter-drops rather than as biased estimates.

These tests validate the corrected semantics:
- All three estimators are public.
- ``build_guide_responses`` defaults to ``mean_ratio``.
- Information-limited targets are dropped by the detection filter, not
  silently rescued with a biased estimator.
- Estimators agree on the well-expressed target regime.
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

    Control target: ~3/400 cells detect any UMI. Perturbed target: 0/100.
    Two other genes carry the perturbation signal so the pipeline can run.
    """
    rng = np.random.default_rng(seed)
    X_ctrl = np.zeros((n_control, 3))
    X_ctrl[:, 0] = rng.gamma(2.0, 1.0, n_control) + 1.0
    X_ctrl[:, 1] = rng.gamma(2.0, 1.0, n_control) + 1.0
    detect_ctrl = rng.random(n_control) < 0.008
    X_ctrl[detect_ctrl, 2] = 1.0

    X_pert = np.zeros((n_perturbed, 3))
    X_pert[:, 0] = rng.gamma(2.0, 1.0, n_perturbed) + 0.6
    X_pert[:, 1] = rng.gamma(2.0, 1.0, n_perturbed) + 1.4

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
    """Well-expressed target with real knockdown, mirroring healthy Perturb-seq."""
    rng = np.random.default_rng(seed)
    X_ctrl = np.zeros((n_control, 3))
    X_ctrl[:, 0] = rng.gamma(2.0, 1.0, n_control) + 1.0
    X_ctrl[:, 1] = rng.gamma(2.0, 1.0, n_control) + 1.0
    detect_ctrl = rng.random(n_control) < 0.95
    X_ctrl[detect_ctrl, 2] = rng.gamma(3.0, 1.5, detect_ctrl.sum()) + 2.0

    X_pert = np.zeros((n_perturbed, 3))
    X_pert[:, 0] = rng.gamma(2.0, 1.0, n_perturbed) + 0.6
    X_pert[:, 1] = rng.gamma(2.0, 1.0, n_perturbed) + 1.4
    detect_pert = rng.random(n_perturbed) < 0.30
    X_pert[detect_pert, 2] = 0.3 * (rng.gamma(3.0, 1.5, detect_pert.sum()) + 2.0)

    X = np.vstack([X_ctrl, X_pert])
    obs = pd.DataFrame({
        "guide": ["non-targeting"] * n_control + ["guide_g2"] * n_perturbed,
        "target_gene": [""] * n_control + ["g2"] * n_perturbed,
    })
    return SimpleNamespace(X=X, obs=obs, var_names=np.array(["g0", "g1", "g2"]))


def test_all_three_estimators_are_public() -> None:
    assert hasattr(ao, "estimate_knockdown_efficiency")
    assert hasattr(ao, "estimate_knockdown_efficiency_detection_rate")
    assert hasattr(ao, "estimate_knockdown_efficiency_poisson_mle")


def test_measure_operator_default_is_auto() -> None:
    sig = inspect.signature(ao.measure_operator)
    assert sig.parameters["efficiency_estimator"].default == "auto"


def test_measure_operator_has_min_control_detection_rate_filter() -> None:
    sig = inspect.signature(ao.measure_operator)
    assert "min_control_detection_rate" in sig.parameters


def test_auto_routes_count_data_to_mean_ratio() -> None:
    """Non-negative count data → mean_ratio."""
    from anchorop.measure import _resolve_estimator
    X_counts = np.abs(np.random.default_rng(0).poisson(2.0, size=(500, 10)).astype(float))
    assert _resolve_estimator("auto", X_counts) == "mean_ratio"


def test_auto_routes_z_scored_data_to_detection_rate() -> None:
    """Pre-scaled residuals (contain negatives) → detection_rate."""
    from anchorop.measure import _resolve_estimator
    X_z = np.random.default_rng(0).normal(0.0, 1.0, size=(500, 10))
    assert _resolve_estimator("auto", X_z) == "detection_rate"


def test_auto_ignores_stray_negatives_from_numerical_noise() -> None:
    """A tiny fraction of negatives from noise should NOT flip the routing."""
    from anchorop.measure import _resolve_estimator
    X = np.abs(np.random.default_rng(0).poisson(2.0, size=(500, 10)).astype(float))
    X[0, 0] = -1e-8  # a single stray negative
    assert _resolve_estimator("auto", X) == "mean_ratio"


def test_explicit_estimator_overrides_auto() -> None:
    """An explicit estimator name is passed through unchanged."""
    from anchorop.measure import _resolve_estimator
    X_z = np.random.default_rng(0).normal(0.0, 1.0, size=(200, 5))
    assert _resolve_estimator("mean_ratio", X_z) == "mean_ratio"
    assert _resolve_estimator("poisson_mle", X_z) == "poisson_mle"


@pytest.mark.acceptance
def test_ACCEPTANCE_info_limited_targets_are_dropped_not_estimated() -> None:
    """Targets with control detection below threshold must be filtered out —
    NOT rescued with a biased estimator that pretends to know κ from ~3
    nonzero cells."""
    adata = _dropout_dominated_target_fixture()
    basis = ao.make_program_basis(
        np.eye(3), adata.var_names, normalize=False, control_count=400
    )
    ctrl_mask = np.zeros(len(adata.obs), dtype=bool); ctrl_mask[:400] = True
    pert_mask = ~ctrl_mask

    # mean_ratio degenerates to ~1.0 on this fixture (as before)
    eff_mean = ao.estimate_knockdown_efficiency(
        adata.X, target_index=2, perturbed_mask=pert_mask, control_mask=ctrl_mask,
    )
    assert eff_mean >= 0.99

    # But the filter drops the guide regardless of estimator choice
    with pytest.raises(ao.AnchorOpError, match="No informative perturbation"):
        ao.measure_operator(
            adata, basis, guide_key="guide", target_key="target_gene",
            control_label="non-targeting",
            min_cells_per_guide=2, min_knockdown_efficiency=0.05,
            reg="tsvd", reg_param="path",
        )

    # Setting min_control_detection_rate=0 disables the filter, and then
    # mean_ratio silently spikes to 1.0 on the info-limited target — the
    # pathology the filter is designed to prevent.
    m_no_filter = ao.measure_operator(
        adata, basis, guide_key="guide", target_key="target_gene",
        control_label="non-targeting",
        min_cells_per_guide=2, min_knockdown_efficiency=0.05,
        min_control_detection_rate=0.0,
        reg="tsvd", reg_param="path",
    )
    assert len(m_no_filter.guide_names) == 1
    assert m_no_filter.report.guide_efficiencies[m_no_filter.guide_names[0]] >= 0.99


def test_min_control_detection_rate_reports_drop_reason() -> None:
    adata = _dropout_dominated_target_fixture()
    basis = ao.make_program_basis(
        np.eye(3), adata.var_names, normalize=False, control_count=400
    )
    with pytest.raises(ao.AnchorOpError):
        ao.measure_operator(
            adata, basis, guide_key="guide", target_key="target_gene",
            control_label="non-targeting",
            min_cells_per_guide=2, reg="tsvd", reg_param="path",
        )


def test_estimators_agree_on_real_knockdown() -> None:
    adata = _real_knockdown_target_fixture()
    ctrl_mask = np.zeros(len(adata.obs), dtype=bool); ctrl_mask[:200] = True
    pert_mask = ~ctrl_mask
    eff_mean = ao.estimate_knockdown_efficiency(
        adata.X, target_index=2, perturbed_mask=pert_mask, control_mask=ctrl_mask,
    )
    eff_pm = ao.estimate_knockdown_efficiency_poisson_mle(
        adata.X, target_index=2, perturbed_mask=pert_mask, control_mask=ctrl_mask,
    )
    # Both should report a substantial knockdown
    assert 0.5 < eff_mean < 0.99
    # poisson_mle is close to mean_ratio in the well-expressed regime
    assert 0.3 < eff_pm < 0.99


def test_detection_rate_is_documented_as_shift_not_kappa() -> None:
    doc = ao.estimate_knockdown_efficiency_detection_rate.__doc__
    assert "NOT an unbiased" in doc or "not an unbiased" in doc


def test_poisson_mle_docstring_documents_dropout_limit() -> None:
    doc = ao.estimate_knockdown_efficiency_poisson_mle.__doc__
    assert "Poisson" in doc


def test_notes_record_estimator_choice() -> None:
    adata = _real_knockdown_target_fixture()
    basis = ao.make_program_basis(
        np.eye(3), adata.var_names, normalize=False, control_count=200
    )
    m = ao.measure_operator(
        adata, basis, guide_key="guide", target_key="target_gene",
        control_label="non-targeting",
        min_cells_per_guide=2, min_knockdown_efficiency=0.01,
        reg="tsvd", reg_param="path",
        efficiency_estimator="mean_ratio",
    )
    assert any("mean_ratio" in n for n in m.report.notes)


def test_invalid_efficiency_estimator_is_rejected() -> None:
    adata = _real_knockdown_target_fixture()
    basis = ao.make_program_basis(
        np.eye(3), adata.var_names, normalize=False, control_count=200
    )
    with pytest.raises(ao.AnchorOpError, match="efficiency_estimator"):
        ao.measure_operator(
            adata, basis, guide_key="guide", target_key="target_gene",
            control_label="non-targeting",
            min_cells_per_guide=2, min_knockdown_efficiency=0.05,
            efficiency_estimator="not_a_real_method",
        )
