"""Regression tests for biology-first Anchor-op measurement safeguards."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import anchorop as ao
from anchorop.measure import GuideResponse


def _paired_assays() -> tuple[SimpleNamespace, SimpleNamespace, object]:
    """Return a signed response assay paired exactly to a count-like assay."""
    controls_response = np.zeros((4, 3), dtype=float)
    g0_response = np.tile(np.array([-0.5, 0.0, 0.0]), (3, 1))
    g1_response = np.tile(np.array([0.0, -0.5, 0.0]), (3, 1))
    response = np.vstack([controls_response, g0_response, g1_response])

    controls_counts = np.tile(np.array([2.0, 2.0, 1.0]), (4, 1))
    g0_counts = np.tile(np.array([1.0, 2.0, 1.0]), (3, 1))
    g1_counts = np.tile(np.array([2.0, 1.0, 1.0]), (3, 1))
    counts = np.vstack([controls_counts, g0_counts, g1_counts])

    obs = pd.DataFrame(
        {
            "guide": ["non-targeting"] * 4 + ["guide_g0"] * 3 + ["guide_g1"] * 3,
            "target_gene": [""] * 4 + ["g0"] * 3 + ["g1"] * 3,
            "batch": ["b1", "b1", "b2", "b2"] + ["b1", "b1", "b2"] + ["b1", "b1", "b2"],
        },
        index=[f"cell_{index}" for index in range(10)],
    )
    genes = np.array(["g0", "g1", "g2"])
    response_adata = SimpleNamespace(X=response, obs=obs.copy(), var_names=genes)
    count_adata = SimpleNamespace(X=counts, obs=obs.copy(), var_names=genes)
    basis = ao.make_program_basis(np.eye(3, 2), genes, normalize=False, control_count=4)
    return response_adata, count_adata, basis


def _measure(response_adata, count_adata, basis):
    return ao.measure_operator(
        response_adata,
        basis,
        guide_key="guide",
        target_key="target_gene",
        control_label="non-targeting",
        batch_key="batch",
        min_cells_per_guide=2,
        min_knockdown_efficiency=0.05,
        calibration_adata=count_adata,
        reg="tsvd",
        reg_param="path",
    )


def test_signed_residual_response_requires_paired_raw_calibration() -> None:
    response, _, basis = _paired_assays()
    with pytest.raises(ao.AnchorOpError, match="calibration_adata"):
        ao.measure_operator(
            response,
            basis,
            guide_key="guide",
            target_key="target_gene",
            control_label="non-targeting",
            batch_key="batch",
            min_cells_per_guide=2,
        )


def test_paired_raw_calibration_accepts_signed_response_and_records_provenance() -> None:
    response, counts, basis = _paired_assays()
    measured = _measure(response, counts, basis)
    assert set(measured.report.guide_targets) == {"guide_g0", "guide_g1"}
    assert measured.report.guide_targets["guide_g0"] == "g0"
    assert any("paired raw/count-like" in note for note in measured.report.notes)


def test_paired_calibration_requires_identical_cell_identifiers_and_order() -> None:
    response, counts, basis = _paired_assays()
    counts.obs.index = list(reversed(counts.obs.index))
    with pytest.raises(ao.AnchorOpError, match="identical cell identifiers"):
        _measure(response, counts, basis)


def test_proxy_responses_can_make_an_atlas_but_not_an_inverse_operator() -> None:
    response, _, basis = _paired_assays()
    guides, _ = ao.build_guide_responses(
        response,
        basis,
        guide_key="guide",
        target_key="target_gene",
        control_label="non-targeting",
        batch_key="batch",
        min_cells_per_guide=2,
        allow_proxy_efficiency=True,
    )
    atlas = ao.build_target_response_atlas(guides)
    assert not atlas.calibrated
    assert atlas.target_names == ("g0", "g1")
    with pytest.raises(ao.AnchorOpError, match="raw/count-calibrated"):
        ao.within_target_dose_response(guides)


def test_target_aggregation_prevents_duplicate_guides_from_receiving_extra_columns() -> None:
    responses = [
        GuideResponse("g0_a", "g0", np.array([1.0, 0.0]), np.array([-0.5, 0.0]), 0.5, 20),
        GuideResponse("g0_b", "g0", np.array([3.0, 0.0]), np.array([-0.8, 0.0]), 0.8, 20),
        GuideResponse("g1", "g1", np.array([0.0, 1.0]), np.array([0.0, -0.6]), 0.6, 20),
    ]
    atlas = ao.build_target_response_atlas(responses)
    np.testing.assert_allclose(atlas.responses[0], [2.0, 0.0])
    assert atlas.n_guides["g0"] == 2


def test_calibrated_dose_response_diagnostic_reports_multi_guide_targets() -> None:
    responses = [
        GuideResponse("g0_a", "g0", np.array([0.2, 0.0]), np.array([-0.2, 0.0]), 0.2, 20),
        GuideResponse("g0_b", "g0", np.array([0.5, 0.0]), np.array([-0.5, 0.0]), 0.5, 20),
        GuideResponse("g0_c", "g0", np.array([0.8, 0.0]), np.array([-0.8, 0.0]), 0.8, 20),
    ]
    result = ao.within_target_dose_response(responses)
    assert result.target_names == ("g0",)
    assert result.r_squared_by_target["g0"] > 0.99
    assert result.response_cosine_by_target["g0"] == pytest.approx(1.0)
