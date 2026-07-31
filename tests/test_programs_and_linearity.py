"""Tests for control-only program handling and linearity diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import anchorop as ao


def test_control_only_program_fit_and_projection() -> None:
    rng = np.random.default_rng(42)
    controls = rng.gamma(shape=2.0, scale=1.0, size=(20, 5))
    perturbations = rng.gamma(shape=2.0, scale=1.0, size=(8, 5))
    perturbations[:, 0] += 20.0
    adata = SimpleNamespace(
        X=np.vstack([controls, perturbations]),
        obs=pd.DataFrame({"is_control": [True] * 20 + [False] * 8}),
        var_names=np.array([f"g{i}" for i in range(5)]),
    )
    basis = ao.fit_programs(
        adata,
        d=2,
        method="cnmf",
        control_mask=adata.obs["is_control"].to_numpy(),
        n_seeds=3,
        max_iter=60,
        seed=7,
    )
    assert basis.control_count == 20
    assert basis.seed_concordance is not None
    assert basis.loadings.shape == (5, 2)
    coordinates = ao.project_expression(adata.X, basis, gene_names=adata.var_names)
    assert coordinates.shape == (28, 2)


def test_linearity_check_matches_same_known_action_across_efficiency_bins() -> None:
    rng = np.random.default_rng(8)
    J = np.array([[-1.0, 0.1], [0.0, -0.9]])
    U = rng.normal(size=(2, 10))
    S = -np.linalg.solve(J, U)
    names = [f"guide_{i}" for i in range(10)]
    efficiencies = {name: 0.2 + 0.07 * index for index, name in enumerate(names)}
    measurement = ao.measure_from_sensitivity(
        S,
        U,
        guide_names=names,
        guide_efficiencies=efficiencies,
        reg="tsvd",
        reg_param=2,
    )
    result = ao.linearity_check(measurement, threshold=1e-8)
    assert result.passed
    assert result.overlap_rank == 2
    assert result.relative_difference < 1e-10
    assert len(result.weak_guides) == 5
    assert len(result.strong_guides) == 5
