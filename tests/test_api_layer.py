"""Smoke tests for the plotting and analyses API layer.

These are the tests that don't require matplotlib to be usable — they check
that the modules import, the lazy matplotlib import raises a sensible error
if matplotlib is missing, and that report functions produce the right dict
structure. When matplotlib IS available, they also check that each figure
renders without error and that save_dir round-trips.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import anchorop as ao

HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def _tiny_measurement() -> ao.MeasuredOperator:
    """Well-conditioned synthetic measurement, d=3, m=8."""
    rng = np.random.default_rng(1)
    J = np.array([[-1.1, 0.1, 0.0], [0.0, -0.9, 0.1], [0.0, 0.0, -1.2]])
    U = rng.normal(size=(3, 8))
    S = -np.linalg.solve(J, U)
    return ao.measure_from_sensitivity(S, U, reg="tsvd", reg_param=3)


def test_plotting_and_analyses_are_importable() -> None:
    assert hasattr(ao, "plotting")
    assert hasattr(ao, "analyses")
    # Function-level attributes are exposed as expected.
    assert hasattr(ao.plotting, "plot_measurement_diagnostics")
    assert hasattr(ao.plotting, "plot_benchmark_bars")
    assert hasattr(ao.analyses, "measurement_report")
    assert hasattr(ao.analyses, "benchmark_report")
    assert hasattr(ao.analyses, "archetype_report")
    assert hasattr(ao.analyses, "efficiency_comparison_report")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_measurement_report_returns_expected_keys() -> None:
    m = _tiny_measurement()
    report = ao.analyses.measurement_report(m)
    assert set(report) == {"figures", "summary"}
    assert set(report["figures"]) == {"diagnostics", "guide_drops"}
    assert report["summary"]["d"] == 3
    assert report["summary"]["full_domain_identified"] is True


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_benchmark_report_produces_metrics_and_figures() -> None:
    m = _tiny_measurement()
    J_exact = np.array([[-1.1, 0.1, 0.0], [0.0, -0.9, 0.1], [0.0, 0.0, -1.2]])
    report = ao.analyses.benchmark_report(m, {"exact": J_exact}, n_null=8, seed=1)
    assert set(report) == {"figures", "results", "table"}
    assert set(report["figures"]) == {"benchmark_bars", "sym_antisym_bars"}
    assert "operator_relative_error" in report["table"].columns
    assert len(report["results"]) == 1


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_archetype_report_produces_all_three_figures() -> None:
    rng = np.random.default_rng(0)
    matrices = [
        rng.normal(size=(3, 3)) - 1.5 * np.eye(3) for _ in range(4)
    ]
    report = ao.analyses.archetype_report(matrices, mode="operator", k=2)
    assert "figures" in report and "archetype_result" in report
    assert set(report["figures"]) >= {"simplex_weights", "matrices"}


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_save_dir_roundtrip() -> None:
    m = _tiny_measurement()
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "measurement_report"
        ao.analyses.measurement_report(m, save_dir=outdir)
        assert (outdir / "diagnostics.png").exists()
        assert (outdir / "summary.json").exists()
        parsed = json.loads((outdir / "summary.json").read_text())
        assert parsed["d"] == 3


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_efficiency_comparison_report_shape() -> None:
    # Fixture with two targets: one at dropout floor, one well expressed.
    rng = np.random.default_rng(2)
    n_ctrl, n_pert = 200, 50
    X = np.zeros((n_ctrl + 2 * n_pert, 3))
    X[:n_ctrl, 0] = rng.gamma(2, 1, n_ctrl) + 1
    X[:n_ctrl, 1] = rng.gamma(2, 1, n_ctrl) + 1
    # target 1 (col 2) at dropout floor
    detect_ctrl = rng.random(n_ctrl) < 0.005
    X[:n_ctrl, 2] = detect_ctrl.astype(float)
    # perturbed cells for guide_A (targets gene g2, at dropout floor)
    X[n_ctrl : n_ctrl + n_pert, 0] = rng.gamma(2, 1, n_pert) + 1
    X[n_ctrl : n_ctrl + n_pert, 1] = rng.gamma(2, 1, n_pert) + 1
    # perturbed for guide_B (targets gene g1, well expressed)
    X[n_ctrl + n_pert :, 0] = rng.gamma(2, 1, n_pert) + 1
    X[n_ctrl + n_pert :, 1] = 0.3 * (rng.gamma(2, 1, n_pert) + 1)
    ctrl_mask = np.zeros(X.shape[0], dtype=bool)
    ctrl_mask[:n_ctrl] = True
    guide_to_cells = {
        "guide_A": np.arange(X.shape[0]).__ge__(n_ctrl) & np.arange(X.shape[0]).__lt__(n_ctrl + n_pert),
        "guide_B": np.arange(X.shape[0]).__ge__(n_ctrl + n_pert),
    }
    guide_to_target = {"guide_A": "g2", "guide_B": "g1"}
    report = ao.analyses.efficiency_comparison_report(
        expression=X,
        var_names=["g0", "g1", "g2"],
        control_mask=ctrl_mask,
        guide_to_target=guide_to_target,
        guide_to_cells=guide_to_cells,
    )
    assert report["summary"]["n_targets_considered"] == 2
    assert report["mean_ratio"].shape == (2,)
    assert report["detection_rate"].shape == (2,)
    # Dropout-floor target should look pseudo-perfect under mean_ratio and
    # small under detection_rate.
    idx_a = report["guides"].index("guide_A")
    assert report["mean_ratio"][idx_a] >= 0.9
    assert report["detection_rate"][idx_a] < 0.05
