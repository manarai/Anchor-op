"""High-level analysis workflows: combine measurement, comparison, and archetype
outputs with the plotting helpers and (optionally) save everything to disk.

Each ``*_report`` function returns a dict with:
- ``figures``: mapping of name → matplotlib Figure
- one or more result objects (``summary``, ``results``, ``archetype_result``, ...)
- optional ``table``: a pandas DataFrame of numeric outputs

When ``save_dir`` is supplied, figures are written as PNG, tables as CSV, and
numeric summaries as JSON — a single call produces a fully self-contained
report directory suitable for a paper's supplementary materials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .plotting import (
    plot_archetype_cv_curve,
    plot_archetype_matrices,
    plot_benchmark_bars,
    plot_efficiency_comparison,
    plot_guide_drop_reasons,
    plot_measurement_diagnostics,
    plot_simplex_weights_heatmap,
    plot_sym_antisym_bars,
    save_figures,
)
from .types import ArchetypeResult, MeasuredOperator


# --------------------------------------------------------------------------- #
# Measurement.
# --------------------------------------------------------------------------- #
def measurement_report(
    measurement: MeasuredOperator, save_dir: Any | None = None
) -> dict[str, Any]:
    """Full measurement diagnostic: figures + summary of the AnchorReport.

    Figures produced:
    - ``diagnostics``: three-panel (singular spectrum + operator heatmap + eigenvalues)
    - ``guide_drops``: pareto bar of guide-drop reasons

    ``summary`` is a JSON-serializable dict of the key AnchorReport quantities.
    When ``save_dir`` is provided, both figures are written as PNGs alongside
    ``summary.json``.
    """
    report = measurement.report
    if report is None:
        raise ValueError("measurement is missing its AnchorReport")
    figures = {
        "diagnostics": plot_measurement_diagnostics(measurement),
        "guide_drops": plot_guide_drop_reasons(measurement),
    }
    summary = {
        "d": report.d,
        "effective_response_rank": report.effective_response_rank,
        "input_subspace_dim": report.input_subspace_dim,
        "response_subspace_dim": report.response_subspace_dim,
        "full_domain_identified": report.full_domain_identified,
        "condition_number": float(report.condition_number),
        "rank_tol": report.rank_tol,
        "retained_guides": len(measurement.guide_names),
        "n_guides_input": report.n_guides_input,
        "regularization_method": report.regularization_method,
        "selected_regularization": report.selected_regularization,
        "notes": list(report.notes),
    }
    if save_dir is not None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_figures(figures, d)
        (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"figures": figures, "summary": summary}


# --------------------------------------------------------------------------- #
# Benchmark.
# --------------------------------------------------------------------------- #
def benchmark_report(
    measurement: MeasuredOperator,
    inferred_operators: Mapping[str, Any],
    *,
    nulls: Sequence[str] = ("shuffled_edges", "random_init"),
    n_null: int = 100,
    seed: int | None = 0,
    metrics: Sequence[tuple[str, str]] | None = None,
    save_dir: Any | None = None,
) -> dict[str, Any]:
    """Run ``compare()`` and produce all standard benchmark figures + a metrics table.

    Figures produced:
    - ``benchmark_bars``: per-method bars for the primary endpoints with null overlays
    - ``sym_antisym_bars``: preregistered sym-vs-antisym comparison

    When ``save_dir`` is provided, figures are written as PNGs and the metrics
    table is written as ``metrics.csv``.
    """
    from .compare import compare, comparison_table

    results = compare(
        measurement, inferred_operators, nulls=nulls, n_null=n_null, seed=seed
    )
    figures = {
        "benchmark_bars": plot_benchmark_bars(results, metrics=metrics),
        "sym_antisym_bars": plot_sym_antisym_bars(results),
    }
    table = comparison_table(results)
    if save_dir is not None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_figures(figures, d)
        table.to_csv(d / "metrics.csv", index=False)
    return {"figures": figures, "results": results, "table": table}


# --------------------------------------------------------------------------- #
# Archetypes.
# --------------------------------------------------------------------------- #
def archetype_report(
    measurements: Sequence[MeasuredOperator | np.ndarray],
    *,
    mode: str = "operator",
    k: int | str = "cv",
    save_dir: Any | None = None,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Fit archetypes and produce the standard three archetype figures.

    Figures produced:
    - ``cv_curve``: candidate-k selection curve
    - ``simplex_weights``: per-state weights heatmap
    - ``matrices``: side-by-side heatmaps of the archetype geometries
    """
    from .archetypes import fit_archetypes

    archetype_result: ArchetypeResult = fit_archetypes(
        measurements, mode=mode, k=k, **fit_kwargs
    )
    figures: dict[str, Any] = {}
    if archetype_result.candidate_errors:
        figures["cv_curve"] = plot_archetype_cv_curve(archetype_result)
    figures["simplex_weights"] = plot_simplex_weights_heatmap(archetype_result)
    figures["matrices"] = plot_archetype_matrices(archetype_result)
    if save_dir is not None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_figures(figures, d)
    return {"figures": figures, "archetype_result": archetype_result}


# --------------------------------------------------------------------------- #
# Efficiency estimator comparison.
# --------------------------------------------------------------------------- #
def efficiency_comparison_report(
    expression: np.ndarray,
    var_names: Sequence[str],
    control_mask: np.ndarray,
    guide_to_target: Mapping[str, str],
    guide_to_cells: Mapping[str, np.ndarray],
    *,
    save_dir: Any | None = None,
) -> dict[str, Any]:
    """Compute both `mean_ratio` and `detection_rate` efficiencies for every target,
    produce the side-by-side comparison figure, and return raw arrays.

    Parameters mirror the raw inputs that :func:`build_guide_responses` sees
    internally so this can be run on any dataset without going through the
    full measurement pipeline.

    - ``expression``: cell-by-gene expression matrix (raw or normalized).
    - ``var_names``: gene names, aligned with ``expression`` columns.
    - ``control_mask``: length-``n_cells`` boolean flagging NT cells.
    - ``guide_to_target``: mapping of guide id → target gene symbol.
    - ``guide_to_cells``: mapping of guide id → boolean mask of perturbed cells.
    """
    from .measure import (
        estimate_knockdown_efficiency,
        estimate_knockdown_efficiency_detection_rate,
    )

    gene_index = {g: i for i, g in enumerate(var_names)}
    mean_ratio: list[float] = []
    detection_rate: list[float] = []
    guides_used: list[str] = []
    for guide, target in guide_to_target.items():
        if target not in gene_index:
            continue
        pert_mask = np.asarray(guide_to_cells.get(guide, np.zeros(expression.shape[0], bool)))
        if not pert_mask.any() or not control_mask.any():
            continue
        target_index = gene_index[target]
        try:
            mr = estimate_knockdown_efficiency(
                expression,
                target_index=target_index,
                perturbed_mask=pert_mask,
                control_mask=control_mask,
            )
            dr = estimate_knockdown_efficiency_detection_rate(
                expression,
                target_index=target_index,
                perturbed_mask=pert_mask,
                control_mask=control_mask,
            )
        except Exception:
            continue
        mean_ratio.append(mr)
        detection_rate.append(dr)
        guides_used.append(guide)

    mean_ratio_arr = np.asarray(mean_ratio)
    detection_rate_arr = np.asarray(detection_rate)
    figures = {"efficiency_comparison": plot_efficiency_comparison(mean_ratio_arr, detection_rate_arr)}
    summary = {
        "n_targets_considered": len(guides_used),
        "mean_ratio_pseudoperfect": int((mean_ratio_arr >= 0.99).sum()),
        "detection_rate_pseudoperfect": int((detection_rate_arr >= 0.99).sum()),
        "mean_ratio_below_20pct": int((mean_ratio_arr < 0.20).sum()),
        "detection_rate_below_20pct": int((detection_rate_arr < 0.20).sum()),
    }
    if save_dir is not None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_figures(figures, d)
        (d / "efficiency_summary.json").write_text(json.dumps(summary, indent=2))
        np.savez(
            d / "efficiency_arrays.npz",
            guides=np.asarray(guides_used),
            mean_ratio=mean_ratio_arr,
            detection_rate=detection_rate_arr,
        )
    return {
        "figures": figures,
        "summary": summary,
        "guides": guides_used,
        "mean_ratio": mean_ratio_arr,
        "detection_rate": detection_rate_arr,
    }
