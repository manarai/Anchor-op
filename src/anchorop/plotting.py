"""Plotting helpers for anchor-op analyses.

All functions return matplotlib Figure objects and accept an optional ``ax``
for embedding. ``matplotlib`` is optional: the module imports it lazily and
raises a friendly ``ImportError`` with an install hint if it is missing.
This keeps the core anchor-op dependency footprint minimal (numpy + pandas
only) — plotting is add-on capability, not a required import path.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .types import ArchetypeResult, ComparisonResult, MeasuredOperator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _mpl():
    """Lazy matplotlib import."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "anchorop.plotting requires matplotlib. Install with "
            "`pip install matplotlib` or `pip install 'anchor-op[docs]'`."
        ) from error
    return plt


# --------------------------------------------------------------------------- #
# Measurement diagnostics.
# --------------------------------------------------------------------------- #
def plot_singular_spectrum(
    measurement: MeasuredOperator, ax: "Axes | None" = None
) -> "Figure":
    """Log-scale singular values of `S` with the `rank_tol` cutoff overlaid."""
    plt = _mpl()
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
    else:
        fig = ax.figure
    report = measurement.report
    if report is None:
        raise ValueError("measurement is missing its AnchorReport")
    ranks = np.arange(1, report.singular_values.size + 1)
    ax.semilogy(ranks, report.singular_values, "o-", color="#1f4e79", label="σ(S)")
    if report.rank_tol is not None:
        cutoff = report.rank_tol * float(report.singular_values[0])
        ax.axhline(
            cutoff, ls="--", color="#c65a30",
            label=f"rank_tol · σ_max = {cutoff:.2g}",
        )
    kept = report.retained_singular_directions
    ax.scatter(ranks[kept], report.singular_values[kept], color="#1f4e79", zorder=3)
    ax.scatter(
        ranks[~kept], report.singular_values[~kept],
        facecolors="none", edgecolors="#8f8f8f", label="dropped", zorder=3,
    )
    ax.set_xlabel("singular index of S")
    ax.set_ylabel("singular value (log)")
    ax.set_title(f"spectrum — effective rank {report.effective_response_rank}/{report.d}")
    ax.legend(loc="lower left", fontsize=9)
    if owns_fig:
        fig.tight_layout()
    return fig


def plot_operator_heatmap(
    measurement: MeasuredOperator, ax: "Axes | None" = None
) -> "Figure":
    """Heatmap of `J` (full-rank measurements) or the identified action `J·P_X` (partial)."""
    plt = _mpl()
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    report = measurement.report
    if report is None:
        raise ValueError("measurement is missing its AnchorReport")
    if report.full_domain_identified:
        matrix = measurement.J
        title = f"measured J  (d={report.d}, cond={report.condition_number:.1f})"
    else:
        matrix = measurement.identified_action
        title = f"identified action J·P_X  ({report.effective_response_rank}/{report.d})"
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    im = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    if owns_fig:
        fig.tight_layout()
    return fig


def plot_eigenvalue_plane(
    measurement: MeasuredOperator, ax: "Axes | None" = None
) -> "Figure":
    """Complex-plane scatter of `J` eigenvalues. Blocked at partial identification."""
    plt = _mpl()
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    report = measurement.report
    if report is None:
        raise ValueError("measurement is missing its AnchorReport")
    if not report.full_domain_identified:
        ax.text(
            0.5, 0.5, "spectrum blocked\n(partial identification)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#8f8f8f",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if owns_fig:
            fig.tight_layout()
        return fig
    e = np.linalg.eigvals(measurement.J)
    ax.axvline(0, color="#999", lw=0.8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.scatter(e.real, e.imag, marker="x", color="#c65a30", s=60)
    tag = "HYPERBOLIC" if e.real.max() > 0 else "all damped"
    ax.set_title(f"eigenvalues (max Re λ = {e.real.max():+.3f}, {tag})")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    if owns_fig:
        fig.tight_layout()
    return fig


def plot_guide_drop_reasons(
    measurement: MeasuredOperator, ax: "Axes | None" = None
) -> "Figure":
    """Pareto bar of guide-drop reasons."""
    plt = _mpl()
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.2))
    else:
        fig = ax.figure
    report = measurement.report
    if report is None:
        raise ValueError("measurement is missing its AnchorReport")
    if not report.dropped_guides:
        ax.text(
            0.5, 0.5, "no guides dropped", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#8f8f8f",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if owns_fig:
            fig.tight_layout()
        return fig
    reasons = Counter(report.dropped_guides.values())
    labels = [k for k, _ in reasons.most_common()]
    counts = [reasons[k] for k in labels]
    ax.barh(range(len(labels)), counts, color="#1f4e79")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("guides dropped")
    ax.set_title(
        f"drop reasons  ({len(report.dropped_guides)} of {report.n_guides_input} dropped)"
    )
    if owns_fig:
        fig.tight_layout()
    return fig


def plot_measurement_diagnostics(measurement: MeasuredOperator) -> "Figure":
    """Three-panel diagnostic: singular spectrum + operator heatmap + eigenvalue plane."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    plot_singular_spectrum(measurement, ax=axes[0])
    plot_operator_heatmap(measurement, ax=axes[1])
    plot_eigenvalue_plane(measurement, ax=axes[2])
    return fig


# --------------------------------------------------------------------------- #
# Benchmark / comparison.
# --------------------------------------------------------------------------- #
def plot_benchmark_bars(
    results: Mapping[str, ComparisonResult],
    metrics: Sequence[tuple[str, str]] | None = None,
) -> "Figure":
    """Per-method bars for each metric with null means overlaid.

    ``metrics`` is a list of ``(metric_key, display_label)`` pairs; defaults to
    the four primary endpoints (operator error, equation residual, sym error,
    antisym error).
    """
    plt = _mpl()
    method_names = list(results.keys())
    if metrics is None:
        metrics = [
            ("operator_relative_error", "operator error"),
            ("equation_relative_residual", "equation residual"),
            ("symmetric_relative_error", "sym error"),
            ("antisymmetric_relative_error", "antisym error"),
        ]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0), constrained_layout=True)
    if n == 1:
        axes = [axes]
    x = np.arange(len(method_names))
    for ax, (key, label) in zip(axes, metrics):
        vals = np.array([results[m].metrics.get(key, np.nan) for m in method_names])
        ax.bar(x, vals, color="#1f4e79", label="method")
        for null_name, ls, color in [
            ("shuffled_edges", "--", "#c65a30"),
            ("random_init", ":", "#7a4d95"),
        ]:
            arr = np.concatenate(
                [results[m].null_metrics.get(f"{null_name}:{key}", np.array([]))
                 for m in method_names]
            )
            arr = arr[np.isfinite(arr)]
            if arr.size:
                ax.axhline(np.mean(arr), color=color, ls=ls, lw=1, label=f"{null_name} null")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=25, ha="right", fontsize=9)
        ax.set_title(label, fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    return fig


def plot_sym_antisym_bars(
    results: Mapping[str, ComparisonResult], ax: "Axes | None" = None
) -> "Figure":
    """Paired bars of sym vs antisym relative error per method — the preregistered comparison."""
    plt = _mpl()
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
    else:
        fig = ax.figure
    method_names = list(results.keys())
    sym = np.array([results[m].metrics["symmetric_relative_error"] for m in method_names])
    antisym = np.array([results[m].metrics["antisymmetric_relative_error"] for m in method_names])
    x = np.arange(len(method_names))
    width = 0.35
    ax.bar(x - width / 2, sym, width, color="#1f4e79", label="sym part")
    ax.bar(x + width / 2, antisym, width, color="#c65a30", label="antisym part")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("relative Frobenius error on identified domain")
    ax.set_title("preregistered sym vs antisym comparison")
    ax.legend()
    if owns_fig:
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Archetype plots.
# --------------------------------------------------------------------------- #
def plot_archetype_cv_curve(
    archetype_result: ArchetypeResult, ax: "Axes | None" = None
) -> "Figure":
    """CV error vs `k` with the selected value highlighted."""
    plt = _mpl()
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
    else:
        fig = ax.figure
    ks = sorted(archetype_result.candidate_errors)
    errs = [archetype_result.candidate_errors[k] for k in ks]
    ax.plot(ks, errs, "o-", color="#1f4e79")
    ax.axvline(
        archetype_result.selected_k, color="#c65a30", ls="--",
        label=f"selected k = {archetype_result.selected_k}",
    )
    ax.set_xlabel("candidate k")
    ax.set_ylabel(archetype_result.metadata.get("selection", "error"))
    ax.set_title("archetype k selection")
    ax.set_xticks(ks)
    ax.legend()
    if owns_fig:
        fig.tight_layout()
    return fig


def plot_simplex_weights_heatmap(
    archetype_result: ArchetypeResult, ax: "Axes | None" = None
) -> "Figure":
    """Heatmap of per-state simplex weights over archetypes (rows sum to 1)."""
    plt = _mpl()
    weights = archetype_result.weights
    n_states = weights.shape[0]
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, max(3.0, 0.4 * n_states)))
    else:
        fig = ax.figure
    im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("archetype")
    ax.set_ylabel("state")
    ax.set_xticks(range(archetype_result.selected_k))
    ax.set_yticks(range(n_states))
    ax.set_title("simplex weights (rows sum to 1)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="weight")
    if owns_fig:
        fig.tight_layout()
    return fig


def plot_archetype_matrices(archetype_result: ArchetypeResult) -> "Figure":
    """Side-by-side heatmaps of every archetype matrix."""
    plt = _mpl()
    k = archetype_result.selected_k
    ncols = min(k, 4)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows),
        constrained_layout=True, squeeze=False,
    )
    vmax = float(np.max(np.abs(archetype_result.archetypes)))
    im = None
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx < k:
            im = ax.imshow(
                archetype_result.archetypes[idx], cmap="RdBu_r",
                vmin=-vmax, vmax=vmax, aspect="auto",
            )
            ax.set_title(f"archetype {idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    part = archetype_result.metadata.get("operator_part") or "?"
    d = archetype_result.archetypes.shape[-1]
    fig.suptitle(f"operator archetypes ({part} part, d={d})", fontsize=11, y=1.02)
    return fig


# --------------------------------------------------------------------------- #
# Efficiency-estimator comparison (05-style diagnostic).
# --------------------------------------------------------------------------- #
def plot_efficiency_comparison(
    mean_ratio_values: np.ndarray,
    detection_rate_values: np.ndarray,
    ax: "Axes | None" = None,
) -> "Figure":
    """Two-panel comparison of `mean_ratio` vs `detection_rate` efficiency distributions.

    Both arrays are 1-D efficiencies (in ``[0, 1]``) computed over the same
    target set. The left panel shows the legacy `mean_ratio` distribution with
    its dropout-driven spike; the right panel shows the current
    `detection_rate` default.
    """
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.8), constrained_layout=True)
    edges = np.linspace(0, 1.001, 41)
    total = mean_ratio_values.size
    ax1.hist(mean_ratio_values, bins=edges, color="#c65a30", edgecolor="white")
    ax1.axvspan(0.99, 1.001, color="#c65a30", alpha=0.15, label="pseudo-perfect (dropout)")
    ax1.set_title(
        f"LEGACY mean_ratio — {(mean_ratio_values >= 0.99).sum()}/{total} at ~1.0"
    )
    ax1.set_xlabel("efficiency")
    ax1.set_ylabel("targets")
    ax1.legend()
    ax2.hist(detection_rate_values, bins=edges, color="#1f4e79", edgecolor="white")
    ax2.set_title(
        f"CURRENT DEFAULT detection_rate — {(detection_rate_values >= 0.99).sum()}/{total} at ~1.0"
    )
    ax2.set_xlabel("efficiency")
    ax2.set_ylabel("targets")
    return fig


# --------------------------------------------------------------------------- #
# Bulk save helper.
# --------------------------------------------------------------------------- #
def save_figures(figures: Mapping[str, Any], directory: Any, dpi: int = 140) -> list[str]:
    """Save each Figure in ``figures`` to ``directory/<name>.png``.

    Returns the list of saved paths. Creates the directory if missing.
    """
    from pathlib import Path
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for name, fig in figures.items():
        path = dir_path / f"{name}.png"
        fig.savefig(path, dpi=dpi)
        saved.append(str(path))
    return saved
