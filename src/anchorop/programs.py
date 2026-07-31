"""Program-space construction and projection.

The package fits expression programs on controls only, preventing perturbation
responses from defining the coordinate system used to assess those responses.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._utils import (
    expression_and_metadata,
    greedy_component_match,
    normalize_columns,
    resolve_mask,
    stable_rng,
    to_dense,
)
from .types import AnchorOpError, DimensionGuardError, ProgramBasis


def _fit_nmf_multiplicative(
    X: np.ndarray,
    d: int,
    *,
    seed: int | None,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Fit a small nonnegative factorization with multiplicative updates.

    This dependency-light implementation is adequate for reproducible tests and
    modest control subsets. Production genome-scale analyses should use a
    dedicated cNMF workflow and import its validated loading matrix.
    """
    if np.any(X < 0):
        raise AnchorOpError("NMF requires a nonnegative expression matrix.")
    n_cells, n_genes = X.shape
    if d > min(n_cells, n_genes):
        raise AnchorOpError(f"d={d} exceeds min(n_control_cells, n_genes)={min(n_cells, n_genes)}.")
    rng = stable_rng(seed)
    scale = max(float(np.mean(X)), 1e-3)
    usages = rng.random((n_cells, d)) * scale + 1e-6
    loadings = rng.random((d, n_genes)) * scale + 1e-6
    eps = np.finfo(float).eps
    previous_loss = np.inf
    for iteration in range(1, max_iter + 1):
        loadings *= (usages.T @ X) / np.maximum(usages.T @ usages @ loadings, eps)
        usages *= (X @ loadings.T) / np.maximum(usages @ (loadings @ loadings.T), eps)
        if iteration % 10 == 0 or iteration == max_iter:
            loss = float(np.linalg.norm(X - usages @ loadings, ord="fro") ** 2)
            if abs(previous_loss - loss) <= tol * max(1.0, previous_loss):
                return usages, loadings, loss, iteration
            previous_loss = loss
    return usages, loadings, float(np.linalg.norm(X - usages @ loadings, ord="fro") ** 2), max_iter


def _normalize_program_loadings(loadings_program_by_gene: np.ndarray) -> np.ndarray:
    """Normalize each program loading vector to unit L2 norm."""
    normalized = normalize_columns(loadings_program_by_gene.T).T
    return normalized


def fit_programs(
    adata: Any,
    *,
    d: int,
    method: str = "cnmf",
    control_mask: Any,
    n_seeds: int = 10,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-5,
) -> ProgramBasis:
    """Fit a nonnegative program basis on matched control cells only.

    Parameters
    ----------
    adata:
        AnnData-compatible input with ``.X``, ``.obs``, and ``.var_names``.
    d:
        Explicit program-space dimension. Values above 100 warn; values above
        200 raise :class:`DimensionGuardError`.
    method:
        ``"nmf"`` fits one seeded NMF. ``"cnmf"`` performs a seed ensemble and
        selects the best reconstruction while reporting matched-program
        concordance. It is a lightweight cNMF approximation, not a replacement
        for the upstream cNMF package's consensus clustering pipeline.
    control_mask:
        Boolean vector selecting non-targeting control cells.
    """
    if not isinstance(d, int) or d < 1:
        raise AnchorOpError("d must be a positive integer.")
    if d > 200:
        raise DimensionGuardError(
            "anchor-op refuses d > 200. Gene-level Jacobians are outside the supported scope."
        )
    if d > 100:
        warnings.warn(
            "d > 100 may make the steady-state inverse poorly conditioned; interpret only with the full report.",
            UserWarning,
            stacklevel=2,
        )
    method = method.lower()
    if method not in {"nmf", "cnmf"}:
        raise AnchorOpError(
            "Supported built-in methods are 'nmf' and 'cnmf'. For SCENIC AUCell or an external cNMF basis, "
            "construct ProgramBasis(loadings=..., gene_names=..., method='external', ...)."
        )
    if n_seeds < 1:
        raise AnchorOpError("n_seeds must be at least one.")
    X, _, gene_names = expression_and_metadata(adata)
    controls = resolve_mask(control_mask, X.shape[0], "control_mask")
    if controls.sum() < 2:
        raise AnchorOpError(
            "At least two matched control cells are required to fit a program basis."
        )
    X_control = X[controls]
    if d > min(X_control.shape):
        raise AnchorOpError(
            f"d={d} exceeds the available control matrix rank bound min{X_control.shape}."
        )

    seed_count = 1 if method == "nmf" else n_seeds
    fits: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    for seed_offset in range(seed_count):
        fits.append(
            _fit_nmf_multiplicative(
                X_control,
                d,
                seed=seed + seed_offset,
                max_iter=max_iter,
                tol=tol,
            )
        )
    best_index = int(np.argmin([fit[2] for fit in fits]))
    _, best_loadings, best_loss, best_iterations = fits[best_index]
    best_loadings = _normalize_program_loadings(best_loadings)

    concordances: list[float] = []
    if seed_count > 1:
        for _, candidate, _, _ in fits:
            candidate_normalized = _normalize_program_loadings(candidate)
            _, concordance = greedy_component_match(best_loadings.T, candidate_normalized.T)
            concordances.append(concordance)
    seed_concordance = float(np.mean(concordances)) if concordances else None
    metadata = {
        "fit_seed": int(seed + best_index),
        "seed_count": seed_count,
        "best_reconstruction_loss": best_loss,
        "iterations": best_iterations,
        "concordances": concordances,
        "implementation": "multiplicative-update NMF",
    }
    if method == "cnmf":
        metadata["cnmf_note"] = (
            "Seed-ensemble stability approximation; validate or import a dedicated cNMF basis for publication."
        )
    return ProgramBasis(
        loadings=best_loadings.T,
        gene_names=gene_names,
        method=method,
        control_count=int(controls.sum()),
        seed_concordance=seed_concordance,
        metadata=metadata,
    )


def basis_row_indices(basis: ProgramBasis, gene_names: Sequence[str]) -> np.ndarray:
    """Return indices that align a source gene order to a basis gene order."""
    source_index = {str(gene): index for index, gene in enumerate(gene_names)}
    missing = [gene for gene in basis.gene_names if gene not in source_index]
    if missing:
        preview = ", ".join(missing[:8])
        raise AnchorOpError(
            f"Expression data are missing {len(missing)} genes required by ProgramBasis; examples: {preview}."
        )
    return np.fromiter((source_index[gene] for gene in basis.gene_names), dtype=int)


def project_expression(
    expression: np.ndarray,
    basis: ProgramBasis,
    *,
    gene_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Project cell-by-gene expression into the basis coordinates ``z = W^T e``.

    Rows of ``expression`` are cells and columns are genes. The returned array is
    cell-by-program. If names are supplied, columns are aligned to the basis.
    """
    expression = to_dense(expression)
    if expression.ndim != 2:
        raise AnchorOpError("expression must be a two-dimensional cell-by-gene matrix.")
    if gene_names is None:
        if expression.shape[1] != basis.n_genes:
            raise AnchorOpError(
                "gene_names are required when expression columns do not exactly match the basis gene count."
            )
        aligned = expression
    else:
        indices = basis_row_indices(basis, gene_names)
        aligned = expression[:, indices]
    return aligned @ basis.loadings


def make_program_basis(
    loadings: np.ndarray,
    gene_names: Sequence[str],
    *,
    method: str = "external",
    control_count: int = 0,
    normalize: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ProgramBasis:
    """Validate an externally fitted gene-by-program loading matrix.

    Use this route for cNMF, SCENIC AUCell, or any validated upstream program
    representation. Program columns are optionally normalized so perturbation
    encodings have a reproducible scale.
    """
    loadings = to_dense(loadings)
    if loadings.ndim != 2:
        raise AnchorOpError("External loadings must be a gene-by-program matrix.")
    d = loadings.shape[1]
    if d > 200:
        raise DimensionGuardError("anchor-op refuses external program spaces with d > 200.")
    if d > 100:
        warnings.warn(
            "External basis has d > 100; proceed only with full conditioning disclosure.",
            UserWarning,
        )
    if normalize:
        loadings = normalize_columns(loadings)
    return ProgramBasis(
        loadings=loadings,
        gene_names=tuple(map(str, gene_names)),
        method=method,
        control_count=int(control_count),
        metadata={} if metadata is None else metadata,
    )
