"""Internal numerical and AnnData-compatible helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import AnchorOpError


def to_dense(value: Any) -> np.ndarray:
    """Convert an array or scipy-like sparse object to a finite dense float array."""
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise AnchorOpError("Input contains NaN or infinite values.")
    return array


def expression_and_metadata(adata: Any) -> tuple[np.ndarray, pd.DataFrame, tuple[str, ...]]:
    """Extract ``X``, ``obs``, and gene names from an AnnData-like object.

    The package intentionally uses duck typing so that unit tests and lightweight
    workflows need not install the optional ``anndata`` dependency.
    """
    if not hasattr(adata, "X") or not hasattr(adata, "obs"):
        raise AnchorOpError("adata must expose .X and .obs, as an AnnData-compatible object does.")
    if not hasattr(adata, "var_names"):
        raise AnchorOpError("adata must expose .var_names aligned to expression columns.")
    X = to_dense(adata.X)
    if X.ndim != 2:
        raise AnchorOpError("adata.X must be a two-dimensional cell-by-gene matrix.")
    obs = adata.obs.copy() if isinstance(adata.obs, pd.DataFrame) else pd.DataFrame(adata.obs)
    gene_names = tuple(map(str, adata.var_names))
    if X.shape[0] != len(obs):
        raise AnchorOpError("adata.X rows must align with adata.obs.")
    if X.shape[1] != len(gene_names):
        raise AnchorOpError("adata.X columns must align with adata.var_names.")
    if len(set(gene_names)) != len(gene_names):
        raise AnchorOpError("adata.var_names must be unique for target-gene encoding.")
    return X, obs.reset_index(drop=True), gene_names


def resolve_mask(mask: Any, n: int, name: str = "mask") -> np.ndarray:
    """Validate and materialize a length-``n`` Boolean mask."""
    if isinstance(mask, (pd.Series, pd.Index)):
        mask = mask.to_numpy()
    array = np.asarray(mask)
    if array.ndim != 1 or len(array) != n:
        raise AnchorOpError(f"{name} must be a one-dimensional Boolean vector of length {n}.")
    if array.dtype != bool:
        if np.issubdtype(array.dtype, np.number) and np.all(np.isin(array, [0, 1])):
            array = array.astype(bool)
        else:
            raise AnchorOpError(f"{name} must be Boolean (or a numeric 0/1 vector).")
    return array.astype(bool, copy=False)


def require_column(obs: pd.DataFrame, key: str) -> pd.Series:
    """Return a metadata column or raise an actionable error."""
    if key not in obs.columns:
        available = ", ".join(map(str, obs.columns[:10]))
        raise AnchorOpError(
            f"obs[{key!r}] is required but absent. Available columns begin: {available}"
        )
    return obs[key]


def orthogonal_projector(
    matrix: np.ndarray, rtol: float | None = None
) -> tuple[np.ndarray, int, np.ndarray]:
    """Return projector, numerical rank, and singular values for a matrix column space."""
    matrix = to_dense(matrix)
    if matrix.ndim != 2:
        raise AnchorOpError("A two-dimensional matrix is required to build a projector.")
    rows, columns = matrix.shape
    if rows == 0:
        raise AnchorOpError("Cannot form a projector in a zero-dimensional space.")
    if columns == 0:
        return np.zeros((rows, rows), dtype=float), 0, np.array([], dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((rows, rows), dtype=float), 0, singular_values
    threshold = (
        rtol if rtol is not None else np.finfo(float).eps * max(matrix.shape)
    ) * singular_values[0]
    rank = int(np.sum(singular_values > threshold))
    Q = left[:, :rank]
    return Q @ Q.T, rank, singular_values


def normalize_columns(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize columns without changing zero columns."""
    matrix = to_dense(matrix)
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / np.maximum(norms, eps)


def cosine_similarity_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarities between columns of two matrices."""
    left_n = normalize_columns(left)
    right_n = normalize_columns(right)
    return left_n.T @ right_n


def greedy_component_match(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, float]:
    """Greedily align candidate columns to reference columns by cosine similarity.

    This avoids a mandatory SciPy dependency. It is documented as a lightweight
    cNMF stability diagnostic rather than a substitute for a dedicated consensus
    clustering implementation at large scale.
    """
    similarity = cosine_similarity_columns(reference, candidate)
    if similarity.shape[0] != similarity.shape[1]:
        raise AnchorOpError("Only equal-rank component matrices can be matched.")
    d = similarity.shape[0]
    remaining_rows = set(range(d))
    remaining_cols = set(range(d))
    assignment = np.full(d, -1, dtype=int)
    scores: list[float] = []
    while remaining_rows:
        row, col = max(
            ((r, c) for r in remaining_rows for c in remaining_cols),
            key=lambda pair: similarity[pair[0], pair[1]],
        )
        assignment[row] = col
        scores.append(float(similarity[row, col]))
        remaining_rows.remove(row)
        remaining_cols.remove(col)
    return assignment, float(np.mean(scores)) if scores else float("nan")


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex using the sorting algorithm."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size == 0:
        raise AnchorOpError("Cannot project an empty vector to a simplex.")
    sorted_v = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_v)
    indices = np.arange(1, vector.size + 1)
    valid = sorted_v - (cumulative - 1.0) / indices > 0
    if not np.any(valid):
        return np.full(vector.size, 1.0 / vector.size)
    rho = np.flatnonzero(valid)[-1]
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(vector - theta, 0.0)


def frobenius_relative_error(
    estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-12
) -> float:
    """Return a stable Frobenius relative error."""
    estimate = to_dense(estimate)
    reference = to_dense(reference)
    if estimate.shape != reference.shape:
        raise AnchorOpError("Matrices must have the same shape for relative error.")
    return float(
        np.linalg.norm(estimate - reference, ord="fro")
        / max(np.linalg.norm(reference, ord="fro"), eps)
    )


def stable_rng(seed: int | None) -> np.random.Generator:
    """Return a reproducibly seeded NumPy generator."""
    return np.random.default_rng(seed)
