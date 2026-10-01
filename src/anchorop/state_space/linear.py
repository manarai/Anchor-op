"""Linear state-space encoders: PCA and FactorAnalysis.

Both are fit on non-targeting control cells only. Both have a
closed-form Jacobian that is constant across cells.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import StateSpace


class PCARep(StateSpace):
    """PCA on control cells, truncated to ``dim`` components.

    Encoder: ``z = (x − μ) · V``, where ``V`` is ``(G, dim)`` with
    orthonormal columns (the top ``dim`` right singular vectors of
    the centred control matrix). Jacobian: ``∂z/∂x = Vᵀ``, constant
    across cells.
    """

    def __init__(self, dim: int = 30, name: Optional[str] = None):
        super().__init__(dim=dim, name=name or f"PCA-{dim}")
        self.mean_: Optional[np.ndarray] = None   # (G,)
        self.components_: Optional[np.ndarray] = None  # (dim, G) rows
        self.n_genes_: Optional[int] = None

    def fit(self, adata) -> "PCARep":
        from sklearn.decomposition import PCA

        X = _as_dense(adata)
        if X.shape[1] < self.dim:
            raise ValueError(
                f"PCARep dim={self.dim} > n_genes={X.shape[1]}")
        pca = PCA(n_components=self.dim, svd_solver="auto", random_state=0)
        pca.fit(X)
        self.mean_ = pca.mean_.astype(np.float64)
        self.components_ = pca.components_.astype(np.float64)  # (dim, G)
        self.n_genes_ = X.shape[1]
        return self

    def _check_fit(self):
        if self.components_ is None:
            raise RuntimeError(f"{self.name} must be fit() before use")

    def encode(self, X: np.ndarray) -> np.ndarray:
        self._check_fit()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        self._check_fit()
        X = np.asarray(X, dtype=np.float64)
        J_const = self.components_   # (dim, G)
        return np.broadcast_to(J_const, (X.shape[0], self.dim, self.n_genes_)).copy()

    def decode_direction(self, u_gene: np.ndarray,
                          X: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_fit()
        u_gene = np.asarray(u_gene, dtype=np.float64).ravel()
        if u_gene.shape[0] != self.n_genes_:
            raise ValueError(
                f"u_gene shape {u_gene.shape} does not match "
                f"n_genes={self.n_genes_}")
        return self.components_ @ u_gene


class FARep(StateSpace):
    """FactorAnalysis on control cells, ``dim`` latent factors.

    Encoder: ``z = (x − μ) · W⁺``, where ``W = components_`` is
    ``(dim, G)`` and ``W⁺`` is its Moore–Penrose pseudo-inverse (so
    ``z`` is the least-squares projection of the centred expression
    onto the fitted factor loadings). Jacobian: ``W⁺ᵀ``, constant
    across cells.
    """

    def __init__(self, dim: int = 30, name: Optional[str] = None):
        super().__init__(dim=dim, name=name or f"FA-{dim}")
        self.mean_: Optional[np.ndarray] = None    # (G,)
        self.components_: Optional[np.ndarray] = None  # (dim, G)
        self.decoder_pinv_: Optional[np.ndarray] = None  # (G, dim)
        self.n_genes_: Optional[int] = None

    def fit(self, adata) -> "FARep":
        from sklearn.decomposition import FactorAnalysis

        X = _as_dense(adata)
        if X.shape[1] < self.dim:
            raise ValueError(
                f"FARep dim={self.dim} > n_genes={X.shape[1]}")
        fa = FactorAnalysis(n_components=self.dim, random_state=0)
        fa.fit(X)
        self.mean_ = fa.mean_.astype(np.float64)
        self.components_ = fa.components_.astype(np.float64)   # (dim, G)
        # Encoder = (x - μ) · W⁺, where W = components_ (dim × G).
        # pinv(W) has shape (G, dim). W⁺ · (x−μ)ᵀ = (dim,) using
        # (G, dim)ᵀ · (G,) convention; precompute W⁺ᵀ with shape
        # (dim, G) so encode is a single matmul and jacobian is just
        # that constant.
        self.decoder_pinv_ = np.linalg.pinv(self.components_)  # (G, dim)
        self.n_genes_ = X.shape[1]
        return self

    def _check_fit(self):
        if self.components_ is None:
            raise RuntimeError(f"{self.name} must be fit() before use")

    def encode(self, X: np.ndarray) -> np.ndarray:
        self._check_fit()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.decoder_pinv_  # (n, dim)

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        self._check_fit()
        X = np.asarray(X, dtype=np.float64)
        J_const = self.decoder_pinv_.T   # (dim, G)
        return np.broadcast_to(J_const, (X.shape[0], self.dim, self.n_genes_)).copy()

    def decode_direction(self, u_gene: np.ndarray,
                          X: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_fit()
        u_gene = np.asarray(u_gene, dtype=np.float64).ravel()
        if u_gene.shape[0] != self.n_genes_:
            raise ValueError(
                f"u_gene shape {u_gene.shape} does not match "
                f"n_genes={self.n_genes_}")
        return self.decoder_pinv_.T @ u_gene


# ─── small helper ─────────────────────────────────────────────────────────
def _as_dense(adata_or_array) -> np.ndarray:
    """Accept an AnnData or an array-like and return a dense ``(n, G)``
    float64 ndarray. Sparse inputs are densified."""
    import scipy.sparse as sp

    if hasattr(adata_or_array, "X"):
        X = adata_or_array.X
    else:
        X = adata_or_array
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)
