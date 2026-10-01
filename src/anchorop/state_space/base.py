"""Abstract :class:`StateSpace` base class.

A :class:`StateSpace` is an encoder from gene-space expression into a
``dim``-dimensional representation shared across all arms of the
experiment. The API is designed so that an operator-fit loop can hold
a ``StateSpace`` instance and interrogate it uniformly, regardless of
whether the encoder is linear (PCA, FA) or nonlinear (scGPT).

Shape conventions
-----------------
Let ``G`` be the number of genes in the shared feature set, ``n`` the
number of cells.

- ``encode(X)``: ``(n, G) → (n, dim)``.
- ``jacobian(X)``: ``(n, G) → (n, dim, G)``. Per-cell Jacobian of
  ``encode``. For linear encoders every slice ``J[i]`` is the same
  constant matrix.
- ``decode_direction(u_gene)``: ``(G,) → (dim,)``. Push a gene-space
  perturbation direction through the encoder at a representative
  state (control-cell mean by default). For linear encoders this
  equals ``J[0] @ u_gene`` exactly.

The ``fit(adata)`` call is a no-op for frozen encoders (scGPT).
Encoders that fit on data (PCA, FA) store their parameters on the
instance; subsequent calls to ``encode`` / ``jacobian`` /
``decode_direction`` must use the fitted parameters.
"""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np


class StateSpace(abc.ABC):
    """Abstract state-space encoder.

    Attributes
    ----------
    name : str
        Short human-readable name of the encoder (e.g. ``"PCA-30"``).
    dim : int
        Output dimensionality shared by all arms.
    """

    name: str
    dim: int

    def __init__(self, dim: int, name: Optional[str] = None):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.name = name if name is not None else f"{type(self).__name__}-{dim}"

    @abc.abstractmethod
    def fit(self, adata) -> "StateSpace":
        """Fit the encoder on an AnnData of control-cell expression.

        Implementations may be a no-op for frozen encoders. Must
        return ``self`` to allow chaining ``rep = SomeRep(d).fit(ctrl)``.
        """

    @abc.abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode ``(n_cells, n_genes) → (n_cells, dim)``."""

    @abc.abstractmethod
    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return the per-cell Jacobian ``∂encode(x_i)/∂x_i``.

        Shape ``(n_cells, dim, n_genes)``. Linear encoders broadcast
        the constant matrix along the per-cell axis so callers do not
        need to branch on linearity.
        """

    @abc.abstractmethod
    def decode_direction(self, u_gene: np.ndarray,
                          X: Optional[np.ndarray] = None) -> np.ndarray:
        """Push a gene-space direction into the state space.

        ``u_gene`` has shape ``(n_genes,)``. If ``X`` is given, use the
        column mean of ``X`` as the evaluation state; otherwise use a
        representative control-cell state stored by :meth:`fit` (if
        available) or raise.

        For linear encoders the result equals ``J @ u_gene`` exactly
        at every evaluation state. For nonlinear encoders it is
        ``J(x_ref) @ u_gene`` at the given ``x_ref``.
        """

    def knockdown_scale_difference(self, X_ctrl: np.ndarray,
                                    target_gene_idx: int,
                                    kappa: float = 0.7,
                                    *,
                                    input_space: str = "log1p",
                                    ) -> np.ndarray:
        """Mean knockdown-scale finite difference on control cells.

        Defined per the amendment to A4 (exp1 PREREG amendment 2,
        2026-09-30, logged in ``EXPERIMENT_LOG.md``):

            u_z = mean_i [ E(x_i^{perturbed}) − E(x_i) ]

        where the target gene's *normalised* expression is scaled by
        ``(1 − κ)``. The scaling happens in the appropriate space:

        - ``input_space='log1p'`` (default) — inputs are log1p-normalised
          counts. For each control cell, un-log the target gene with
          ``expm1``, multiply by ``(1 − κ)``, and re-log with ``log1p``
          before passing through :meth:`encode`. This is the
          preregistered definition for scGPT→30 and for the
          PCA / FA arms fit on log1p-normalised data.
        - ``input_space='linear'`` — inputs are already in a linear
          (not log) representation; the scale ``(1 − κ)`` is applied
          directly to the target gene's value. Used only for
          synthetic tests and legacy residual h5ads.

        For zero-centred residual inputs under ``input_space='linear'``,
        the per-gene control-cell mean is 0 by construction and the
        knockdown-scale FD collapses to 0 for every linear encoder —
        the amendment-triggered reason to require log1p-normalised
        inputs on real data.

        The default implementation uses two :meth:`encode` calls and
        is numerically correct for any encoder that supports
        :meth:`encode` on a modified input matrix.
        """
        X_ctrl = np.asarray(X_ctrl, dtype=np.float64)
        X_perturbed = X_ctrl.copy()
        if input_space == "log1p":
            y = np.expm1(X_perturbed[:, target_gene_idx])
            y = np.clip(y, 0.0, None)
            X_perturbed[:, target_gene_idx] = np.log1p(y * (1.0 - kappa))
        elif input_space == "linear":
            X_perturbed[:, target_gene_idx] = (
                X_perturbed[:, target_gene_idx] * (1.0 - kappa))
        else:
            raise ValueError(
                f"input_space must be 'log1p' or 'linear', got {input_space!r}")
        Z_ctrl = self.encode(X_ctrl)
        Z_perturbed = self.encode(X_perturbed)
        return (Z_perturbed - Z_ctrl).mean(axis=0)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"
