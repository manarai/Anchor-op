"""Stub multiome state-space encoder.

Raises :class:`NotImplementedError` on every call until a multiome
Perturb-seq (RNA + ATAC + protein) dataset is confirmed. Kept here
so the StateSpace catalog is exhaustive and downstream code can
check for its presence by name.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import StateSpace


class MultiomeRep(StateSpace):
    """Placeholder for a multiome state-space encoder.

    Not implemented. Attempting to :meth:`fit`, :meth:`encode`,
    :meth:`jacobian`, or :meth:`decode_direction` raises
    :class:`NotImplementedError`.
    """

    def __init__(self, dim: int = 30, name: Optional[str] = None):
        super().__init__(dim=dim, name=name or f"Multiome-{dim}")

    def fit(self, adata):
        raise NotImplementedError(
            "MultiomeRep is a stub; no multiome Perturb-seq dataset is "
            "confirmed for experiment 1 yet.")

    def encode(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MultiomeRep.encode is a stub.")

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MultiomeRep.jacobian is a stub.")

    def decode_direction(self, u_gene: np.ndarray,
                          X=None) -> np.ndarray:
        raise NotImplementedError("MultiomeRep.decode_direction is a stub.")
