"""State-space representations for anchor-op experiment 1.

Four frozen (or control-only-fit) encoders presented through a common
API so a target-grouped operator fit can be swapped between them in a
single experiment loop. Preregistered 2026-09-30 in
``experiments/exp1_statespace/PREREG_exp1_statespace.md``.

Public types
------------
- :class:`StateSpace` — abstract base. Subclasses expose ``name``,
  ``dim``, :meth:`fit` (no-op for frozen encoders), :meth:`encode`,
  :meth:`jacobian`, and :meth:`decode_direction`.
- :class:`PCARep` — PCA on non-targeting-control cells. Linear.
  Jacobian is a constant ``(d, G)`` loading matrix.
- :class:`FARep` — FactorAnalysis on non-targeting-control cells.
  Linear. Jacobian is the Moore–Penrose pseudo-inverse of
  ``components_``.
- :class:`ScGPTRep` — frozen scGPT cell encoder followed by a PCA
  head to the shared ``d_out``. Nonlinear in gene space; Jacobian
  by autograd through the continuous input path, with a
  finite-difference feasibility check at construction time.
- :class:`MultiomeRep` — stub, raises
  :class:`NotImplementedError` until a multiome perturbation dataset
  is confirmed.

Design rules
------------
- All ``encode(X)`` return ``(n_cells, dim)`` arrays.
- All ``jacobian(X)`` return ``(n_cells, dim, n_genes)`` arrays
  ``J[i] = ∂encode(x_i) / ∂x_i``. For linear encoders every slice
  is the same constant matrix; the per-cell axis is kept so the
  caller does not need to special-case linear vs nonlinear.
- All ``decode_direction(u_gene)`` return a ``(dim,)`` vector that
  equals ``J @ u_gene`` for the first control cell under linearity.
  Tests enforce this.
- The default shared ``dim`` is 30, matching anchor-op paper 1.
"""
from __future__ import annotations

from .base import StateSpace
from .linear import PCARep, FARep
from .scgpt import ScGPTRep
from .multiome import MultiomeRep

__all__ = [
    "StateSpace",
    "PCARep",
    "FARep",
    "ScGPTRep",
    "MultiomeRep",
]
