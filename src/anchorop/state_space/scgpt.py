"""Frozen scGPT cell encoder with a PCA head to ``dim``.

Design
------
scGPT (Cui et al. 2024, *Nature Methods*) ingests expression as
log-normalised bin indices. Autograd through the binning operation
is not a valid Jacobian because binning is piecewise-constant and
has zero derivative almost everywhere. The amendment A4 of the
experiment 1 preregistration requires that this be the FIRST check:

- (a) If the checkpoint supports a continuous-value input path
      (i.e. accepts raw log-normalised expression as float embeddings
      rather than discrete bin indices), use that path and compute
      the Jacobian by autograd through the continuous embedding.
- (b) Otherwise, estimate the Jacobian by finite differences at a
      step larger than one bin, swept across several step sizes to
      check stability.

The autograd Jacobian (if valid) is compared to the finite-difference
Jacobian on 5 control cells. If the two agree to within a documented
tolerance across multiple step sizes, the scGPT arm proceeds. If
they do not, the arm is halted and the failure is reported — no
workaround (per A4).

This file provides the scaffolding for the feasibility check. The
scGPT checkpoint and the torch runtime are loaded lazily; the module
imports cleanly without ``torch`` or ``scgpt`` installed so the rest
of the StateSpace catalog can be used on CPU-only environments.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import StateSpace

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class ScGPTCheckpoint:
    """Record of a loaded scGPT checkpoint.

    Attributes
    ----------
    name : str
        Short human-readable name (e.g. ``"scGPT_human"``).
    path : Path
        Local path to the checkpoint file.
    sha256 : str
        SHA-256 hash of the checkpoint file. Recorded for provenance.
    native_dim : int
        Native embedding dimensionality of the scGPT encoder (model
        hidden size, before the PCA head).
    continuous_input_supported : bool
        Whether the checkpoint accepts a continuous expression input
        path (A4 feasibility — see module docstring).
    """

    name: str
    path: Path
    sha256: str
    native_dim: int
    continuous_input_supported: bool


class ScGPTRep(StateSpace):
    """Frozen scGPT cell encoder followed by PCA to ``dim``.

    Parameters
    ----------
    d_out : int
        Output dimension of the PCA head. All arms use the shared
        experiment-1 default of 30.
    checkpoint_path : str or Path or None
        Path to the pretrained scGPT checkpoint. If ``None``, the
        constructor looks for ``$SCGPT_CKPT`` in the environment.
    name : str, optional
        Human-readable name.

    Notes
    -----
    ``fit`` is a no-op for the scGPT body (the pretrained weights
    are frozen) but DOES fit the PCA head on the control-cell
    embeddings projected by scGPT.

    See the A4 feasibility check in :meth:`check_jacobian_feasibility`.
    """

    def __init__(self, d_out: int = 30,
                 checkpoint_path: Optional[os.PathLike] = None,
                 name: Optional[str] = None):
        super().__init__(dim=d_out, name=name or f"scGPT→{d_out}")
        if checkpoint_path is None:
            checkpoint_path = os.environ.get("SCGPT_CKPT")
        self._checkpoint_path: Optional[Path] = (
            Path(checkpoint_path) if checkpoint_path else None)
        self._ckpt: Optional[ScGPTCheckpoint] = None
        self._encoder = None  # lazy — the scGPT model instance
        self._pca_head = None  # sklearn PCA fit on control embeddings
        self.mean_: Optional[np.ndarray] = None   # (G,) — scGPT input mean
        self.n_genes_: Optional[int] = None

    # ─── public API ───────────────────────────────────────────────────
    def fit(self, adata) -> "ScGPTRep":
        """Load scGPT (if not already loaded), embed control cells,
        fit the PCA head to ``dim`` components, and record the PCA
        head's explained variance.
        """
        self._ensure_loaded()
        X = _as_dense(adata)
        self.n_genes_ = X.shape[1]
        self.mean_ = X.mean(axis=0)
        # Embed control cells with scGPT (frozen).
        Z_native = self._embed_with_scgpt(X)
        # Fit the PCA head.
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.dim, svd_solver="auto", random_state=0)
        pca.fit(Z_native)
        self._pca_head = pca
        _log.info(
            "ScGPTRep(%s): PCA head fit on %d control cells; "
            "native_dim=%d, retained_variance=%.4f",
            self.name, X.shape[0], Z_native.shape[1],
            float(np.sum(pca.explained_variance_ratio_)))
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        self._check_fit()
        X = np.asarray(X, dtype=np.float64)
        Z_native = self._embed_with_scgpt(X)
        return self._pca_head.transform(Z_native)

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        self._check_fit()
        if self._ckpt is None or not self._ckpt.continuous_input_supported:
            raise RuntimeError(
                "ScGPTRep.jacobian requires a continuous-input scGPT "
                "path (A4 feasibility). Call check_jacobian_feasibility() "
                "first and inspect the result.")
        import torch

        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        J = np.zeros((n, self.dim, self.n_genes_), dtype=np.float64)
        for i in range(n):
            x_t = torch.tensor(X[i], dtype=torch.float64, requires_grad=True)
            z_t = self._encode_continuous(x_t.unsqueeze(0)).squeeze(0)
            # jacobian via vmap over output dims; fall back to loop
            # if torch doesn't have functional jacobian in this version.
            jac = torch.autograd.functional.jacobian(
                lambda u: self._pca_head_forward(
                    self._encode_continuous(u.unsqueeze(0)).squeeze(0)),
                x_t,
                create_graph=False, strict=True)
            J[i] = jac.detach().cpu().numpy()
        return J

    def decode_direction(self, u_gene: np.ndarray,
                          X: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_fit()
        if X is None:
            x_ref = self.mean_  # control-cell mean
        else:
            x_ref = _as_dense(X).mean(axis=0)
        J = self.jacobian(x_ref[None, :])[0]  # (dim, G)
        return J @ np.asarray(u_gene, dtype=np.float64).ravel()

    # ─── A4 feasibility check (standalone, used in Task 1) ────────────
    def check_jacobian_feasibility(self, X_control: np.ndarray,
                                    n_cells: int = 5,
                                    step_sizes: Tuple[float, ...] = (
                                        1e-3, 1e-2, 1e-1),
                                    tol_cosine: float = 0.9,
                                    ) -> Dict[str, Any]:
        """Test that the autograd Jacobian agrees with finite differences
        across step sizes larger than one scGPT expression bin.

        Returns a report dict with keys:

        - ``continuous_input_supported`` : bool
        - ``per_cell_cosine`` : list of per-cell mean cosines between
          autograd and FD Jacobians, for each step size.
        - ``stable`` : bool, True iff all per-cell cosines across all
          step sizes exceed ``tol_cosine``.
        - ``halt_reason`` : str or None. Non-None means the scGPT arm
          should stop per the A4 amendment.
        - ``checkpoint`` : ScGPTCheckpoint record.

        No workaround is attempted on failure; the caller must
        inspect the report and halt the arm if ``halt_reason`` is set.
        """
        self._ensure_loaded()
        if self._ckpt is None:
            return {
                "continuous_input_supported": False,
                "per_cell_cosine": [],
                "stable": False,
                "halt_reason": "scGPT checkpoint failed to load",
                "checkpoint": None,
            }
        if not self._ckpt.continuous_input_supported:
            return {
                "continuous_input_supported": False,
                "per_cell_cosine": [],
                "stable": False,
                "halt_reason": (
                    "scGPT checkpoint does not expose a continuous-value "
                    "input path; autograd through the binning operation "
                    "is not a valid Jacobian (A4). Arm halted."),
                "checkpoint": self._ckpt,
            }
        X_sub = _as_dense(X_control)[:n_cells]
        import torch
        per_cell_cosine = {}
        for h in step_sizes:
            cosines = []
            for i in range(X_sub.shape[0]):
                x = X_sub[i]
                # autograd Jacobian
                x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                jac_ag = torch.autograd.functional.jacobian(
                    lambda u: self._pca_head_forward(
                        self._encode_continuous(u.unsqueeze(0)).squeeze(0)),
                    x_t, create_graph=False, strict=True).detach().cpu().numpy()
                # finite-difference Jacobian, step h
                jac_fd = np.zeros_like(jac_ag)
                for g in range(x.size):
                    e = np.zeros_like(x); e[g] = h
                    z_plus = self.encode((x + e)[None, :])[0]
                    z_minus = self.encode((x - e)[None, :])[0]
                    jac_fd[:, g] = (z_plus - z_minus) / (2 * h)
                # Flatten and compute cosine across the whole Jacobian
                flat_ag = jac_ag.ravel(); flat_fd = jac_fd.ravel()
                na = np.linalg.norm(flat_ag); nf = np.linalg.norm(flat_fd)
                if na < 1e-12 or nf < 1e-12:
                    cos = 0.0
                else:
                    cos = float(np.dot(flat_ag, flat_fd) / (na * nf))
                cosines.append(cos)
            per_cell_cosine[f"h={h:.0e}"] = cosines
        all_stable = all(
            all(c >= tol_cosine for c in cs)
            for cs in per_cell_cosine.values())
        halt_reason = None
        if not all_stable:
            halt_reason = (
                "autograd vs finite-difference Jacobian cosine falls "
                f"below {tol_cosine} on at least one (cell × step) "
                "cell; A4 halt rule invoked.")
        return {
            "continuous_input_supported": True,
            "per_cell_cosine": per_cell_cosine,
            "stable": all_stable,
            "halt_reason": halt_reason,
            "checkpoint": self._ckpt,
        }

    # ─── internals ────────────────────────────────────────────────────
    def _check_fit(self):
        if self._pca_head is None:
            raise RuntimeError(f"{self.name} must be fit() before use")

    def _ensure_loaded(self):
        if self._ckpt is not None:
            return
        if self._checkpoint_path is None:
            raise RuntimeError(
                "ScGPTRep requires a checkpoint path. Pass "
                "checkpoint_path=... or set $SCGPT_CKPT.")
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"scGPT checkpoint not found at {self._checkpoint_path}")
        # Hash the file for provenance.
        sha = _sha256(self._checkpoint_path)
        try:
            scgpt = importlib.import_module("scgpt")
        except Exception as e:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "scGPT package is not installed. "
                "`pip install scgpt` and provide a checkpoint path.") from e
        # The actual loader hook lives in scGPT itself; we delegate.
        encoder, native_dim, continuous_input_supported = _load_scgpt_encoder(
            scgpt, self._checkpoint_path)
        self._encoder = encoder
        self._ckpt = ScGPTCheckpoint(
            name=self._checkpoint_path.stem,
            path=self._checkpoint_path,
            sha256=sha,
            native_dim=int(native_dim),
            continuous_input_supported=bool(continuous_input_supported),
        )

    def _embed_with_scgpt(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the frozen scGPT encoder on a batch
        of cells. Returns a ``(n_cells, native_dim)`` array."""
        if self._encoder is None:
            raise RuntimeError("scGPT encoder not loaded")
        return _scgpt_forward(self._encoder, X)

    def _encode_continuous(self, X_torch):
        """Continuous-input forward pass used only by autograd and A4.
        Delegates to the checkpoint's continuous-input head when
        available."""
        if self._encoder is None:
            raise RuntimeError("scGPT encoder not loaded")
        return _scgpt_continuous_forward(self._encoder, X_torch)

    def _pca_head_forward(self, z_native_torch):
        """Apply the fitted PCA head inside a torch graph (so autograd
        can propagate from the output back to the gene-space input)."""
        import torch
        W = torch.tensor(self._pca_head.components_, dtype=z_native_torch.dtype,
                          device=z_native_torch.device)
        mu = torch.tensor(self._pca_head.mean_, dtype=z_native_torch.dtype,
                          device=z_native_torch.device)
        return (z_native_torch - mu) @ W.T


# ─── scGPT package glue — concrete wiring kept in one place ────────────
def _load_scgpt_encoder(scgpt, path: Path):
    """Load a frozen scGPT encoder from the given checkpoint path.

    The exact loader entry point depends on the scGPT release; this
    function is kept as a single shim so a release-specific tweak
    only needs to change here. On failure it raises
    :class:`NotImplementedError` with the attempted path so Task 1
    can report a precise feasibility blocker.
    """
    raise NotImplementedError(
        "_load_scgpt_encoder is wired to the scGPT release at run "
        "time. See docstring. Task 1 populates this once the "
        "checkpoint and scgpt version are confirmed.")


def _scgpt_forward(encoder, X: np.ndarray) -> np.ndarray:
    """Discrete (binned) forward pass of scGPT on a numpy batch."""
    raise NotImplementedError(
        "_scgpt_forward is populated in Task 1 once the scGPT loader "
        "returns a concrete encoder.")


def _scgpt_continuous_forward(encoder, X_torch):
    """Continuous-input forward pass, used only when a checkpoint
    exposes a continuous embedding path (A4 feasibility)."""
    raise NotImplementedError(
        "_scgpt_continuous_forward is populated in Task 1 once the "
        "scGPT loader returns a concrete encoder.")


def _as_dense(adata_or_array) -> np.ndarray:
    from .linear import _as_dense as _impl
    return _impl(adata_or_array)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
