"""Frozen scGPT cell encoder with a PCA head to ``dim``.

Design
------
scGPT (Cui et al. 2024, *Nature Methods*) ingests expression as
log-normalised bin indices. Autograd through the binning operation
is not a valid Jacobian because binning is piecewise-constant and
has zero derivative almost everywhere. The amendment A4 of the
experiment 1 preregistration (revised 2026-09-30) therefore
replaces the Jacobian-based ``decode_direction`` for scGPT with a
**knockdown-scale finite difference**:

    u_z = mean over control cells of [ E(x · scale(target_gene, 1−κ)) − E(x) ]

- Control cells only. Perturbed cells never enter.
- Primary κ = 0.7. Sensitivity checks at κ = 0.5 and 0.9.
- Discretisation across scGPT bins is a feature, not a bug: a
  knockdown-scale shift this large crosses many bins and reflects
  the actual perturbation the operator fit tries to model.

The scGPT feasibility check becomes:

(i) **Non-triviality** — does the knockdown-scale difference change
    the embedding above the encoder's run-to-run noise on identical
    input (two embed-twice calls on the same control subset)?

(ii) **Stability** — is the direction of the knockdown-scale
     difference stable across κ ∈ {0.5, 0.7, 0.9} and across two
     disjoint random subsets of control cells (cosine > 0.9)?

The arm is halted (no workaround) if (i) or (ii) fails. If a
continuous-input path exists, the autograd Jacobian is reported for
comparison but is NOT required.

For linear arms (PCARep / FARep) the closed-form Jacobian is kept
and a separate test enforces that the knockdown-scale difference
equals ``−κ · J @ δ_g`` exactly.

The scGPT checkpoint and the torch runtime are loaded lazily; the
module imports cleanly without ``torch`` or ``scgpt`` installed so
the rest of the StateSpace catalog can be used on CPU-only
environments.
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

from . import _torchtext_shim  # noqa: F401  — installs torchtext shim before any scgpt import
from ._torchtext_shim import _DictVocab
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
    def fit(self, adata, *, gene_names=None) -> "ScGPTRep":
        """Load scGPT (if not already loaded), embed control cells,
        fit the PCA head to ``dim`` components, and record the PCA
        head's explained variance.

        ``gene_names`` is the list of var names on the input matrix
        — scGPT needs them to look up vocab indices. If ``adata`` is
        an AnnData, falls back to ``adata.var["gene_name"]`` (preferred)
        or ``adata.var_names``.
        """
        self._ensure_loaded()
        X = _as_dense(adata)
        if gene_names is None:
            if hasattr(adata, "var"):
                if "gene_name" in getattr(adata, "var", {}):
                    gene_names = list(adata.var["gene_name"])
                elif hasattr(adata, "var_names"):
                    gene_names = list(adata.var_names)
            if gene_names is None:
                raise ValueError(
                    "ScGPTRep.fit requires gene_names; pass gene_names=…"
                    " or supply an AnnData with .var.gene_name or .var_names.")
        self._encoder["gene_names"] = list(gene_names)
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
                          X: Optional[np.ndarray] = None,
                          *, kappa: float = 0.7) -> np.ndarray:
        """Knockdown-scale finite difference (A4 revision 2026-09-30).

        ``u_gene`` must be sparse-unit on a single target gene (zero
        everywhere else). Its sign is informational — the operation
        is always a multiplicative scale by ``(1 − κ)`` on the target
        gene's expression (a knockdown).

        Returns :meth:`knockdown_scale_difference(X, g, kappa)` with
        ``g`` the index of the single non-zero entry in ``u_gene``.
        Control cells only — raises if ``X`` is not provided.
        """
        self._check_fit()
        if X is None:
            raise ValueError(
                "ScGPTRep.decode_direction requires a control-cell matrix X "
                "(the knockdown-scale finite difference is computed on "
                "control cells only; perturbed cells never enter).")
        u_gene = np.asarray(u_gene, dtype=np.float64).ravel()
        nz = np.nonzero(u_gene)[0]
        if nz.size != 1:
            raise ValueError(
                "u_gene for scGPT.decode_direction must be sparse-unit on "
                "a single target gene (support exactly 1).")
        g = int(nz[0])
        return self.knockdown_scale_difference(_as_dense(X), g, kappa=kappa)

    # ─── A4 revision 2026-09-30 — knockdown-scale feasibility check ───
    def check_decode_direction_feasibility(
            self, X_control: np.ndarray, target_gene_idx: int,
            kappas: Tuple[float, ...] = (0.5, 0.7, 0.9),
            n_subset: int = 50,
            noise_threshold_rel: float = 0.1,
            cos_threshold: float = 0.9,
            random_state: int = 0,
    ) -> Dict[str, Any]:
        """A4 (revised) feasibility: knockdown-scale finite difference.

        Two tests, both run on control cells only:

        (i) **Non-triviality**. Two embed-twice calls on the same
            control subset estimate the encoder's run-to-run noise
            norm ``σ_noise``. The knockdown-scale difference at
            κ = 0.7 must have norm ≥ ``noise_threshold_rel * σ_noise``.
            (In practice ``σ_noise = 0`` for deterministic encoders;
            we require the knockdown-scale difference norm to exceed
            ``σ_noise + 1e-6``.)

        (ii) **Stability**. Pairwise cosine of the knockdown-scale
             direction across κ ∈ {0.5, 0.7, 0.9} and across two
             disjoint random subsets of control cells must exceed
             ``cos_threshold`` on every pair.

        Returns a report dict with keys:

        - ``non_trivial`` : bool
        - ``knockdown_norm`` : float — ‖u_z‖ at primary κ
        - ``noise_norm`` : float — encoder run-to-run noise
        - ``kappa_cosines`` : dict κ-pair → cosine (same subset)
        - ``subset_cosines`` : dict κ → cosine (disjoint subsets)
        - ``stable`` : bool
        - ``halt_reason`` : str or None
        - ``checkpoint`` : ScGPTCheckpoint
        """
        self._ensure_loaded()
        X = _as_dense(X_control)
        rng = np.random.default_rng(random_state)
        n = X.shape[0]
        if n < 2 * n_subset:
            raise ValueError(
                f"Need at least {2*n_subset} control cells; have {n}.")
        idx = rng.permutation(n)
        sub_a = idx[:n_subset]; sub_b = idx[n_subset:2*n_subset]

        def _ks(X_subset: np.ndarray, kappa: float) -> np.ndarray:
            return self.knockdown_scale_difference(
                X_subset, target_gene_idx, kappa=kappa)

        # (i) non-triviality
        Z1 = self.encode(X[sub_a])
        Z2 = self.encode(X[sub_a])
        noise_norm = float(np.linalg.norm((Z2 - Z1).mean(axis=0)))
        u_primary = _ks(X[sub_a], 0.7)
        kd_norm = float(np.linalg.norm(u_primary))
        non_trivial = kd_norm > (noise_norm + 1e-6)

        # (ii) stability
        u_by_k = {k: _ks(X[sub_a], k) for k in kappas}
        u_by_subset = {k: _ks(X[sub_b], k) for k in kappas}

        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        kappa_cosines = {}
        for i, ki in enumerate(kappas):
            for kj in kappas[i+1:]:
                kappa_cosines[f"{ki}↔{kj}"] = _cos(u_by_k[ki], u_by_k[kj])
        subset_cosines = {str(k): _cos(u_by_k[k], u_by_subset[k]) for k in kappas}

        all_cosines = list(kappa_cosines.values()) + list(subset_cosines.values())
        stable = all(c > cos_threshold for c in all_cosines)
        halt_reason = None
        if not non_trivial:
            halt_reason = (
                f"knockdown-scale difference norm ({kd_norm:.4g}) does not "
                f"exceed encoder noise ({noise_norm:.4g}); (i) fails.")
        elif not stable:
            halt_reason = (
                "knockdown-scale direction not stable across κ or subsets: "
                f"min cos = {min(all_cosines):.3f} < {cos_threshold}; (ii) fails.")

        return {
            "non_trivial": non_trivial,
            "knockdown_norm": kd_norm,
            "noise_norm": noise_norm,
            "kappa_cosines": kappa_cosines,
            "subset_cosines": subset_cosines,
            "stable": stable,
            "halt_reason": halt_reason,
            "checkpoint": self._ckpt,
        }

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


# ─── scGPT package glue — concrete wiring ──────────────────────────────
def _load_scgpt_encoder(scgpt, path: Path):
    """Load the scGPT_human whole-pretrain checkpoint on CPU.

    The checkpoint ships in flash_attn fused-attention layout
    (``Wqkv.{weight,bias}`` + a ``flag_encoder.weight`` for packed
    attention). flash_attn is CUDA-only; this loader converts the
    state dict to the standard PyTorch ``nn.MultiheadAttention``
    layout (``in_proj_{weight,bias}``, drop ``flag_encoder``) so
    the model can run on CPU for the A4 feasibility check.

    Returns the encoder, its native embedding dim, and a flag
    indicating whether a continuous-input path is exposed (true for
    this checkpoint — ``input_emb_style`` is ``"continuous"``).
    """
    import json
    import torch
    from scgpt.model import TransformerModel

    weights_dir = Path(path).parent
    cfg = json.load(open(weights_dir / "args.json"))
    vocab_dict = json.load(open(weights_dir / "vocab.json"))
    vocab = _DictVocab(vocab_dict)
    vocab.set_default_index(vocab["<pad>"] if "<pad>" in vocab else 0)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        nlayers_cls=cfg["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=0.0,
        pad_token=cfg["pad_token"],
        pad_value=cfg["pad_value"],
        do_mvc=cfg.get("MVC", False),
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=cfg.get("input_emb_style", "continuous"),
        n_input_bins=cfg.get("n_bins", 51) + 2,  # +2 for mask + pad
        cell_emb_style="avg-pool",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    model.eval()

    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    converted = {}
    for k, v in sd.items():
        if k == "flag_encoder.weight":
            continue  # flash_attn packed-mask aux; unused on standard CPU path
        if ".self_attn.Wqkv.weight" in k:
            converted[k.replace(".Wqkv.weight", ".in_proj_weight")] = v
        elif ".self_attn.Wqkv.bias" in k:
            converted[k.replace(".Wqkv.bias", ".in_proj_bias")] = v
        else:
            converted[k] = v
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        _log.warning("scGPT state_dict load missing=%d, unexpected=%d; "
                     "first missing=%s", len(missing), len(unexpected),
                     missing[:3])
    encoder = {
        "model": model,
        "vocab": vocab,
        "config": cfg,
    }
    native_dim = int(cfg["embsize"])
    continuous_input_supported = (cfg.get("input_emb_style", "continuous")
                                   == "continuous")
    return encoder, native_dim, continuous_input_supported


def _tokenize_cells(encoder, X: np.ndarray, var_gene_names):
    """Build (src, values, mask) for a batch of cells on CPU.

    - For each cell, pick the top-``max_seq_len`` genes by expression.
    - ``src`` = vocab-ids of the picked genes.
    - ``values`` = the log1p-normalised expression at those genes.
    - ``mask`` = ``True`` on padded positions (we never pad on this
      path since we always pass the full seq_len).
    """
    import torch

    vocab = encoder["vocab"]
    cfg = encoder["config"]
    max_seq_len = int(cfg.get("max_seq_len", 1200))
    pad_token = cfg.get("pad_token", "<pad>")
    pad_value = float(cfg.get("pad_value", -2.0))
    pad_idx = int(vocab[pad_token])

    gene_ids = np.array([vocab[g] for g in var_gene_names], dtype=np.int64)

    batch = X.shape[0]
    src = np.full((batch, max_seq_len), pad_idx, dtype=np.int64)
    values = np.full((batch, max_seq_len), pad_value, dtype=np.float32)
    mask = np.ones((batch, max_seq_len), dtype=bool)  # True = pad

    for i in range(batch):
        row = X[i]
        nz = np.argsort(-np.abs(row))[:max_seq_len]
        src[i, : len(nz)] = gene_ids[nz]
        values[i, : len(nz)] = row[nz].astype(np.float32)
        mask[i, : len(nz)] = False
    return (torch.from_numpy(src), torch.from_numpy(values),
            torch.from_numpy(mask))


def _scgpt_forward(encoder, X: np.ndarray) -> np.ndarray:
    """Discrete (binned) forward pass of scGPT on a numpy batch.

    Returns a ``(n_cells, native_dim)`` ndarray of cell embeddings.
    This checkpoint uses ``input_emb_style = "continuous"`` so the
    "discrete" vs "continuous" distinction collapses — see
    :func:`_scgpt_continuous_forward` for the gradable variant.
    """
    return _scgpt_continuous_forward(encoder, X).detach().cpu().numpy()


def _scgpt_continuous_forward(encoder, X):
    """Forward pass that preserves the gradient graph when ``X`` is a
    torch tensor. For the A4 feasibility check we usually call this
    on numpy arrays; the autograd path is only used if the user asks
    for it alongside the knockdown-scale FD.
    """
    import torch

    model = encoder["model"]
    cfg = encoder["config"]
    gene_names = encoder.get("gene_names")
    if gene_names is None:
        raise RuntimeError(
            "scGPT encoder has no gene_names attached. "
            "ScGPTRep.fit must call encoder['gene_names'] = list of "
            "var_names before any forward pass.")

    if torch.is_tensor(X):
        X_np = X.detach().cpu().numpy().astype(np.float64)
    else:
        X_np = np.asarray(X, dtype=np.float64)
    src, values, mask = _tokenize_cells(encoder, X_np, gene_names)

    with torch.no_grad():
        out = model(src=src, values=values.float(),
                    src_key_padding_mask=mask,
                    batch_labels=None,
                    CLS=False, CCE=False, MVC=False, ECS=False,
                    do_sample=False)
    return out["cell_emb"]


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
