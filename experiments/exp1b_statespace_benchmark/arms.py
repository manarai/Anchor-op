"""Six state-space arms for the exp1b benchmark (coding only; feasibility gate runs these but
nothing is fit at full benchmark scale here — see PREREG_exp1b.md).

Arms
----
1. PCA(log1p)             — paper-1 reference, Part A v0.3.4 arm, fit on log1p-normalized
                            NT control counts.
2. FA(log1p, QR)          — scJDO's default state space, fit on log1p controls, QR-
                            orthonormalized loadings (same convention as anchor-op v0.3.4).
3. LDVAE encoder          — scvi.model.LinearSCVI(n_latent=30), raw counts, gem-group batch
                            key; encode = posterior-mean latent `get_latent_representation()`.
4. LDVAE loadings         — same trained LDVAE, decoder loadings QR-orthonormalized, used
                            as a linear basis on log1p-normalized counts (parallels PCA/FA).
5. scVI                   — scvi.model.SCVI(n_latent=30), raw counts, gem-group batch key;
                            encode = posterior-mean latent (no sampling).
6. scGPT fixed-ingestion  — scGPT_human (continuous-input head), top-k token set with
                            TARGET force-included at position 0 and deterministic
                            tie-breaking via np.argsort(..., kind='stable'); PCA head to 30.
7. scGPT random-weights   — same architecture + ingestion as 6, weights re-initialised
                            with a fixed seed (pretraining control).

Knockdown definition (feasibility and full benchmark)
-----------------------------------------------------
- log1p arms (PCA, FA, LDVAE-loadings, scGPT): scale the target's log1p expression via
  expm1 → multiply by (1 − κ) → log1p (base-class knockdown_scale_difference with
  input_space='log1p').
- count arms (LDVAE-encoder, scVI): scale the target's raw count by (1 − κ) BEFORE log1p
  of the ingestion path; library size is left as observed, posterior-mean encoding of the
  scaled cell matrix is used.

Nothing fits full-benchmark scale here. VAE instances are produced by the feasibility
harness (one model per screen for the five-clause gate). The full 3-seed × 3-screen
benchmark is gated on user confirmation of compute venue (local vs cluster GPU).
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class Arm(abc.ABC):
    """A six-arm benchmark arm. Subclasses implement `fit` and `encode`.

    Shape conventions
    -----------------
    - fit(adata_ctrl, batch_key)       -> self, stashing fit state
    - encode(X)                        -> (n, dim)   posterior-mean (no sampling for VAEs)
    - knockdown_u_z(X_ctrl, target_g,
                    kappa)              -> (dim,)    mean(E(x_kd) − E(x)) across cells
    - input_space                       -> 'log1p' or 'count' (whichever the ingestion sees)

    `knockdown_u_z` is the primary feasibility-gate statistic.
    """

    name: str
    dim: int
    input_space: str   # 'log1p' or 'count' -> determines kd scaling path

    def __init__(self, dim: int = 30, name: Optional[str] = None):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.name = name or type(self).__name__

    @abc.abstractmethod
    def fit(self, adata, batch_key: Optional[str] = None) -> "Arm":
        """Fit arm state on NT control cells only.

        For VAE arms this trains the model. For linear and scGPT arms this does the
        corresponding one-shot fit / load. Must return self.
        """

    @abc.abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode (n, G) -> (n, dim). Posterior MEAN for VAE arms (no sampling)."""

    def _apply_kd_scaling(self, X: np.ndarray, target_gene_idx: int,
                           kappa: float) -> np.ndarray:
        """Apply knockdown scaling in the arm's native input space."""
        Y = X.copy()
        if self.input_space == "log1p":
            y = np.expm1(Y[:, target_gene_idx])
            y = np.clip(y, 0.0, None)
            Y[:, target_gene_idx] = np.log1p(y * (1.0 - kappa))
        elif self.input_space == "count":
            Y[:, target_gene_idx] = Y[:, target_gene_idx] * (1.0 - kappa)
        else:
            raise ValueError(f"unknown input_space {self.input_space!r}")
        return Y

    def knockdown_u_z(self, X_ctrl: np.ndarray, target_gene_idx: int,
                       kappa: float = 0.7) -> np.ndarray:
        """Mean knockdown-scale finite difference on control cells.

            u_z = mean_i [ E(x_i^{κ}) − E(x_i) ]

        where x_i^{κ} scales the target gene by (1 − κ) in the arm's native input
        space (log1p for arms 1/2/4/6/7; raw count for arms 3/5).
        """
        X_ctrl = np.asarray(X_ctrl, dtype=np.float64)
        X_perturbed = self._apply_kd_scaling(X_ctrl, target_gene_idx, kappa)
        return (self.encode(X_perturbed) - self.encode(X_ctrl)).mean(axis=0)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"


# -----------------------------------------------------------------------------
# Arm 1: PCA(log1p).
# -----------------------------------------------------------------------------
class PCALog1pArm(Arm):
    """Fit sklearn PCA on log1p-normalized controls. Encode = (log1p(x) − μ)·V.

    Loadings V have shape (dim, G); columns are orthonormal by construction.
    """

    input_space = "log1p"

    def __init__(self, dim: int = 30, name: Optional[str] = None):
        super().__init__(dim=dim, name=name or f"PCA(log1p)-{dim}")
        self.mean_ = None
        self.components_ = None   # (dim, G)
        self.n_genes_ = None

    def fit(self, adata, batch_key: Optional[str] = None) -> "PCALog1pArm":
        from sklearn.decomposition import PCA
        X = _as_log1p_dense(adata)
        if X.shape[1] < self.dim:
            raise ValueError(f"{self.name}: dim={self.dim} > n_genes={X.shape[1]}")
        pca = PCA(n_components=self.dim, random_state=0)
        pca.fit(X)
        self.mean_ = pca.mean_.astype(np.float64)
        self.components_ = pca.components_.astype(np.float64)
        self.n_genes_ = X.shape[1]
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T


# -----------------------------------------------------------------------------
# Arm 2: FA(log1p, QR-orthonormalized).
# -----------------------------------------------------------------------------
class FAQRLog1pArm(Arm):
    """sklearn FactorAnalysis fit on log1p-normalized controls, loadings QR-orthonormalized.

    Encoder: (log1p(x) − μ) · Q, where Q = QR(components_.T) has orthonormal columns.
    This matches anchor-op v0.3.4's FA(QR) arm from Part A.
    """

    input_space = "log1p"

    def __init__(self, dim: int = 30, name: Optional[str] = None):
        super().__init__(dim=dim, name=name or f"FA(log1p,QR)-{dim}")
        self.mean_ = None
        self.Q_ = None   # (G, dim) orthonormal columns
        self.n_genes_ = None

    def fit(self, adata, batch_key: Optional[str] = None) -> "FAQRLog1pArm":
        from sklearn.decomposition import FactorAnalysis
        X = _as_log1p_dense(adata)
        if X.shape[1] < self.dim:
            raise ValueError(f"{self.name}: dim={self.dim} > n_genes={X.shape[1]}")
        fa = FactorAnalysis(n_components=self.dim, random_state=0,
                             tol=1e-2, max_iter=1000)
        fa.fit(X)
        self.mean_ = fa.mean_.astype(np.float64)
        W_raw = fa.components_.T.astype(np.float64)        # (G, dim), not orthonormal
        Q, _ = np.linalg.qr(W_raw, mode="reduced")          # (G, dim), orthonormal cols
        self.Q_ = Q
        self.n_genes_ = X.shape[1]
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.Q_


# -----------------------------------------------------------------------------
# Arm 3: LDVAE encoder (posterior mean of z given raw counts).
# Arm 4: LDVAE loadings (decoder loadings, QR-orthonormalized, used as linear basis).
# -----------------------------------------------------------------------------
class LDVAEEncoderArm(Arm):
    """scvi.model.LinearSCVI with n_latent=dim, raw counts, batch_key = gem group.

    `encode(X)` returns the posterior mean of z given the (count) matrix X. The raw-count
    AnnData must be wired via `fit(adata, batch_key='<gem_col>')` with the SAME var_names
    the counts matrix will use at `encode` time.
    """

    input_space = "count"

    def __init__(self, dim: int = 30, name: Optional[str] = None, train_seed: int = 0,
                 n_epochs: int = 400):
        super().__init__(dim=dim, name=name or f"LDVAE-enc-{dim}(seed={train_seed})")
        self.train_seed = int(train_seed)
        self.n_epochs = int(n_epochs)
        self.model = None
        self._template_adata = None
        self.n_genes_ = None

    def fit(self, adata, batch_key: Optional[str] = None) -> "LDVAEEncoderArm":
        import scvi
        import torch
        scvi.settings.seed = self.train_seed
        torch.manual_seed(self.train_seed)
        a = adata.copy()
        scvi.model.LinearSCVI.setup_anndata(a, batch_key=batch_key, layer=None)
        self.model = scvi.model.LinearSCVI(a, n_latent=self.dim)
        self.model.train(max_epochs=self.n_epochs, use_gpu=False, enable_progress_bar=False)
        self._template_adata = a
        self.n_genes_ = a.shape[1]
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        import anndata as ad
        X = np.asarray(X, dtype=np.float64)
        a = ad.AnnData(X.astype(np.float32), var=self._template_adata.var.copy())
        # inherit obs schema (batch key) from template; LinearSCVI needs the batch column.
        for col in self._template_adata.obs.columns:
            a.obs[col] = self._template_adata.obs[col].iloc[0]
        return self.model.get_latent_representation(a).astype(np.float64)


class LDVAELoadingsArm(Arm):
    """LDVAE's decoder loadings QR-orthonormalized, used as a linear basis on log1p counts.

    Loadings are read via `model.get_loadings()` from a `LinearSCVI` already trained by
    `LDVAEEncoderArm.fit`. Then QR to orthonormalize. Encoder: `(log1p(x) − μ_log) · Q`.

    `fit(ldvae_arm)` reuses the trained encoder (do not retrain). This makes LDVAE-encoder
    vs LDVAE-loadings a within-model comparison: encoder gets the nonlinear transform,
    loadings get only the linear decoder basis.
    """

    input_space = "log1p"

    def __init__(self, dim: int = 30, name: Optional[str] = None):
        super().__init__(dim=dim, name=name or f"LDVAE-load(QR)-{dim}")
        self.mean_ = None
        self.Q_ = None
        self.n_genes_ = None

    def fit(self, ldvae_arm: LDVAEEncoderArm,
             batch_key: Optional[str] = None) -> "LDVAELoadingsArm":
        if ldvae_arm.model is None:
            raise RuntimeError("LDVAELoadingsArm.fit requires a trained LDVAEEncoderArm")
        loadings_df = ldvae_arm.model.get_loadings()
        var_names = list(ldvae_arm._template_adata.var_names)
        W_raw = loadings_df.loc[var_names].values.astype(np.float64)   # (G, dim)
        Q, _ = np.linalg.qr(W_raw, mode="reduced")
        self.Q_ = Q
        self.n_genes_ = W_raw.shape[0]
        # Mean fit on log1p-normalised ctrl matrix (stored at feasibility time).
        self.mean_ = None   # set by `set_mean` after log1p control fit
        return self

    def set_mean(self, X_log1p_ctrl: np.ndarray):
        X_log1p_ctrl = np.asarray(X_log1p_ctrl, dtype=np.float64)
        self.mean_ = X_log1p_ctrl.mean(axis=0)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("LDVAELoadingsArm: call set_mean(log1p_ctrl) after fit")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.Q_


# -----------------------------------------------------------------------------
# Arm 5: scVI (nonlinear encoder/decoder, posterior-mean latent).
# -----------------------------------------------------------------------------
class SCVIArm(Arm):
    input_space = "count"

    def __init__(self, dim: int = 30, name: Optional[str] = None, train_seed: int = 0,
                 n_epochs: int = 400):
        super().__init__(dim=dim, name=name or f"scVI-{dim}(seed={train_seed})")
        self.train_seed = int(train_seed)
        self.n_epochs = int(n_epochs)
        self.model = None
        self._template_adata = None
        self.n_genes_ = None

    def fit(self, adata, batch_key: Optional[str] = None) -> "SCVIArm":
        import scvi
        import torch
        scvi.settings.seed = self.train_seed
        torch.manual_seed(self.train_seed)
        a = adata.copy()
        scvi.model.SCVI.setup_anndata(a, batch_key=batch_key, layer=None)
        self.model = scvi.model.SCVI(a, n_latent=self.dim)
        self.model.train(max_epochs=self.n_epochs, use_gpu=False, enable_progress_bar=False)
        self._template_adata = a
        self.n_genes_ = a.shape[1]
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        import anndata as ad
        X = np.asarray(X, dtype=np.float64)
        a = ad.AnnData(X.astype(np.float32), var=self._template_adata.var.copy())
        for col in self._template_adata.obs.columns:
            a.obs[col] = self._template_adata.obs[col].iloc[0]
        return self.model.get_latent_representation(a).astype(np.float64)


# -----------------------------------------------------------------------------
# Arm 6: scGPT_human with fixed, deterministic ingestion.
# Arm 7: scGPT with random weights, same ingestion (pretraining control).
# -----------------------------------------------------------------------------
class ScGPTFixedIngestionArm(Arm):
    """scGPT_human (continuous-input head) with:

      - target gene force-included at token position 0 (bypasses argsort)
      - deterministic tie-breaking via np.argsort(..., kind='stable')
      - identical ingestion for real perturbed cells and in-silico knockdowns
      - PCA head fitted on controls to collapse native-512 -> dim

    This fixes the two mechanisms the exp1 A4 diagnosis identified as causing the HALT:
    (i) tokenization drop-out of mid-expression targets, and (iii) argsort-tie-breaking
    pipeline non-determinism. See
    `experiments/exp1_statespace/REPORT_exp1_statespace.md` for the diagnosis.
    """

    input_space = "log1p"

    def __init__(self, dim: int = 30, name: Optional[str] = None,
                 max_seq_len: int = 1200, random_weights: bool = False):
        super().__init__(dim=dim, name=name or
                          ("scGPT-rand" if random_weights else "scGPT-pre") +
                          f"-fix-{dim}")
        self.max_seq_len = int(max_seq_len)
        self.random_weights = bool(random_weights)
        self.scgpt = None          # underlying ScGPTRep (from src/anchorop/state_space/scgpt.py)
        self.pca_head_ = None      # (512, dim)
        self.ctrl_mean_log1p_ = None   # (G,)
        self.gene_vocab_ids_ = None    # (G,) token ids for each ingestion gene
        self.n_genes_ = None

    def fit(self, adata, batch_key: Optional[str] = None) -> "ScGPTFixedIngestionArm":
        """Load scGPT_human (or random-weight twin) and fit a PCA head on controls.

        Delegates checkpoint load + vocab + model setup to `anchorop.state_space.scgpt.ScGPTRep`;
        then re-fits the PCA head so this arm owns a fresh head trained on controls under
        this arm's deterministic ingestion.
        """
        from anchorop.state_space.scgpt import ScGPTRep
        from sklearn.decomposition import PCA
        self.scgpt = ScGPTRep(d_out=self.dim, max_seq_len=self.max_seq_len,
                               random_weights=self.random_weights)
        # scGPT fit loads the checkpoint; we override tokenization in `_embed_native` below
        # so we don't rely on scGPT's own fit() to produce a usable head.
        self.scgpt._ensure_loaded()
        gene_names = (list(adata.var["gene_name"]) if "gene_name" in adata.var
                       else list(adata.var_names))
        self.scgpt._encoder["gene_names"] = gene_names
        self.gene_vocab_ids_ = np.array(
            [self.scgpt._encoder["vocab"][g] for g in gene_names], dtype=np.int64)
        self.n_genes_ = adata.shape[1]
        from anchorop.state_space.linear import _as_dense
        X_ctrl = _as_dense(adata)
        self.ctrl_mean_log1p_ = X_ctrl.mean(axis=0)
        # Embed controls with this arm's deterministic + force-include-NOT ingestion
        # (controls get a plain stable-argsort top-k; force-include only enters at
        # knockdown_u_z, where we pin the target into position 0 for BOTH ctrl and kd).
        Z_native = self._embed_native(X_ctrl, target_gene_idx=None)
        pca = PCA(n_components=self.dim, random_state=0)
        pca.fit(Z_native)
        self.pca_head_ = pca
        return self

    def _tokenize_force_include(self, x_log1p: np.ndarray,
                                 target_gene_idx: Optional[int]) -> tuple:
        """Return (src_ids, values, pad_mask) with stable argsort and optional
        target force-include at token position 0.

        - np.argsort(..., kind='stable') removes the pipeline non-determinism
          identified as cause (iii) in the exp1 A4 diagnosis.
        - When `target_gene_idx is not None`, the target gene is placed at position 0
          regardless of its expression rank, removing cause (i).
        """
        import torch
        encoder = self.scgpt._encoder
        cfg = encoder["config"]
        max_seq_len = int(cfg.get("max_seq_len", 1200))
        pad_value = float(cfg.get("pad_value", -2.0))
        pad_idx = int(encoder["vocab"][cfg.get("pad_token", "<pad>")])
        gene_ids = self.gene_vocab_ids_

        n = x_log1p.shape[0]
        src = np.full((n, max_seq_len), pad_idx, dtype=np.int64)
        values = np.full((n, max_seq_len), pad_value, dtype=np.float32)
        mask = np.ones((n, max_seq_len), dtype=bool)   # True = pad

        for i in range(n):
            row = x_log1p[i]
            order = np.argsort(-np.abs(row), kind="stable")
            if target_gene_idx is not None:
                order = order[order != target_gene_idx]
                order = np.concatenate([[target_gene_idx], order])
            take = order[:max_seq_len]
            src[i, :len(take)] = gene_ids[take]
            values[i, :len(take)] = row[take].astype(np.float32)
            mask[i, :len(take)] = False
        return (torch.from_numpy(src), torch.from_numpy(values),
                torch.from_numpy(mask))

    def _embed_native(self, X: np.ndarray,
                       target_gene_idx: Optional[int]) -> np.ndarray:
        """Native 512-dim embedding under this arm's deterministic ingestion.

        Mirrors `_scgpt_continuous_forward` in src/anchorop/state_space/scgpt.py but
        with the fixed tokenizer above substituted in.
        """
        import torch
        encoder = self.scgpt._encoder
        model = encoder["model"]
        src, values, pad_mask = self._tokenize_force_include(X, target_gene_idx)
        with torch.no_grad():
            out = model(src=src, values=values.float(), src_key_padding_mask=pad_mask,
                        batch_labels=None, CLS=False, CCE=False, MVC=False, ECS=False,
                        do_sample=False)
        return out["cell_emb"].detach().cpu().numpy().astype(np.float64)

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z_native = self._embed_native(X, target_gene_idx=None)
        return self.pca_head_.transform(Z_native)

    def knockdown_u_z(self, X_ctrl: np.ndarray, target_gene_idx: int,
                       kappa: float = 0.7) -> np.ndarray:
        X_ctrl = np.asarray(X_ctrl, dtype=np.float64)
        X_perturbed = self._apply_kd_scaling(X_ctrl, target_gene_idx, kappa)
        # Force-include the target in BOTH ctrl and kd ingestions so the only
        # difference between ingestion sequences is the target's value.
        Z0 = self.pca_head_.transform(
            self._embed_native(X_ctrl, target_gene_idx=target_gene_idx))
        Z1 = self.pca_head_.transform(
            self._embed_native(X_perturbed, target_gene_idx=target_gene_idx))
        return (Z1 - Z0).mean(axis=0)


# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------
def _as_log1p_dense(adata_or_array) -> np.ndarray:
    """Return a dense log1p-normalized matrix.

    If the input is already log1p-normalized (non-negative, max < 20), return as-is.
    Otherwise apply `np.log1p` after library-size normalization to target_sum=1e4.
    """
    import scipy.sparse as sp
    if hasattr(adata_or_array, "X"):
        X = adata_or_array.X
    else:
        X = adata_or_array
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    # Heuristic: if already log1p, do nothing.
    if (X >= 0).all() and X.max() < 20:
        return X
    # Normalize to 1e4 counts per cell then log1p.
    counts = X.sum(axis=1, keepdims=True)
    counts = np.where(counts > 0, counts, 1.0)
    X_norm = X * (1e4 / counts)
    return np.log1p(X_norm)
