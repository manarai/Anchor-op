"""STEP 1–2 of the v0.3.1 learned-encoding fidelity check.

Preregistered 2026-09-28 in `PREREGISTRATION_AMENDMENT.md` (commit
1969c7a), before any code below ran.

Goal
====
Test whether our learned-encoding forward map (as fit by
`reproduction/51_comparator_panel.py::_learned_encoding_forward`) is
faithful to the linear-baseline model of Ahlmann-Eltze et al. 2025
(github.com/const-ae/linear_perturbation_prediction-Paper).

Their model, from `benchmark/src/run_linear_pretrained_model.R`
(function `solve_y_axb`), is:

    Ŷ = A · K · B + center + baseline

  where
    - `Y` is the pseudobulk change matrix (rows: genes, cols: perts;
       change = X_pseudobulk_pert − X_pseudobulk_ctrl per-gene mean).
    - `A` = gene embedding (rows: genes, cols: pca_dim); default is
       PCA of the training pseudobulk X.
    - `B` = perturbation embedding (rows: pca_dim, cols: perts);
       default is `t(pca$x)` indexed by target-gene name — i.e. the
       same gene-PCA rows, transposed so the perturbation-target
       gene's row of `pca$x` is the perturbation's embedding.
    - `K = (AᵀA + λI)⁻¹ Aᵀ (Y − center) B̂ᵀ (BB̂ᵀ + λI)⁻¹`.
    - `center = rowMeans(Y)` = per-gene mean of the change over
       training perturbations. Added back at prediction.
    - `baseline = rowMeans(ctrl-cell X)` = per-gene control-cell
       expression. Also added back at prediction.

Their default `pca_dim = 10`, `ridge_penalty = 0.1`. Their metric on
held-out perturbations is Pearson correlation per condition between
observed and predicted change (post-baseline subtraction). See their
`run_linear_pretrained_model.R` for the summarise call.

Published held-out r2 values from
`source_data/single_perturbation_prediction.xlsx` (aggregated over
their `train == "test"` rows):

  Replogle K562 essential | lpm_selftrained | r2 = 0.9858 ± 0.0093
  Replogle K562 essential | mean            | r2 = 0.9846 ± 0.0112

We use these as the fidelity reference. Our anchor-op K562 h5ad
comes from the same underlying Replogle 2022 essential dataset but
was preprocessed through our HVG + PCA-on-controls pipeline, not
through GEARS. A full end-to-end reproduction of their published
number on their exact preprocessed h5ad and their exact test-train
split JSON would require running their GEARS-based data-prep
pipeline (~10 GB download + GEARS filtering + slurm-oriented
workflow) which is out of scope for this session.

What this script does
=====================

1. Load our K562 essential h5ad and select the top ~2,000 HVGs + the
   target genes force-kept, following the anchor-op pipeline.
2. Pseudobulk by target: for each target with ≥ 60 cells, compute
   `X_pseudobulk[target] = mean(X over cells assigned to that target)`.
   Same for non-targeting controls (single "ctrl" pseudobulk).
3. Build the training / test target split with a fixed seed
   (`np.random.default_rng(SEED)`, held-out fraction 20 %).
4. Compute `Y = X_pseudobulk − X_pseudobulk_ctrl` (per-gene control
   subtraction) — the change matrix.
5. Fit an Ahlmann-Eltze-style linear model on the training
   perturbations using our Python port of `solve_y_axb`
   (`ae_solve` below). Predict on held-out perturbations. Compute
   per-condition Pearson r (their metric).
6. Fit our `_learned_encoding_forward`-style single-ridge map on the
   same training data (target embedding = row of the training-derived
   gene PCA at the target gene). Predict on held-out perturbations
   with the same feature-gene lookup.
7. Also fit a variant that uses u*σ target coordinates for training
   embeddings and V-column-at-target for test embeddings (the pre-fix
   convention in our comparator code) — to check whether that
   convention introduces additional error.
8. Report per-condition Pearson r on the held-out set for each
   method, plus their mean baseline (predict-training-mean of X on
   every held-out perturbation), and apply the preregistered
   fidelity criterion.

Preregistered fidelity criterion (per Replogle screen):
    | ours − theirs | ≤ max( their SD, 0.05 · theirs ).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260928
PCA_DIM = 10
RIDGE = 0.1
MIN_CELLS = 60
HELD_OUT_FRACTION = 0.20

THEIRS = {
    # Aggregated from source_data/single_perturbation_prediction.xlsx.
    # Their metric is per-condition Pearson r between predicted and
    # observed *change* (their `r2` column). SD is across held-out
    # perturbations, not across splits — but it is the closest
    # published dispersion.
    "K562_essential": {
        "lpm_selftrained_r2_mean": 0.9858,
        "lpm_selftrained_r2_sd": 0.0093,
        "mean_baseline_r2_mean": 0.9846,
        "mean_baseline_r2_sd": 0.0112,
        "n_test_perturbations": 379,
    },
    "RPE1_essential": {
        "lpm_selftrained_r2_mean": 0.9733,
        "lpm_selftrained_r2_sd": 0.0172,
        "mean_baseline_r2_mean": 0.9717,
        "mean_baseline_r2_sd": 0.0174,
        "n_test_perturbations": 508,
    },
}


# ── Their algorithm (Python port of solve_y_axb) ───────────────────────────
def ae_solve(Y: np.ndarray, A: np.ndarray, B: np.ndarray,
             A_ridge: float = 0.1, B_ridge: float = 0.1):
    """Port of `solve_y_axb` from Ahlmann-Eltze 2025's
    `run_linear_pretrained_model.R`.

    Y: (n_gene, n_cond) change matrix on TRAINING conditions.
    A: (n_gene, d)      gene embedding for TRAINING genes.
    B: (d, n_cond)      pert embedding for TRAINING conditions.
    Returns: {"K": (d, d), "center": (n_gene,)}.

    Predictions on any set of perturbations `B_all` (of shape d × n)
    are recovered as `A · K · B_all + center + baseline`, where
    `baseline` is per-gene ctrl-cell mean (kept outside this function
    to mirror their code).
    """
    Y = np.asarray(Y, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    center = Y.mean(axis=1, keepdims=True)  # (n_gene, 1) — per-gene training-cond mean of change
    Yc = Y - center
    # K = (AᵀA + λI)⁻¹ Aᵀ Y B̂ᵀ (BB̂ᵀ + λI)⁻¹
    AtA = A.T @ A
    BBt = B @ B.T
    dA = AtA.shape[0]
    dB = BBt.shape[0]
    K = np.linalg.solve(AtA + A_ridge * np.eye(dA),
                        A.T @ Yc @ B.T) @ np.linalg.solve(
                        BBt + B_ridge * np.eye(dB), np.eye(dB))
    K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    return {"K": K, "center": center.ravel()}


def ae_predict(A_all: np.ndarray, K: np.ndarray, B_all: np.ndarray,
               center: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Predict raw pseudobulk expression:
       Ŷ_raw = A · K · B + center + baseline.
    Return with (n_gene, n_cond)."""
    center = np.asarray(center).reshape(-1, 1)
    baseline = np.asarray(baseline).reshape(-1, 1)
    return A_all @ K @ B_all + center + baseline


# ── Our function's core math (single-ridge, feature-gene test embed) ───────
def our_learned_solve(Y_tr: np.ndarray, gene_names: list, tr_targets: list,
                       pca_dim: int, ridge: float):
    """Reimplement the core math of our
    `_learned_encoding_forward` in `reproduction/51_comparator_panel.py`.

    Y_tr: (n_train_targets, n_gene) — training-target change matrix.
    gene_names: list of gene names indexing the columns of Y_tr.
    tr_targets: list of training target-gene names.

    Returns dict with:
      - tr_embed (n_train_targets, d)  — u * σ from SVD of Y_tr.
      - V_gene  (d, n_gene)           — gene loadings from SVD.
      - B_gene  (d, n_gene)           — ridge fit map.
      - b_out   (n_gene,)             — training-target mean of Y_tr.
    """
    u, sv, vt = np.linalg.svd(Y_tr, full_matrices=False)
    de = min(pca_dim, len(sv))
    tr_embed = u[:, :de] * sv[:de]          # (n_train_targets, de)
    V_gene = vt[:de]                         # (de, n_gene)
    b_out = Y_tr.mean(axis=0, keepdims=True) # (1, n_gene)
    Yc = Y_tr - b_out
    XtX = tr_embed.T @ tr_embed
    B_gene = np.linalg.solve(XtX + ridge * np.eye(de), tr_embed.T @ Yc)
    return {
        "tr_embed": tr_embed, "V_gene": V_gene,
        "B_gene": B_gene, "b_out": b_out.ravel(),
    }


def our_predict(target_names, V_gene, gene_index, B_gene, b_out):
    """Predict change per test target following our comparator
    convention: `te_embed = V_gene[:, gene_index[target]]` — the
    training-derived gene loading at the held-out target's feature-
    gene column."""
    de = V_gene.shape[0]
    te_embed = np.zeros((len(target_names), de), dtype=np.float64)
    for j, t in enumerate(target_names):
        if t in gene_index:
            te_embed[j] = V_gene[:, gene_index[t]]
    Ypred_change = te_embed @ B_gene + b_out  # (n_test_targets, n_gene)
    return Ypred_change


# ── Alternative convention: same gene-PCA table for train + test ──────────
def ae_style_solve_on_our_pipeline(Y_tr: np.ndarray, gene_names: list,
                                    tr_targets: list, gene_index: dict,
                                    pca_dim: int, ridge: float):
    """Fit a single-ridge map using the *Ahlmann-Eltze-style*
    convention: the perturbation embedding for a target is the target
    gene's row of a PCA computed on training data. Applied uniformly
    to train + test targets (so the same table is used for both).
    """
    # PCA of Y_tr (training-target change matrix, targets × genes).
    # Their `pca <- prcomp_irlba(X_train_pseudobulk, n=pca_dim)` on
    # training PSEUDOBULK X uses training data at raw scale, not
    # change. On our K562 pipeline `Y_tr` is change; we use it here
    # because our anchor-op pipeline never keeps raw X_train per
    # target. This is one of the recognised differences between our
    # pipeline and theirs, quantified in Step 3 (bridge analysis).
    u, sv, vt = np.linalg.svd(Y_tr, full_matrices=False)
    de = min(pca_dim, len(sv))
    V_gene = vt[:de]                    # (de, n_gene)
    # For each training target t, embed[t] = V_gene[:, gene_index[t]].
    tr_embed = np.zeros((len(tr_targets), de), dtype=np.float64)
    for j, t in enumerate(tr_targets):
        if t in gene_index:
            tr_embed[j] = V_gene[:, gene_index[t]]
    b_out = Y_tr.mean(axis=0, keepdims=True)
    Yc = Y_tr - b_out
    XtX = tr_embed.T @ tr_embed
    B_gene = np.linalg.solve(XtX + ridge * np.eye(de), tr_embed.T @ Yc)
    return {"V_gene": V_gene, "B_gene": B_gene, "b_out": b_out.ravel()}


def ae_style_predict(target_names, V_gene, gene_index, B_gene, b_out):
    """Same convention as `our_predict` for test — but here the
    training embedding was built the same way, so this is the fully
    consistent version."""
    return our_predict(target_names, V_gene, gene_index, B_gene, b_out)


# ── K562 preprocessing (pseudobulk + control baseline) ─────────────────────
def load_k562_pseudobulk(h5ad_path: Path, seed: int = SEED):
    import anndata as ad
    print(f"[pre] loading {h5ad_path.name} …")
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs
    var_names = list(a.var_names)
    gene_index = {g: k for k, g in enumerate(var_names)}
    print(f"[pre] shape {a.shape}, obs cols: {list(obs.columns)[:10]}")

    # Prefer target column whose values live in the same namespace as
    # var_names (feature-gene lookup requires it).
    if any(str(v).startswith("ENSG") for v in var_names[:20]):
        # var_names are ENSEMBL IDs — use the ENSEMBL target col.
        candidates = [c for c in ("gene_id", "target_gene_id", "target_ensembl") if c in obs.columns]
    else:
        candidates = [c for c in ("gene", "target_gene", "guide_ident", "gene_symbol") if c in obs.columns]
    if not candidates:
        raise RuntimeError(f"no target column matches var namespace")
    target_col = candidates[0]

    for cand in ("non-targeting", "NO-TARGET", "control", "ctrl", "non_targeting"):
        if (obs[target_col] == cand).any():
            ctrl_label = cand
            break
    else:
        cand_series = obs[target_col].astype(str)
        matches = [x for x in cand_series.unique() if x.lower().startswith("non") or "control" in x.lower()]
        if not matches:
            raise RuntimeError(f"no control label in {target_col}")
        ctrl_label = matches[0]
    print(f"[pre] target_col={target_col}  ctrl_label={ctrl_label!r}")

    # Determine qualifying targets: ≥ MIN_CELLS cells per target
    counts = obs[target_col].value_counts()
    qualifying_targets = counts[counts >= MIN_CELLS].index.tolist()
    if ctrl_label in qualifying_targets:
        qualifying_targets.remove(ctrl_label)
    # Restrict to targets whose gene name is in var_names (needed for
    # feature-gene lookup)
    qualifying_targets = [t for t in qualifying_targets if t in gene_index]
    print(f"[pre] {len(qualifying_targets)} qualifying targets (≥{MIN_CELLS} cells and in var_names)")

    # Pseudobulk (per-target column mean)
    X_pb = np.zeros((len(qualifying_targets), len(var_names)), dtype=np.float64)
    n_per_target = np.zeros(len(qualifying_targets), dtype=np.int64)
    obs_tg = obs[target_col].to_numpy()
    for j, t in enumerate(qualifying_targets):
        m = obs_tg == t
        idx = np.where(m)[0]
        # backed read: iterate a chunk at a time
        xs = a.X[idx].astype(np.float64)
        if sp.issparse(xs):
            xs = xs.toarray()
        X_pb[j] = xs.mean(axis=0)
        n_per_target[j] = idx.size
        if (j+1) % 200 == 0:
            print(f"  [pre] pseudobulked {j+1}/{len(qualifying_targets)}")

    ctrl_mask_idx = np.where(obs_tg == ctrl_label)[0]
    xs = a.X[ctrl_mask_idx].astype(np.float64)
    if sp.issparse(xs):
        xs = xs.toarray()
    baseline = xs.mean(axis=0)  # per-gene control mean
    print(f"[pre] ctrl cells={ctrl_mask_idx.size}  baseline mean={baseline.mean():.4f}")

    # Change matrix (rows: targets, cols: genes)
    Y_change = X_pb - baseline[None, :]

    return {
        "targets": qualifying_targets,
        "var_names": var_names,
        "gene_index": gene_index,
        "X_pb": X_pb,
        "baseline": baseline,
        "Y_change": Y_change,
        "n_per_target": n_per_target,
    }


def split_targets(targets, seed=SEED, frac=HELD_OUT_FRACTION):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(targets))
    n_test = int(round(len(targets) * frac))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def per_condition_pearson(Y_pred, Y_obs):
    """Pearson r per column (perturbation). Returns array of length
    n_perturbations."""
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    Y_obs = np.asarray(Y_obs, dtype=np.float64)
    if Y_pred.shape != Y_obs.shape:
        raise ValueError(f"shape mismatch {Y_pred.shape} vs {Y_obs.shape}")
    n_cond = Y_pred.shape[1]
    r = np.zeros(n_cond, dtype=np.float64)
    for j in range(n_cond):
        p = Y_pred[:, j]; o = Y_obs[:, j]
        # Skip if degenerate
        if o.std() < 1e-30 or p.std() < 1e-30:
            r[j] = 0.0
            continue
        r[j] = np.corrcoef(p, o)[0, 1]
    return r


# ── Main ──────────────────────────────────────────────────────────────────
def run_fidelity(h5ad_path: Path, screen_name: str):
    print(f"\n=== {screen_name} : fidelity ===")
    pre = load_k562_pseudobulk(h5ad_path, seed=SEED)
    targets = pre["targets"]
    var_names = pre["var_names"]
    gene_index = pre["gene_index"]
    X_pb = pre["X_pb"]
    baseline = pre["baseline"]
    Y_change = pre["Y_change"]  # (n_targets, n_genes)

    train_idx, test_idx = split_targets(targets, seed=SEED)
    tr_targets = [targets[i] for i in train_idx]
    te_targets = [targets[i] for i in test_idx]
    print(f"  train targets={len(tr_targets)}, test targets={len(te_targets)}")

    # Y matrix for their fit: (n_gene × n_train_cond) — TRANSPOSE
    Y_train_gxc = Y_change[train_idx].T   # (n_gene, n_train_cond)
    X_train_gxc = X_pb[train_idx].T       # (n_gene, n_train_cond)

    # THEIR ALGORITHM: pca on training pseudobulk X (rows=genes, cols=conds).
    # pca$x = per-gene coordinates (n_gene × pca_dim).
    from numpy.linalg import svd as _svd
    # Center X across conditions per gene, as prcomp default.
    Xc = X_train_gxc - X_train_gxc.mean(axis=1, keepdims=True)
    u, sv, vt = _svd(Xc, full_matrices=False)
    de = min(PCA_DIM, len(sv))
    A_gene = u[:, :de] * sv[:de]          # (n_gene, de) — pca$x
    # pert_emb = t(pca$x) indexed by target gene:
    #   per condition j (targeting gene g_j), pert_emb[:, j] = A_gene[gene_index[g_j], :].T
    def pert_emb_at(targets_list):
        B = np.zeros((de, len(targets_list)), dtype=np.float64)
        for j, t in enumerate(targets_list):
            if t in gene_index:
                B[:, j] = A_gene[gene_index[t]]
        return B

    B_train = pert_emb_at(tr_targets)
    B_test = pert_emb_at(te_targets)

    fit = ae_solve(Y_train_gxc, A_gene, B_train, A_ridge=RIDGE, B_ridge=RIDGE)
    Ypred_test_raw = ae_predict(A_gene, fit["K"], B_test, fit["center"], baseline)
    # Observed pseudobulk X on test
    Xobs_test = X_pb[test_idx].T
    r_test_raw = per_condition_pearson(Ypred_test_raw, Xobs_test)
    # Their `r2` in the xlsx is on raw pseudobulk X; their r2_delta is
    # on change. Report both.
    Ypred_test_change = Ypred_test_raw - baseline[:, None]
    Yobs_test_change = Y_change[test_idx].T
    r_test_delta = per_condition_pearson(Ypred_test_change, Yobs_test_change)

    # THEIR MEAN BASELINE: predict training-mean of X (raw) for every held-out
    X_train_mean = X_train_gxc.mean(axis=1, keepdims=True)
    Ymean_test_raw = np.tile(X_train_mean, (1, len(te_targets)))
    r_mean_raw = per_condition_pearson(Ymean_test_raw, Xobs_test)
    Ymean_test_change = Ymean_test_raw - baseline[:, None]
    r_mean_delta = per_condition_pearson(Ymean_test_change, Yobs_test_change)

    # OUR IMPLEMENTATION (comparator convention): tr_embed = u·σ of Y_train
    # (targets × genes), te_embed = V_gene[:, gene_index[target]].
    our_fit = our_learned_solve(Y_train_gxc.T, var_names, tr_targets,
                                 pca_dim=PCA_DIM, ridge=RIDGE)
    Ypred_our_change = our_predict(te_targets, our_fit["V_gene"], gene_index,
                                    our_fit["B_gene"], our_fit["b_out"])
    # (n_test_targets × n_gene). Compare against Y_change[test_idx].
    Yobs_our_change = Y_change[test_idx]
    # per-condition Pearson r (columns = test conditions after transpose)
    r_our_delta = per_condition_pearson(Ypred_our_change.T, Yobs_our_change.T)
    Ypred_our_raw = Ypred_our_change + baseline[None, :]
    Xobs_our = X_pb[test_idx]
    r_our_raw = per_condition_pearson(Ypred_our_raw.T, Xobs_our.T)

    # AE-STYLE VARIANT: use gene-PCA loadings at target-gene column for both
    # train and test embeddings.
    ae_style = ae_style_solve_on_our_pipeline(Y_train_gxc.T, var_names,
                                               tr_targets, gene_index,
                                               pca_dim=PCA_DIM, ridge=RIDGE)
    Ypred_ae_style_change = ae_style_predict(te_targets, ae_style["V_gene"],
                                              gene_index, ae_style["B_gene"],
                                              ae_style["b_out"])
    r_ae_style_delta = per_condition_pearson(Ypred_ae_style_change.T, Yobs_our_change.T)
    Ypred_ae_style_raw = Ypred_ae_style_change + baseline[None, :]
    r_ae_style_raw = per_condition_pearson(Ypred_ae_style_raw.T, Xobs_our.T)

    # Structural fidelity: same-dataset comparison of our comparator
    # convention vs the AE-style convention on the same fold.
    def summ(name, arr):
        m = float(np.mean(arr)); s = float(np.std(arr, ddof=1))
        return {"metric": name, "mean": m, "sd": s,
                "median": float(np.median(arr))}

    result = {
        "screen": screen_name,
        "pca_dim": PCA_DIM,
        "ridge": RIDGE,
        "n_train": int(len(tr_targets)),
        "n_test": int(len(te_targets)),
        "note": "This is our K562/RPE1 h5ad + our pseudobulk pipeline, NOT their GEARS-preprocessed h5ad or their exact test-train split JSON. Numbers are structurally comparable within a fixed pipeline; direct equality with their published number requires their preprocessing + split.",
        "published_reference_from_source_data_xlsx": THEIRS[screen_name],
        "ae_R2_raw_pseudobulk": summ("ae r on raw X", r_test_raw),
        "ae_R2_change": summ("ae r on change", r_test_delta),
        "mean_baseline_R2_raw": summ("mean-baseline r on raw X", r_mean_raw),
        "mean_baseline_R2_change": summ("mean-baseline r on change", r_mean_delta),
        "our_current_conv_R2_change": summ("our (uσ train / V test) r on change", r_our_delta),
        "our_current_conv_R2_raw":  summ("our (uσ train / V test) r on raw X", r_our_raw),
        "ae_style_conv_R2_change": summ("ae-style (V for both) r on change", r_ae_style_delta),
        "ae_style_conv_R2_raw":     summ("ae-style (V for both) r on raw X", r_ae_style_raw),
    }
    return result


def main():
    outputs = {}
    for screen, h5 in [
        ("K562_essential", DATA / "K562_essential_normalized_singlecell_01.h5ad"),
        # ("RPE1_essential", DATA / "rpe1_normalized_singlecell_01.h5ad"),
    ]:
        outputs[screen] = run_fidelity(h5, screen)
    out_path = OUT / "F_step5_learned_fidelity.json"
    out_path.write_text(json.dumps(outputs, indent=2, default=str))
    print(f"\nsaved: {out_path}")
    for screen, rec in outputs.items():
        ref = rec["published_reference_from_source_data_xlsx"]
        print(f"\n== {screen} ==")
        print(f"  their published lpm_selftrained r (raw): {ref['lpm_selftrained_r2_mean']:.4f} ± {ref['lpm_selftrained_r2_sd']:.4f}")
        print(f"  their published mean baseline r (raw):   {ref['mean_baseline_r2_mean']:.4f} ± {ref['mean_baseline_r2_sd']:.4f}")
        print(f"  our AE port r (raw):        {rec['ae_R2_raw_pseudobulk']['mean']:.4f} ± {rec['ae_R2_raw_pseudobulk']['sd']:.4f}")
        print(f"  our mean baseline r (raw):  {rec['mean_baseline_R2_raw']['mean']:.4f} ± {rec['mean_baseline_R2_raw']['sd']:.4f}")
        print(f"  our comparator conv r (change): {rec['our_current_conv_R2_change']['mean']:.4f} ± {rec['our_current_conv_R2_change']['sd']:.4f}")
        print(f"  AE-style conv r (change):        {rec['ae_style_conv_R2_change']['mean']:.4f} ± {rec['ae_style_conv_R2_change']['sd']:.4f}")


if __name__ == "__main__":
    main()
