"""A2b — re-anchor σ and α_S for the log1p PCA-30 basis.

Follows the paper-1 methodology (MANUSCRIPT.md §4.4 noise anchor and
§4.6 matched-SNR control) but applied to the log1p PCA-30 basis
instead of the residual basis.

- σ_log1p: within-guide cell-level split-half bootstrap. For each
  target with ≥ 20 cells on the log1p-normalised K562 raw-counts
  h5ad, split cells into equal halves, compute half-Δz vectors in
  the log1p PCA-30 basis, and take the median per-entry standard
  deviation `‖d₁ − d₂‖_F / (2√d)` across targets.
- α_S_log1p: scaling such that median column norm of
  `S_true = −J_ref·α_S⁻¹·U` on the log1p basis matches the observed
  median column norm of `S` on the same basis.
- Rerun the matched-linear-truth part of A2 with the re-anchored
  σ and α_S.

Output: `A2b_reanchored_sigma_alpha.json`.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = Path(__file__).parent / "A2b_reanchored_sigma_alpha.json"
RAW_H5AD = REPO / "examples" / "data" / "K562_essential_raw_singlecell_01.h5ad"
PKL = REPO / "results" / "k562_essential_measurement.pkl"

TARGET_SUM = 1e4
SEED = 20260930
N_OUTER = 5
N_INNER = 3
N_LIN_SIM = 15
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
MIN_CELLS_FOR_BOOTSTRAP = 20


def _target_folds(target_of_g, k, seed):
    rng = np.random.default_rng(seed)
    targets = sorted(set(target_of_g.tolist()))
    rng.shuffle(targets)
    tg = np.array_split(np.array(targets, dtype=object), k)
    out = []
    for i in range(k):
        te = set(tg[i].tolist())
        m = np.array([t in te for t in target_of_g])
        out.append((np.where(~m)[0], np.where(m)[0]))
    return out


def _tsvd_A(S, U, r):
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]))
    Us, s, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(s))
    S_pinv = Vt[:r].T @ np.diag(1.0 / s[:r]) @ Us[:, :r].T
    return -U @ S_pinv


def _nested_rho(S, U, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    rhos, num, den = [], 0.0, 0.0
    for i, (tr, te) in enumerate(outer):
        inner_targets = target_of_g[tr]
        inner = _target_folds(inner_targets, min(N_INNER, len(set(inner_targets))), seed + i * 17)
        best_r, best_mse = None, np.inf
        for r in RANK_GRID:
            pooled = 0.0
            for tr_in, val_in in inner:
                gtr = tr[tr_in]; gval = tr[val_in]
                A = _tsvd_A(S[:, gtr], U[:, gtr], r)
                pooled += float(np.sum((A @ S[:, gval] + U[:, gval]) ** 2))
            if pooled < best_mse:
                best_mse, best_r = pooled, r
        A = _tsvd_A(S[:, tr], U[:, tr], best_r)
        num += float(np.sum((A @ S[:, te] + U[:, te]) ** 2))
        den += float(np.sum(U[:, te] ** 2))
        rhos.append(float(np.sqrt(np.sum((A @ S[:, te] + U[:, te]) ** 2) / max(np.sum(U[:, te] ** 2), 1e-30))))
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_rho": rhos,
            "per_fold_sd": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0}


def _matched_linear_truth(U, target_of_g, sigma, alpha_S, seed, n_reps=N_LIN_SIM):
    rhos = []
    d = U.shape[0]
    for r in range(n_reps):
        rng = np.random.default_rng(seed + r + 8000)
        G = rng.normal(size=(d, d)) / np.sqrt(d)
        J_ref = G - 1.5 * np.eye(d)
        J = J_ref / max(alpha_S, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_sim = S_true + sigma * rng.normal(size=S_true.shape)
        rhos.append(_nested_rho(S_sim, U, target_of_g, seed=seed + r + 8000)["rho_pooled"])
    return {"rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos, ddof=1)),
            "n_reps": n_reps}


def main():
    import anndata as ad
    import scipy.sparse as sp
    from anchorop.state_space import PCARep

    print(f"[A2b] loading {RAW_H5AD.name} …")
    a = ad.read_h5ad(RAW_H5AD, backed="r")
    obs = a.obs
    var_names = list(a.var_names)
    gene_index = {g: i for i, g in enumerate(var_names)}

    bundle = pickle.load(open(PKL, "rb"))
    retained_guides = list(bundle["measurement"].report.retained_guides)
    stored = getattr(bundle["measurement"].report, "guide_targets", None)
    if stored:
        clean = {str(k): str(v) for k, v in dict(stored).items()}
        targets = sorted({clean[str(g)] for g in retained_guides})
    else:
        targets = sorted({(str(g)[6:] if str(g).startswith("guide_") else str(g).split("_")[0])
                          for g in retained_guides})

    def _lognorm(X):
        s = X.sum(axis=1, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        return np.log1p(X * (TARGET_SUM / s))

    ctrl_mask = (obs["gene_id"] == "non-targeting").to_numpy()
    ctrl_idx = np.where(ctrl_mask)[0]
    print(f"[A2b] NT control cells: {ctrl_idx.size}; loading + lognorm …")
    Xc = a.X[ctrl_idx]
    if sp.issparse(Xc): Xc = Xc.toarray()
    Xc = _lognorm(np.asarray(Xc, dtype=np.float64))

    rep = PCARep(dim=30).fit(Xc)
    d = rep.dim

    # ── σ anchor via within-guide split-half (paper §4.4) ──────────────
    print(f"[A2b] σ anchor: within-guide split-half on log1p PCA-30 basis …")
    rng = np.random.default_rng(SEED)
    gt_arr = obs["gene_id"].to_numpy()
    per_target_sigma = []
    for t in targets:
        pert_idx = np.where(gt_arr == t)[0]
        if pert_idx.size < MIN_CELLS_FOR_BOOTSTRAP:
            continue
        perm = rng.permutation(pert_idx)
        half = pert_idx.size // 2
        a_idx = np.sort(perm[:half])         # backed h5py needs sorted indexing
        b_idx = np.sort(perm[half:2*half])
        Xa = a.X[a_idx]; Xb = a.X[b_idx]
        if sp.issparse(Xa): Xa = Xa.toarray()
        if sp.issparse(Xb): Xb = Xb.toarray()
        Xa = _lognorm(np.asarray(Xa, dtype=np.float64))
        Xb = _lognorm(np.asarray(Xb, dtype=np.float64))
        # per-cell encode, then take the mean per split
        z_a = rep.encode(Xa).mean(axis=0)
        z_b = rep.encode(Xb).mean(axis=0)
        # per-entry SD of the half-shift: ‖d₁ − d₂‖ / (2√d) is the paper's formula
        sigma_t = float(np.linalg.norm(z_a - z_b) / (2 * np.sqrt(d)))
        per_target_sigma.append(sigma_t)
    per_target_sigma = np.array(per_target_sigma)
    sigma_log1p = float(np.median(per_target_sigma))
    print(f"[A2b]   σ_log1p (median across {len(per_target_sigma)} targets) = {sigma_log1p:.4f}")

    # ── Build S, U on log1p basis exactly as A2 did ────────────────────
    print(f"[A2b] building S, U on log1p PCA-30 basis …")
    S_cols, U_cols, target_list = [], [], []
    z_ctrl_mean = rep.encode(Xc).mean(axis=0)
    for t in targets:
        pert_idx = np.where(gt_arr == t)[0]
        if pert_idx.size < 60:
            continue
        if t not in gene_index:
            continue
        g_idx = gene_index[t]
        Xp = a.X[pert_idx]
        if sp.issparse(Xp): Xp = Xp.toarray()
        Xp = _lognorm(np.asarray(Xp, dtype=np.float64))
        z_pert = rep.encode(Xp).mean(axis=0)
        s_vec = z_pert - z_ctrl_mean
        u_vec = rep.knockdown_scale_difference(Xc, g_idx, kappa=0.7, input_space="log1p")
        S_cols.append(s_vec); U_cols.append(u_vec); target_list.append(t)
    S = np.column_stack(S_cols)
    U = np.column_stack(U_cols)
    target_of_g = np.array(target_list)
    print(f"[A2b]   S shape {S.shape}, U shape {U.shape}, n_targets {len(target_list)}")

    # ── α_S anchor so median column norm of S_true matches observed ────
    print(f"[A2b] α_S anchor: match median column norm of S_true to observed on log1p basis …")
    median_S_norm = float(np.median(np.linalg.norm(S, axis=0)))
    # S_true = −J⁻¹·U with J = J_ref / α_S, so S_true = −α_S · J_ref⁻¹·U.
    # Median column norm scales linearly with α_S. Compute the baseline
    # (α_S = 1) column-norm median, then set α_S = median_S_norm /
    # median_baseline. Average over 10 draws of J_ref.
    rng = np.random.default_rng(SEED + 1)
    base_norms = []
    for r in range(10):
        G = rng.normal(size=(d, d)) / np.sqrt(d)
        J_ref = G - 1.5 * np.eye(d)
        S_true_unit = -np.linalg.solve(J_ref, U)
        base_norms.append(np.median(np.linalg.norm(S_true_unit, axis=0)))
    median_baseline = float(np.median(base_norms))
    alpha_S_log1p = median_S_norm / max(median_baseline, 1e-30)
    print(f"[A2b]   median_S_norm = {median_S_norm:.4f}")
    print(f"[A2b]   median_baseline (α_S = 1) = {median_baseline:.4f}")
    print(f"[A2b]   α_S_log1p = {alpha_S_log1p:.2f}")

    # ── Rerun real + matched-linear with re-anchored σ, α_S ────────────
    print(f"[A2b] real nested-CV ρ (unchanged from A2) …")
    real = _nested_rho(S, U, target_of_g, SEED)
    print(f"[A2b]   ρ = {real['rho_pooled']:.4f} ± {real['per_fold_sd']:.4f}")

    print(f"[A2b] matched linear-truth with re-anchored σ = {sigma_log1p:.4f}, "
          f"α_S = {alpha_S_log1p:.2f} …")
    lin = _matched_linear_truth(U, target_of_g, sigma_log1p, alpha_S_log1p, SEED)
    print(f"[A2b]   ρ = {lin['rho_mean']:.4f} ± {lin['rho_std']:.4f}")

    report = {
        "seed": SEED,
        "sigma_residual_reference_from_paper1": 0.240,
        "alpha_S_residual_reference_from_paper1": 369.0,
        "sigma_log1p": sigma_log1p,
        "sigma_log1p_per_target_summary": {
            "n_targets": int(per_target_sigma.size),
            "p05": float(np.percentile(per_target_sigma, 5)),
            "median": float(np.median(per_target_sigma)),
            "p95": float(np.percentile(per_target_sigma, 95)),
        },
        "alpha_S_log1p": alpha_S_log1p,
        "median_S_column_norm_log1p": median_S_norm,
        "alpha_S_baseline_log1p": median_baseline,
        "n_targets": int(S.shape[1]),
        "real": real,
        "matched_linear_truth_reanchored": lin,
        "a1_reference": {"real_rho": 0.9658, "matched_linear_rho": 0.1818,
                           "sigma": 0.240, "alpha_S": 369.0},
        "a2_reference_wrong_anchors": {"real_rho": 0.9208, "matched_linear_rho": 0.1320,
                                        "sigma": 0.240, "alpha_S": 369.0},
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[A2b] saved: {OUT}")


if __name__ == "__main__":
    main()
