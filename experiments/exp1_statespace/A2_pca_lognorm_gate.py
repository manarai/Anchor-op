"""A2 second sanity gate — PCA-lognorm nested CV (PREREG amendment 2).

Same target-grouped 5-outer × 3-inner nested-CV recipe as A1
(``A1_reproduction_gate.py``), rank grid including 0, seed 20260930,
15 matched-linear-truth replicates — but on **log1p-normalised raw
counts**, not on the preprint's residual h5ad.

Pipeline:
1. Load `examples/data/K562_essential_raw_singlecell_01.h5ad`.
2. Match cells / guides / targets with the measurement bundle used
   in A1 (identical set per PREREG amendment 2).
3. `sc.pp.normalize_total(target_sum=1e4)` + `sc.pp.log1p` on all
   retained cells.
4. Fit a `PCARep(dim=30)` on NT control cells only.
5. For each retained guide, build `U_g` as the log1p-normalised
   NT-control mean projected to d=30 minus the perturbed-cell mean
   projected to d=30 (the paper-1 pseudobulk convention adapted to
   log1p inputs; this is NOT the paper-1 PCA-on-residuals basis).
6. Build `S` as the per-guide-mean projected d=30 shift vector.
7. Target-grouped nested-CV on (S, U) with the same recipe as A1.
8. 15 matched-linear-truth replicates at the arm's own σ and α_S.
9. Report real nested-CV ρ and matched-linear ρ. The user reviews
   this against A1 (ρ real 0.9658, matched-linear 0.1818 on the
   residual-space PCA-30 basis) before any scGPT arm runs.

Output: `experiments/exp1_statespace/A2_pca_lognorm_gate.json`.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = Path(__file__).parent / "A2_pca_lognorm_gate.json"
RAW_H5AD = REPO / "examples" / "data" / "K562_essential_raw_singlecell_01.h5ad"
PKL = REPO / "results" / "k562_essential_measurement.pkl"

TARGET_SUM = 1e4
SEED = 20260930
N_OUTER = 5
N_INNER = 3
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
N_LIN_SIM = 15


# ─── nested-CV plumbing (shared with A1_reproduction_gate.py) ─────────
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


def _inv_rho(A, S_te, U_te):
    num = np.sum((A @ S_te + U_te) ** 2)
    den = np.sum(U_te ** 2)
    return float(np.sqrt(num / max(den, 1e-30)))


def _nested_rho(S, U, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    picks, rhos, num, den = [], [], 0.0, 0.0
    for i, (tr, te) in enumerate(outer):
        inner_targets = target_of_g[tr]
        inner = _target_folds(inner_targets,
                              min(N_INNER, len(set(inner_targets))),
                              seed + i * 17)
        best_r, best_mse = None, np.inf
        for r in RANK_GRID:
            pooled = 0.0
            for tr_in, val_in in inner:
                gtr = tr[tr_in]; gval = tr[val_in]
                A = _tsvd_A(S[:, gtr], U[:, gtr], r)
                pooled += float(np.sum((A @ S[:, gval] + U[:, gval]) ** 2))
            if pooled < best_mse:
                best_mse, best_r = pooled, r
        picks.append(best_r)
        A = _tsvd_A(S[:, tr], U[:, tr], best_r)
        num += float(np.sum((A @ S[:, te] + U[:, te]) ** 2))
        den += float(np.sum(U[:, te] ** 2))
        rhos.append(_inv_rho(A, S[:, te], U[:, te]))
    return {
        "rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
        "per_fold_rho": rhos,
        "per_fold_sd": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
        "picked_ranks": picks,
    }


def _matched_linear_truth(U, target_of_g, sigma, alpha_S, seed,
                           n_reps=N_LIN_SIM):
    rhos = []
    d = U.shape[0]
    for r in range(n_reps):
        rng = np.random.default_rng(seed + r + 8000)
        G = rng.normal(size=(d, d)) / np.sqrt(d)
        J_ref = G - 1.5 * np.eye(d)
        J = J_ref / max(alpha_S, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_sim = S_true + sigma * rng.normal(size=S_true.shape)
        out = _nested_rho(S_sim, U, target_of_g, seed=seed + r + 8000)
        rhos.append(out["rho_pooled"])
    return {
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos, ddof=1)),
        "n_reps": n_reps,
    }


def main():
    import anndata as ad
    import scipy.sparse as sp

    from anchorop.state_space import PCARep

    if not RAW_H5AD.exists():
        print(f"ERROR: raw-counts h5ad missing at {RAW_H5AD}.")
        sys.exit(1)
    print(f"[A2] loading {RAW_H5AD.name} …")
    a = ad.read_h5ad(RAW_H5AD, backed="r")
    obs = a.obs
    var_names = list(a.var_names)

    bundle = pickle.load(open(PKL, "rb"))
    retained_guides = list(bundle["measurement"].report.retained_guides)
    stored = getattr(bundle["measurement"].report, "guide_targets", None)
    if stored:
        clean = {str(k): str(v) for k, v in dict(stored).items()}
        target_of_guide = {str(g): clean[str(g)] for g in retained_guides}
    else:
        target_of_guide = {
            str(g): (str(g)[6:] if str(g).startswith("guide_") else str(g).split("_")[0])
            for g in retained_guides
        }
    unique_targets = sorted(set(target_of_guide.values()))

    # Non-targeting controls.
    ctrl_mask = (obs["gene_id"] == "non-targeting").to_numpy()
    ctrl_idx = np.where(ctrl_mask)[0]
    print(f"[A2] NT control cells: {ctrl_idx.size}")

    print(f"[A2] loading raw counts for NT controls into memory …")
    X_raw_ctrl = a.X[ctrl_idx]
    if sp.issparse(X_raw_ctrl):
        X_raw_ctrl = X_raw_ctrl.toarray()
    X_raw_ctrl = np.asarray(X_raw_ctrl, dtype=np.float64)
    sums = X_raw_ctrl.sum(axis=1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    X_log_ctrl = np.log1p(X_raw_ctrl * (TARGET_SUM / sums))

    print(f"[A2] fitting PCARep(30) on log1p NT controls …")
    rep = PCARep(dim=30).fit(X_log_ctrl)
    print(f"[A2] PCA fitted; mean(Z_ctrl)=",
          f"{float(np.mean(rep.encode(X_log_ctrl))):.4e}")

    # Per-target perturbed cells. Build S and U per guide (one per target
    # on the Replogle essential library after the released aggregation).
    gt_arr = obs["gene_id"].to_numpy()
    print(f"[A2] building S and U for {len(unique_targets)} targets …")
    S_cols, U_cols, target_list = [], [], []
    for t in unique_targets:
        pert_idx = np.where(gt_arr == t)[0]
        if pert_idx.size < 60:
            continue
        X_pert = a.X[pert_idx]
        if sp.issparse(X_pert):
            X_pert = X_pert.toarray()
        X_pert = np.asarray(X_pert, dtype=np.float64)
        s_pert = X_pert.sum(axis=1, keepdims=True)
        s_pert = np.where(s_pert == 0, 1.0, s_pert)
        X_log_pert = np.log1p(X_pert * (TARGET_SUM / s_pert))
        z_pert = rep.encode(X_log_pert).mean(axis=0)   # (d,)
        z_ctrl = rep.encode(X_log_ctrl).mean(axis=0)   # (d,) — same for every target
        s_vec = z_pert - z_ctrl                        # measured shift in state space
        # Encoded perturbation input via knockdown-scale FD at κ = 0.7
        # (primary) under the log1p convention.
        g_idx = var_names.index(t) if t in var_names else -1
        if g_idx < 0:
            continue
        u_vec = rep.knockdown_scale_difference(
            X_log_ctrl, g_idx, kappa=0.7, input_space="log1p")
        S_cols.append(s_vec); U_cols.append(u_vec); target_list.append(t)
        if len(target_list) % 50 == 0:
            print(f"[A2]   built {len(target_list)} targets …")
    S = np.column_stack(S_cols)   # (d, n_targets)
    U = np.column_stack(U_cols)   # (d, n_targets)
    target_of_g = np.array(target_list)
    print(f"[A2] final S shape {S.shape}, U shape {U.shape}, "
          f"n_targets={len(target_list)}")

    # σ from the paper-1 anchor (K562 essential = 0.240) is on the
    # residual basis; on the log1p PCA basis we re-anchor by the
    # observed median column norm of S and the matched α_S convention.
    # For the gate we reuse α_S = 369 and σ = 0.240 for comparability
    # with A1; a per-arm reanchoring will happen in Task 4 proper.
    SIGMA = 0.240
    ALPHA_S = 369.0

    print(f"[A2] real target-held-out nested-CV ρ …")
    real = _nested_rho(S, U, target_of_g, SEED)
    print(f"      ρ = {real['rho_pooled']:.4f} ± {real['per_fold_sd']:.4f}  "
          f"picked ranks = {real['picked_ranks']}")

    print(f"[A2] matched linear-truth ρ at α_S = {ALPHA_S}, σ = {SIGMA} "
          f"({N_LIN_SIM} replicates) …")
    lin = _matched_linear_truth(U, target_of_g, SIGMA, ALPHA_S, SEED)
    print(f"      ρ = {lin['rho_mean']:.4f} ± {lin['rho_std']:.4f} "
          f"(N = {lin['n_reps']})")

    report = {
        "seed": SEED,
        "raw_h5ad": str(RAW_H5AD.relative_to(REPO)),
        "normalisation": "sc.pp.normalize_total(target_sum=1e4) + sc.pp.log1p (reimplemented inline)",
        "sigma_used": SIGMA,
        "alpha_S_used": ALPHA_S,
        "n_targets": int(S.shape[1]),
        "n_outer": N_OUTER, "n_inner": N_INNER,
        "rank_grid": RANK_GRID,
        "real": real,
        "matched_linear_truth": lin,
        "a1_reference_real_rho": 0.9658,
        "a1_reference_matched_linear_rho": 0.1818,
        "comparison_note": (
            "A1 ran on the preprint's residual-space PCA-30 basis; "
            "A2 runs on a log1p-space PCA-30 basis. These are NOT the "
            "same encoder; a close match would say the operator-level "
            "picture is invariant to the basis, a mismatch would be "
            "reported and discussed before any scGPT comparison."),
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[A2] saved: {OUT}")


if __name__ == "__main__":
    main()
