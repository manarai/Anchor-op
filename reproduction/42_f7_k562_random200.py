"""F7 K562 random-200 refit with the nested-CV ρ metric.

One run, then freeze. This is the reviewer's chosen sensitivity check
against the notebook's ``value_counts().head(200)`` selection bias:
resample 200 targets uniformly at random from the ≥ 60-cell-qualifying
pool, refit the entire pipeline (HVG force-keeping targets, PCA on
controls, ``measure_operator`` with aggregate_replicate_guides=True to
match the manuscript's K562 convention), and report the nested-CV
target-grouped-fold held-out ρ from ``reproduction/40_nested_cv_rho.py``
against a matched-SNR linear-truth control.

Also saves ``Σ_ctrl`` (the non-targeting-control covariance in
gene-space HVGs) to ``results/k562_random200_sigma_ctrl.npz`` so the
footprint-encoding follow-on can be run without a second h5ad load.

Runtime ~15–25 min on the 10 GB K562 h5ad.
Output: ``results/recheck/F7_K562_random200.json``,
``results/k562_random200_measurement.pkl``,
``results/k562_random200_sigma_ctrl.npz``.
"""
from __future__ import annotations
import json, pickle
from pathlib import Path

import numpy as np

import anchorop as ao

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260927
D = 30
RANK_TOL = 1e-2
N_HVG = 3000
MIN_CELLS_PER_TARGET = 60
MIN_CELLS_PER_GUIDE = 30
MIN_KD_EFF = 0.05
N_TARGETS_KEEP = 200

# Nested-CV recipe (matches reproduction/40_nested_cv_rho.py)
N_OUTER = 5
N_INNER = 3
N_REPS_LIN_SIM = 15
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]


# ----- nested-CV helpers (inline copy from script 40) -----
def _fit_A_at_rank(S, U, r):
    r = int(r)
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]), dtype=float)
    Us, sv, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(sv))
    sv_inv = np.zeros_like(sv); sv_inv[:r] = 1.0 / sv[:r]
    S_pinv = (Vt.T * sv_inv) @ Us.T
    return -U @ S_pinv


def _mse(A, S_te, U_te):
    r = A @ S_te + U_te
    return float(np.sum(r ** 2))


def _target_folds(target_of_g, k, seed):
    rng = np.random.default_rng(seed)
    targets = sorted(set(target_of_g.tolist()))
    rng.shuffle(targets)
    tgroups = np.array_split(np.array(targets, dtype=object), k)
    folds = []
    for i in range(k):
        te_t = set(tgroups[i].tolist())
        te_mask = np.array([t in te_t for t in target_of_g])
        folds.append((np.where(~te_mask)[0], np.where(te_mask)[0]))
    return folds


def _inner_pick(S_tr, U_tr, target_of_tr, seed):
    unique_targets = sorted(set(target_of_tr.tolist()))
    k = min(N_INNER, len(unique_targets))
    if k < 2:
        return 0
    rng = np.random.default_rng(seed)
    tgt_perm = np.array(unique_targets, dtype=object); rng.shuffle(tgt_perm)
    tgroups = np.array_split(tgt_perm, k)
    inner = []
    for i in range(k):
        te_t = set(tgroups[i].tolist())
        te_mask = np.array([t in te_t for t in target_of_tr])
        inner.append((np.where(~te_mask)[0], np.where(te_mask)[0]))
    best_r, best = None, np.inf
    for r in RANK_GRID:
        pooled = 0.0; ok = True
        for tr, val in inner:
            if r > 0 and len(tr) <= r:
                ok = False; break
            A = _fit_A_at_rank(S_tr[:, tr], U_tr[:, tr], r)
            pooled += _mse(A, S_tr[:, val], U_tr[:, val])
        if ok and pooled < best:
            best, best_r = pooled, r
    return best_r if best_r is not None else 0


def _nested_rho(S, U, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, picks = 0.0, 0.0, []
    for i, (tr, te) in enumerate(outer):
        r = _inner_pick(S[:, tr], U[:, tr], target_of_g[tr], seed + i * 17)
        if r > 0 and len(tr) <= r:
            r = 0
        A = _fit_A_at_rank(S[:, tr], U[:, tr], r)
        resid = A @ S[:, te] + U[:, te]
        num += float(np.sum(resid ** 2))
        den += float(np.sum(U[:, te] ** 2))
        picks.append(int(r))
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "picked_rank_median": int(np.median(picks)) if picks else None,
            "picked_rank_min": int(min(picks)) if picks else None,
            "picked_rank_max": int(max(picks)) if picks else None}


def main():
    import scanpy as sc
    from sklearn.decomposition import PCA
    path = DATA / "K562_essential_normalized_singlecell_01.h5ad"
    if not path.exists():
        raise SystemExit(f"h5ad not found: {path}")
    print(f"[1/6] loading {path} …")
    adata = ao.load_replogle_h5ad(str(path))
    target_counts = adata.obs.loc[adata.obs["target_gene"] != "", "target_gene"].value_counts()
    qualifying = target_counts[target_counts >= MIN_CELLS_PER_TARGET].index.tolist()
    rng = np.random.default_rng(SEED)
    keep_targets = list(rng.choice(np.array(qualifying), size=min(N_TARGETS_KEEP, len(qualifying)),
                                    replace=False))
    print(f"  qualifying targets: {len(qualifying)}; random-sampled {len(keep_targets)} for random-200")
    keep = adata.obs["target_gene"].isin(keep_targets) | (adata.obs["target_gene"] == "")
    a = adata[keep].copy()
    a.obs.loc[a.obs["target_gene"] != "", "guide"] = "guide_" + a.obs.loc[a.obs["target_gene"] != "", "target_gene"].astype(str)

    print("[2/6] finite-gene filter …")
    sample = np.asarray(a.X[:min(20000, a.n_obs), :].toarray() if hasattr(a.X, "toarray") else a.X[:min(20000, a.n_obs), :])
    finite = np.isfinite(sample).all(axis=0)
    if not finite.all():
        a = a[:, finite].copy()

    print("[3/6] HVG (variance-based, matches notebook 01b pre-scaled path; force-keeping targets) …")
    # Replogle 'normalized_singlecell' is pre-scaled (z-score / Pearson residuals);
    # variance-based HVG is what the notebook uses. Stream-chunked to bound RAM.
    X = a.X
    n_chunks = 20
    chunk = X.shape[0] // n_chunks + 1
    sums = np.zeros(X.shape[1], dtype=np.float64)
    sums_sq = np.zeros(X.shape[1], dtype=np.float64)
    counts = np.zeros(X.shape[1], dtype=np.float64)
    for k in range(n_chunks):
        block = X[k * chunk:(k + 1) * chunk, :]
        block = np.asarray(block.toarray() if hasattr(block, "toarray") else block)
        fin = np.isfinite(block)
        block_c = np.where(fin, block, 0.0)
        sums += block_c.sum(axis=0)
        sums_sq += (block_c ** 2).sum(axis=0)
        counts += fin.sum(axis=0)
    counts = np.maximum(counts, 1.0)
    gene_var = np.where(np.isfinite(sums), sums_sq / counts - (sums / counts) ** 2, 0.0).ravel()
    top_idx = np.argsort(gene_var)[::-1][:N_HVG]
    hvg_mask = np.zeros(X.shape[1], dtype=bool); hvg_mask[top_idx] = True
    force = np.asarray(a.var_names.isin(set(keep_targets)))
    a.var["highly_variable"] = hvg_mask | force
    a = a[:, a.var["highly_variable"]].copy()
    print(f"  kept {a.shape[1]} HVGs (of which {int(force.sum())} force-kept targets from the selection)")

    print("[4/6] PCA on controls, save Σ_ctrl …")
    ctrl_mask = (a.obs["target_gene"] == "").to_numpy()
    X_ctrl = np.asarray(a[ctrl_mask].X.toarray() if hasattr(a.X, "toarray") else a[ctrl_mask].X, dtype=np.float32)
    pca = PCA(n_components=D, random_state=SEED).fit(X_ctrl)
    loadings = pca.components_.T.astype(np.float32)
    basis = ao.make_program_basis(loadings, a.var_names, method="pca_external",
                                  control_count=int(ctrl_mask.sum()), normalize=False)
    # Σ_ctrl for the footprint-encoding follow-on (u_g ∝ W^T Σ_ctrl δ_g).
    sigma_ctrl = np.cov(X_ctrl, rowvar=False).astype(np.float32)
    np.savez_compressed(RESULTS / "k562_random200_sigma_ctrl.npz",
                        sigma_ctrl=sigma_ctrl, gene_symbols=np.array(a.var_names))
    print(f"  saved Σ_ctrl {sigma_ctrl.shape} to results/k562_random200_sigma_ctrl.npz")

    print("[5/6] build_guide_responses + measure_from_sensitivity (proxy-efficiency path) …")
    # Replogle pre-scaled residual data — current measure_operator refuses the
    # detection_rate proxy path (calibrated=False guard), so we call the
    # underlying helpers directly. This matches the encoding used in the
    # tracked pkls (examples/01b/01c were run against an older API that
    # allowed the proxy path in measure_operator).
    responses, dropped = ao.build_guide_responses(
        a, basis,
        guide_key="guide", control_label="non-targeting", target_key="target_gene",
        min_cells_per_guide=MIN_CELLS_PER_GUIDE,
        min_knockdown_efficiency=MIN_KD_EFF,
        efficiency_estimator="auto",
        allow_proxy_efficiency=True,
    )
    print(f"  responses: {len(responses)} retained, {len(dropped)} dropped")
    S_cols = [r.response for r in responses]
    U_cols = [r.input_vector for r in responses]
    guide_names = [r.guide for r in responses]
    guide_effs = {r.guide: r.efficiency for r in responses}
    guide_tgts = {r.guide: r.target for r in responses if r.target is not None}
    S = np.stack(S_cols, axis=1)
    U = np.stack(U_cols, axis=1)
    meas = ao.measure_from_sensitivity(
        S=S, U=U,
        guide_names=guide_names, guide_efficiencies=guide_effs, guide_targets=guide_tgts,
        reg="tsvd", reg_param="path", rank_tol=RANK_TOL, state_label="K562_random200",
    )
    with (RESULTS / "k562_random200_measurement.pkl").open("wb") as f:
        pickle.dump({"measurement": meas, "basis": basis, "keep_targets": keep_targets,
                     "provenance": {"seed": SEED, "d": D, "rank_tol": RANK_TOL,
                                    "n_hvg": N_HVG, "n_targets_keep": N_TARGETS_KEEP,
                                    "selection": "random_200"}}, f)
    print(f"  saved measurement pkl ({len(meas.report.retained_guides)} guides retained)")

    print("[6/6] nested-CV ρ + matched-SNR linear truth …")
    guides = list(meas.report.retained_guides)
    target_of_g = np.array([g[len("guide_"):] if g.startswith("guide_") else g for g in guides])
    S, U = meas.S, meas.U
    # α̂ from median column norms on this refit.
    rng0 = np.random.default_rng(SEED)
    J_ref0 = rng0.normal(size=(D, D)) / np.sqrt(D) - 1.5 * np.eye(D)
    S_ref = -np.linalg.solve(J_ref0, U)
    alpha_hat = float(np.median(np.linalg.norm(S, axis=0)) / max(np.median(np.linalg.norm(S_ref, axis=0)), 1e-30))
    # σ: use K562 anchor 0.240 for the linear-truth control (matches nested-CV recipe on the published measurement).
    sigma = 0.240

    real = _nested_rho(S, U, target_of_g, SEED)
    rhos = []; picks = []
    for r in range(N_REPS_LIN_SIM):
        rng = np.random.default_rng(SEED + r + 9000)
        J_ref = rng.normal(size=(D, D)) / np.sqrt(D) - 1.5 * np.eye(D)
        J = J_ref / max(alpha_hat, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_obs = S_true + sigma * rng.normal(size=S_true.shape)
        out = _nested_rho(S_obs, U, target_of_g, SEED + r + 9000)
        rhos.append(out["rho_pooled"]); picks.append(out["picked_rank_median"])

    payload = {
        "seed": SEED, "d": D, "rank_tol": RANK_TOL,
        "selection": "random_200",
        "n_targets_kept": len(keep_targets),
        "n_guides_retained": int(len(meas.report.retained_guides)),
        "input_subspace_dim": int(meas.report.input_subspace_dim),
        "effective_response_rank": int(meas.report.effective_response_rank),
        "full_domain_identified": bool(meas.report.full_domain_identified),
        "sigma_used_for_lin_sim": sigma,
        "alpha_hat_S": alpha_hat,
        "real_nested_cv": real,
        "linear_truth_matched_alpha_nested_cv": {
            "n_reps": N_REPS_LIN_SIM,
            "rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos)),
            "picked_rank_median_across_reps": int(np.median(picks)),
        },
        "sigma_ctrl_saved_to": "results/k562_random200_sigma_ctrl.npz",
    }
    print(json.dumps(payload, indent=2))
    (OUT / "F7_K562_random200.json").write_text(json.dumps(payload, indent=2))
    print("saved:", OUT / "F7_K562_random200.json")


if __name__ == "__main__":
    main()
