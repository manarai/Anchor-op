"""Step 3: random-panel distribution on all K562 essential qualifying targets.

Preregistered 2026-09-28 (commit 08560c3).

Recipe:
  1. Fit anchor-op once on all ~1,740 K562 targets with ≥ 60 cells (same
     filters as the notebooks). Save Σ_ctrl and the full-panel measurement.
  2. Subsample 50 random 200-target panels from that single measurement
     (fixed basis W, fixed κ). Per panel:
       - real nested-CV ρ under Table 1's recipe;
       - matched-linear-truth ρ (N = 5 sims per panel).
  3. Report the distribution (median, 5th–95th percentile) for both.
  4. Report the fraction of panels where real ρ < 0.95.

Output: results/recheck/F_step3_random_panels.json and
  results/k562_all_targets_measurement.pkl (regenerable).
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

import anchorop as ao

SEED = 20260928
D = 30
RANK_TOL = 1e-2
N_HVG = 3000
MIN_CELLS_PER_TARGET = 60
MIN_CELLS_PER_GUIDE = 30
MIN_KD_EFF = 0.05
N_PANELS = 50
PANEL_SIZE = 200
N_LIN_SIM_PER_PANEL = 5
N_OUTER = 5
N_INNER = 3
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]

SIGMA_K562 = 0.240
ALPHA_S_HINT = 369.0  # used only as a starting reference; α_hat re-derived per panel


# ─── nested-CV helpers ────────────────────────────────────────────────────
def _target_folds(target_of_g, k, seed):
    rng = np.random.default_rng(seed)
    ts = sorted(set(target_of_g.tolist()))
    rng.shuffle(ts)
    tg = np.array_split(np.array(ts, dtype=object), k)
    out = []
    for i in range(k):
        te = set(tg[i].tolist())
        m = np.array([t in te for t in target_of_g])
        out.append((np.where(~m)[0], np.where(m)[0]))
    return out


def _fit_A_at_rank(S, U, r):
    r = int(r)
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]))
    Us, sv, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(sv))
    inv = np.zeros_like(sv); inv[:r] = 1.0 / sv[:r]
    return -U @ ((Vt.T * inv) @ Us.T)


def _inner_pick(S_tr, U_tr, target_of_tr, seed):
    ts = sorted(set(target_of_tr.tolist()))
    k = min(N_INNER, len(ts))
    if k < 2: return 0
    rng = np.random.default_rng(seed)
    tp = np.array(ts, dtype=object); rng.shuffle(tp)
    tg = np.array_split(tp, k)
    inner = []
    for i in range(k):
        te = set(tg[i].tolist())
        m = np.array([t in te for t in target_of_tr])
        inner.append((np.where(~m)[0], np.where(m)[0]))
    best_r, best = None, np.inf
    for r in RANK_GRID:
        pooled, ok = 0.0, True
        for tr_in, val in inner:
            if r > 0 and len(tr_in) <= r: ok = False; break
            A = _fit_A_at_rank(S_tr[:, tr_in], U_tr[:, tr_in], r)
            pooled += float(np.sum((A @ S_tr[:, val] + U_tr[:, val]) ** 2))
        if ok and pooled < best: best, best_r = pooled, r
    return best_r if best_r is not None else 0


def _nested_rho(S, U, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den = 0.0, 0.0
    for i, (tr, te) in enumerate(outer):
        r = _inner_pick(S[:, tr], U[:, tr], target_of_g[tr], seed + i * 17)
        if r > 0 and len(tr) <= r: r = 0
        A = _fit_A_at_rank(S[:, tr], U[:, tr], r)
        resid = A @ S[:, te] + U[:, te]
        num += float(np.sum(resid ** 2))
        den += float(np.sum(U[:, te] ** 2))
    return float(np.sqrt(num / max(den, 1e-30)))


def _fit_all_targets_k562(force=False):
    """Refit anchor-op on all K562 essential targets with ≥ 60 cells."""
    out_pkl = RESULTS / "k562_all_targets_measurement.pkl"
    if out_pkl.exists() and not force:
        print(f"  loading cached full-panel measurement from {out_pkl}")
        with out_pkl.open("rb") as f:
            return pickle.load(f)
    import scanpy as sc
    from sklearn.decomposition import PCA
    h5ad = DATA / "K562_essential_normalized_singlecell_01.h5ad"
    print(f"  reloading {h5ad} …")
    adata = ao.load_replogle_h5ad(str(h5ad))
    counts = adata.obs.loc[adata.obs["target_gene"] != "", "target_gene"].value_counts()
    qualifying = counts[counts >= MIN_CELLS_PER_TARGET].index.tolist()
    print(f"  {len(qualifying)} qualifying targets (≥ {MIN_CELLS_PER_TARGET} cells)")
    keep = adata.obs["target_gene"].isin(qualifying) | (adata.obs["target_gene"] == "")
    a = adata[keep].copy()
    a.obs.loc[a.obs["target_gene"] != "", "guide"] = "guide_" + a.obs.loc[a.obs["target_gene"] != "", "target_gene"].astype(str)
    # finite-gene filter
    sample = np.asarray(a.X[:min(20000, a.n_obs), :].toarray() if hasattr(a.X, "toarray") else a.X[:min(20000, a.n_obs), :])
    finite = np.isfinite(sample).all(axis=0)
    if not finite.all():
        a = a[:, finite].copy()
    # variance-based HVG with target force-keep (Replogle pre-scaled path)
    X = a.X
    n_chunks = 20
    chunk = X.shape[0] // n_chunks + 1
    sums = np.zeros(X.shape[1], dtype=np.float64)
    sumsq = np.zeros(X.shape[1], dtype=np.float64)
    counts_c = np.zeros(X.shape[1], dtype=np.float64)
    for k in range(n_chunks):
        block = X[k * chunk:(k + 1) * chunk, :]
        block = np.asarray(block.toarray() if hasattr(block, "toarray") else block)
        fin = np.isfinite(block); block_c = np.where(fin, block, 0.0)
        sums += block_c.sum(axis=0); sumsq += (block_c ** 2).sum(axis=0); counts_c += fin.sum(axis=0)
    counts_c = np.maximum(counts_c, 1.0)
    gene_var = np.where(np.isfinite(sums), sumsq / counts_c - (sums / counts_c) ** 2, 0.0).ravel()
    top_idx = np.argsort(gene_var)[::-1][:N_HVG]
    hvg = np.zeros(X.shape[1], dtype=bool); hvg[top_idx] = True
    force = np.asarray(a.var_names.isin(set(qualifying)))
    a.var["highly_variable"] = hvg | force
    a = a[:, a.var["highly_variable"]].copy()
    print(f"  {a.shape[1]} HVGs (of which {int(force.sum())} force-kept targets)")

    # PCA on controls
    ctrl_mask = (a.obs["target_gene"] == "").to_numpy()
    X_ctrl = np.asarray(a[ctrl_mask].X.toarray() if hasattr(a.X, "toarray") else a[ctrl_mask].X, dtype=np.float32)
    pca = PCA(n_components=D, random_state=SEED).fit(X_ctrl)
    loadings = pca.components_.T.astype(np.float32)
    basis = ao.make_program_basis(loadings, a.var_names, method="pca_external",
                                  control_count=int(ctrl_mask.sum()), normalize=False)
    # Σ_ctrl
    sigma_ctrl = np.cov(X_ctrl, rowvar=False).astype(np.float32)
    np.savez_compressed(RESULTS / "k562_all_targets_sigma_ctrl.npz",
                        sigma_ctrl=sigma_ctrl, gene_names=np.array(a.var_names))

    # measure via build_guide_responses + measure_from_sensitivity (proxy-efficiency path)
    responses, dropped = ao.build_guide_responses(
        a, basis,
        guide_key="guide", control_label="non-targeting", target_key="target_gene",
        min_cells_per_guide=MIN_CELLS_PER_GUIDE, min_knockdown_efficiency=MIN_KD_EFF,
        efficiency_estimator="auto", allow_proxy_efficiency=True)
    print(f"  {len(responses)} guides retained, {len(dropped)} dropped")
    S = np.stack([r.response for r in responses], axis=1)
    U = np.stack([r.input_vector for r in responses], axis=1)
    guide_names = [r.guide for r in responses]
    guide_effs = {r.guide: r.efficiency for r in responses}
    guide_tgts = {r.guide: r.target for r in responses if r.target is not None}
    meas = ao.measure_from_sensitivity(
        S=S, U=U, guide_names=guide_names, guide_efficiencies=guide_effs,
        guide_targets=guide_tgts, reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
        state_label="K562_all_qualifying")

    bundle = {"measurement": meas, "basis": basis, "n_qualifying": len(qualifying),
              "provenance": {"seed": SEED, "d": D, "rank_tol": RANK_TOL,
                             "n_hvg": N_HVG, "min_cells_per_target": MIN_CELLS_PER_TARGET,
                             "min_cells_per_guide": MIN_CELLS_PER_GUIDE}}
    with out_pkl.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"  saved {out_pkl}")
    return bundle


def main():
    bundle = _fit_all_targets_k562()
    meas = bundle["measurement"]
    S = meas.S; U = meas.U
    guides = list(meas.report.retained_guides)
    guide_effs = dict(meas.report.guide_efficiencies)
    stored = getattr(meas.report, "guide_targets", None)
    if stored:
        stored = {str(k): str(v) for k, v in dict(stored).items()}
        target_of_g = np.array([stored.get(str(g), str(g)) for g in guides])
    else:
        target_of_g = np.array([str(g)[len("guide_"):] if str(g).startswith("guide_") else str(g).split("_")[0]
                                 for g in guides])
    print(f"  full-panel n_guides = {S.shape[1]}, n_targets = {len(set(target_of_g.tolist()))}")

    # Subsample 50 random panels
    unique_targets = sorted(set(target_of_g.tolist()))
    rng = np.random.default_rng(SEED)
    panels = []
    for p in range(N_PANELS):
        chosen = rng.choice(np.array(unique_targets, dtype=object), size=min(PANEL_SIZE, len(unique_targets)), replace=False)
        chosen_set = set(chosen.tolist())
        mask = np.array([t in chosen_set for t in target_of_g])
        panels.append(np.where(mask)[0])
    print(f"  {N_PANELS} random panels sampled, size ≈ {PANEL_SIZE}")

    real_rhos, lin_rhos = [], []
    for i, idx in enumerate(panels):
        S_p = S[:, idx]; U_p = U[:, idx]; tog_p = target_of_g[idx]
        r_real = _nested_rho(S_p, U_p, tog_p, seed=SEED + i * 31)
        # matched linear-truth on this panel's U — α_S from median col-norm
        d = S_p.shape[0]
        S_med = float(np.median(np.linalg.norm(S_p, axis=0)))
        rng0 = np.random.default_rng(SEED + i * 31)
        J_ref0 = rng0.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
        S_ref = -np.linalg.solve(J_ref0, U_p)
        ref_med = float(np.median(np.linalg.norm(S_ref, axis=0)))
        alpha_hat = S_med / max(ref_med, 1e-30)
        lin_r = []
        for k in range(N_LIN_SIM_PER_PANEL):
            rng_k = np.random.default_rng(SEED + i * 31 + 7 + k)
            J_ref = rng_k.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
            J = J_ref / max(alpha_hat, 1e-30)
            S_true = -np.linalg.solve(J, U_p)
            S_sim = S_true + SIGMA_K562 * rng_k.normal(size=S_true.shape)
            lin_r.append(_nested_rho(S_sim, U_p, tog_p, seed=SEED + i * 31 + 7 + k))
        real_rhos.append(r_real); lin_rhos.append(float(np.mean(lin_r)))
        if (i + 1) % 10 == 0:
            print(f"    panel {i+1}/{N_PANELS}: real ρ={r_real:.3f}   lin-truth mean ρ={np.mean(lin_r):.3f}")

    payload = {
        "seed": SEED, "n_panels": N_PANELS, "panel_size": PANEL_SIZE,
        "n_lin_sim_per_panel": N_LIN_SIM_PER_PANEL,
        "sigma_used": SIGMA_K562,
        "n_qualifying_targets": len(unique_targets),
        "real_rho_distribution": {
            "median": float(np.median(real_rhos)),
            "p05": float(np.percentile(real_rhos, 5)),
            "p95": float(np.percentile(real_rhos, 95)),
            "min": float(np.min(real_rhos)),
            "max": float(np.max(real_rhos)),
        },
        "linear_truth_rho_distribution": {
            "median": float(np.median(lin_rhos)),
            "p05": float(np.percentile(lin_rhos, 5)),
            "p95": float(np.percentile(lin_rhos, 95)),
        },
        "fraction_panels_real_rho_below_0_95": float(np.mean(np.array(real_rhos) < 0.95)),
        "all_real_rhos": real_rhos,
        "all_lin_truth_rhos": lin_rhos,
    }
    out_path = OUT / "F_step3_random_panels.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {out_path}")
    print(f"\n=== SUMMARY ===")
    print(f"real ρ: median={payload['real_rho_distribution']['median']:.4f}, "
          f"[5th–95th] = [{payload['real_rho_distribution']['p05']:.4f}, {payload['real_rho_distribution']['p95']:.4f}]")
    print(f"matched linear-truth ρ (per panel mean): median={payload['linear_truth_rho_distribution']['median']:.4f}, "
          f"[5th–95th] = [{payload['linear_truth_rho_distribution']['p05']:.4f}, {payload['linear_truth_rho_distribution']['p95']:.4f}]")
    print(f"fraction of panels with real ρ < 0.95: {payload['fraction_panels_real_rho_below_0_95']:.2%}")


if __name__ == "__main__":
    main()
