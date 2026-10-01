"""Supplementary Table S3: PCA(d=30) vs FA(d=30) under the SAME pipeline.

Journal-version robustness check (v0.3.3-corrections). scJDO uses FA as its
default state space, so a reviewer will ask (a) whether the choice of PCA
in paper 1's Table 1 is responsible for the operator-level failure story,
and (b) whether a FA fit — with orthonormal loadings so the comparison is
on the same geometric footing as PCA — tells a different story.

To get a clean apples-to-apples PCA vs FA, both arms run through the SAME
code path (unlike paper-1 Table 1, which uses `measure_operator` + the
hardcoded F1/F2 sigma/alpha_S anchors). Each arm:
  1. Fits its basis on NT controls (same HVG, filters, seeds).
     - PCA: sklearn PCA (components are orthonormal by construction).
     - FA : sklearn FactorAnalysis -> `fa.components_.T` is NOT orthonormal
            by construction; we QR-orthonormalize so ||Q^T Q - I||_F = 0.
            Semantically: use the FA fit to pick a d-dim subspace, but
            rotate within that subspace to an orthonormal basis so the
            projection has the same scale properties as PCA.
  2. `build_guide_responses` with the proxy-efficiency path (Replogle
     normalized residuals; same as reproduction/42, /53 and Methods §4.4).
  3. `measure_from_sensitivity` for S, U, bootstrap.
  4. sigma bootstrap in-basis (split-half within-guide; §4.4).
  5. alpha_S by matched-median column-norm vs dense random J_ref in-basis.
  6. Target-grouped nested CV (identical to 40_nested_cv_rho.py). Report
     both the pooled rho AND the per-outer-fold rho SD.

Outputs:
  - results/{k562_essential,rpe1_essential,jost}_pca55_measurement.pkl
  - results/{k562_essential,rpe1_essential}_fa_measurement.pkl
    (plus results/jost_fa_measurement.pkl) — FA pkls rewritten with QR loadings
  - results/recheck/F_pca_fa_nested_cv_rho.json

Runtime ~1 hr total (each screen gets 2 fits).
"""
from __future__ import annotations
import gzip
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
RESULTS = REPO / "results"
OUT = RESULTS / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260927
D = 30
N_HVG = 3000
N_TARGETS_KEEP = 200
MIN_CELLS_PER_GUIDE = 30
RANK_TOL = 1e-2
N_OUTER = 5
N_INNER = 3
N_REPS_LIN_SIM = 15
N_REPS_ALPHA = 200
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]

K562_H5AD = DATA / "K562_essential_normalized_singlecell_01.h5ad"
RPE1_H5AD = DATA / "rpe1_normalized_singlecell_01.h5ad"
JOST_DIR = DATA / "jost2020"


# -----------------------------------------------------------------------------
# Basis fits.
# -----------------------------------------------------------------------------
def _fit_pca_loadings(X_ctrl, d, seed):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=d, random_state=seed).fit(X_ctrl)
    W = pca.components_.T.astype(np.float32)  # (n_genes, d), orthonormal columns
    return W, {"explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))}


def _fit_fa_loadings_qr(X_ctrl, d, seed):
    """sklearn FactorAnalysis, then QR-orthonormalize the loading matrix so
    ||Q^T Q - I|| = 0. Same column space as `fa.components_.T` but with
    orthonormal columns, so the per-coordinate scale matches PCA.
    """
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components=d, random_state=seed,
                        tol=1e-2, max_iter=1000)
    fa.fit(X_ctrl)
    W_raw = fa.components_.T.astype(np.float32)  # (n_genes, d), NOT orthonormal
    Q, _ = np.linalg.qr(W_raw, mode="reduced")   # (n_genes, d), orthonormal cols
    return Q.astype(np.float32), {
        "fa_noise_variance_mean": float(np.mean(fa.noise_variance_)),
        "orthogonality_before_qr": float(np.linalg.norm(W_raw.T @ W_raw - np.eye(d))),
        "orthogonality_after_qr": float(np.linalg.norm(Q.T @ Q - np.eye(d))),
    }


BASIS_FITTERS = {"pca": _fit_pca_loadings, "fa": _fit_fa_loadings_qr}


# -----------------------------------------------------------------------------
# Replogle K562 / RPE1 pipeline.
# -----------------------------------------------------------------------------
def _build_replogle_measurement(h5ad_path, tag, basis_name):
    import anchorop as ao
    print(f"[{tag}/{basis_name}] loading {h5ad_path.name} …", flush=True)
    adata = ao.load_replogle_h5ad(str(h5ad_path))
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    finite = np.isfinite(X).all(axis=0)
    adata = adata[:, finite].copy(); X = X[:, finite]

    target_counts = adata.obs.loc[adata.obs["target_gene"] != "", "target_gene"].value_counts()
    qualifying = target_counts[target_counts >= 60].index.tolist()[:N_TARGETS_KEEP]
    keep_mask = adata.obs["target_gene"].isin(qualifying) | (adata.obs["target_gene"] == "")
    adata = adata[keep_mask].copy()

    perturbed = adata.obs["target_gene"] != ""
    adata.obs.loc[perturbed, "guide"] = "guide_" + adata.obs.loc[perturbed, "target_gene"].astype(str)

    X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_var = X_dense.var(axis=0)
    top_hvg = np.argsort(-gene_var)[:N_HVG]
    target_syms = {t for t in adata.obs["target_gene"].astype(str).unique() if t}
    extra_idx = np.array([i for i, g in enumerate(np.asarray(adata.var_names))
                           if g in target_syms])
    keep_feats = np.union1d(top_hvg, extra_idx)
    adata_hvg = adata[:, keep_feats].copy()

    ctrl_mask = (adata_hvg.obs["guide"] == "non-targeting").to_numpy()
    X_ctrl = adata_hvg.X.toarray() if hasattr(adata_hvg.X, "toarray") else adata_hvg.X
    X_ctrl = X_ctrl[ctrl_mask]
    print(f"[{tag}/{basis_name}] fitting {basis_name.upper()}(d={D}) on "
          f"{X_ctrl.shape[0]} NT controls, {X_ctrl.shape[1]} HVGs …", flush=True)
    W, basis_meta = BASIS_FITTERS[basis_name](X_ctrl - X_ctrl.mean(0), D, SEED)
    print(f"[{tag}/{basis_name}] ||W^T W - I||_F = "
          f"{np.linalg.norm(W.T @ W - np.eye(D)):.3e}", flush=True)
    basis = ao.make_program_basis(
        W, adata_hvg.var_names,
        method=f"{basis_name}_external",
        control_count=int(ctrl_mask.sum()), normalize=False,
    )

    print(f"[{tag}/{basis_name}] build_guide_responses + measure_from_sensitivity "
          f"(proxy-efficiency, Methods §4.4) …", flush=True)
    responses, dropped = ao.build_guide_responses(
        adata_hvg, basis,
        guide_key="guide", control_label="non-targeting", target_key="target_gene",
        min_cells_per_guide=MIN_CELLS_PER_GUIDE,
        min_knockdown_efficiency=0.05,
        efficiency_estimator="auto", allow_proxy_efficiency=True,
    )
    print(f"[{tag}/{basis_name}] responses: {len(responses)} retained, "
          f"{len(dropped)} dropped", flush=True)
    S = np.stack([r.response for r in responses], axis=1)
    U = np.stack([r.input_vector for r in responses], axis=1)
    guide_names = [r.guide for r in responses]
    guide_effs = {r.guide: r.efficiency for r in responses}
    guide_tgts = {r.guide: r.target for r in responses if r.target is not None}
    measurement = ao.measure_from_sensitivity(
        S=S, U=U,
        guide_names=guide_names, guide_efficiencies=guide_effs, guide_targets=guide_tgts,
        reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
        bootstrap=100, bootstrap_seed=SEED, state_label=f"{tag}_{basis_name}",
    )
    r = measurement.report
    print(f"[{tag}/{basis_name}] retained {r.n_guides_retained}/{r.n_guides_input}, "
          f"rank {r.effective_response_rank}/{r.d}, cond {r.condition_number:.2f}",
          flush=True)
    return {"measurement": measurement, "basis": basis, "basis_meta": basis_meta}, adata_hvg


# -----------------------------------------------------------------------------
# Jost end-to-end pipeline.
# -----------------------------------------------------------------------------
def _build_jost_adata():
    import anndata as ad
    import scipy.io as sio
    import pandas as pd
    from anndata.utils import make_index_unique

    print("[Jost] loading raw 10x matrix …", flush=True)
    mat = sio.mmread(JOST_DIR / "GSE132080_10X_matrix.mtx.gz").T.tocsr()
    with gzip.open(JOST_DIR / "GSE132080_10X_barcodes.tsv.gz", "rt") as f:
        barcodes = [ln.strip() for ln in f]
    with gzip.open(JOST_DIR / "GSE132080_10X_genes.tsv.gz", "rt") as f:
        gene_rows = [ln.strip().split("\t") for ln in f]
    gene_ids = [row[0] for row in gene_rows]
    gene_syms = [row[1] if len(row) > 1 else row[0] for row in gene_rows]
    ids = pd.read_csv(JOST_DIR / "GSE132080_cell_identities.csv.gz").set_index("cell_barcode")
    obs = pd.DataFrame(index=barcodes)
    id_col = ids["guide_identity"].reindex(obs.index).astype(str)
    obs["sgRNA"] = id_col.fillna("NA")
    obs["assigned"] = ~obs["sgRNA"].isin({"NA", "*", "nan"})
    obs["target"] = obs["sgRNA"].str.split("_").str[0].fillna("NA")
    obs["is_control"] = obs["target"].isin({"neg"})
    var = pd.DataFrame({"gene_id": gene_ids, "gene_symbol": gene_syms})
    var.index = make_index_unique(pd.Index(gene_syms))
    a = ad.AnnData(mat, obs=obs, var=var)
    keep = a.obs["assigned"].to_numpy()
    a = a[keep].copy()
    print(f"[Jost] loaded {a.shape}, control cells: {int(a.obs['is_control'].sum())}, "
          f"perturbed cells: {int((~a.obs['is_control']).sum())}", flush=True)
    return a


def _build_jost_measurement(basis_name):
    import anchorop as ao
    import scanpy as sc
    a = _build_jost_adata()
    print("[Jost] HVG + log1p normalization …", flush=True)
    sc.pp.filter_cells(a, min_counts=200)
    sc.pp.filter_genes(a, min_cells=10)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat")
    targets = set(a.obs.loc[~a.obs["is_control"], "target"].astype(str).unique())
    force_keep = a.var_names.isin(targets)
    a.var["highly_variable"] = a.var["highly_variable"] | force_keep
    a = a[:, a.var["highly_variable"]].copy()
    print(f"[Jost] HVG: {a.shape[1]} (of which {int(force_keep.sum())} force-kept targets)",
          flush=True)

    ctrl = a[a.obs["is_control"].to_numpy()].copy()
    X = np.asarray(ctrl.X.toarray() if hasattr(ctrl.X, "toarray") else ctrl.X,
                   dtype=np.float32)
    print(f"[Jost/{basis_name}] fitting {basis_name.upper()}(d={D}) on {X.shape[0]} "
          f"NT controls, {X.shape[1]} HVGs …", flush=True)
    W, basis_meta = BASIS_FITTERS[basis_name](X - X.mean(0), D, SEED)
    basis = ao.make_program_basis(
        W, a.var_names, method=f"{basis_name}_external",
        control_count=int(a.obs["is_control"].sum()), normalize=False,
    )
    print(f"[Jost/{basis_name}] ||W^T W - I||_F = "
          f"{np.linalg.norm(W.T @ W - np.eye(D)):.3e}", flush=True)

    a.obs["guide"] = a.obs["sgRNA"].astype(str)
    a.obs["target_gene"] = a.obs["target"].astype(str)
    ctrl_mask = a.obs["is_control"].to_numpy()
    a.obs.loc[ctrl_mask, "guide"] = "non-targeting"
    a.obs.loc[ctrl_mask, "target_gene"] = ""

    print(f"[Jost/{basis_name}] running measure_operator (log1p counts, standard path) …",
          flush=True)
    measurement = ao.measure_operator(
        a, basis,
        guide_key="guide", target_key="target_gene", control_label="non-targeting",
        min_cells_per_guide=MIN_CELLS_PER_GUIDE, min_knockdown_efficiency=0.05,
        reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
        state_label=f"Jost_2020_{basis_name}",
        aggregate_replicate_guides=False,
    )
    r = measurement.report
    print(f"[Jost/{basis_name}] retained {r.n_guides_retained}/{r.n_guides_input}, "
          f"rank {r.effective_response_rank}/{r.d}, cond {r.condition_number:.2f}",
          flush=True)
    return {"measurement": measurement, "basis": basis, "basis_meta": basis_meta}, a


# -----------------------------------------------------------------------------
# sigma bootstrap (within-guide split-half, in-basis).
# -----------------------------------------------------------------------------
def _bootstrap_sigma(adata, basis, retained_guides, guide_key, control_label,
                     min_cells=20, seed=SEED):
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    W = np.asarray(basis.loadings)
    guide_labels = adata.obs[guide_key].astype(str).to_numpy()
    ctrl_mask = guide_labels == control_label
    X = adata.X
    def _rows(mask_or_idx):
        A = X[mask_or_idx]
        return np.asarray(A.toarray() if sp.issparse(A) else A, dtype=np.float32)
    Z_ctrl = _rows(ctrl_mask) @ W
    z_ctrl_mean = Z_ctrl.mean(axis=0)
    d = W.shape[1]
    per_target = []
    retained_set = set(str(g) for g in retained_guides)
    for g in retained_set:
        idx = np.where(guide_labels == g)[0]
        if len(idx) < min_cells:
            continue
        rng.shuffle(idx)
        half = len(idx) // 2
        z1 = _rows(idx[:half]).mean(axis=0) @ W - z_ctrl_mean
        z2 = _rows(idx[half:2*half]).mean(axis=0) @ W - z_ctrl_mean
        per_target.append(float(np.linalg.norm(z1 - z2) / (2 * np.sqrt(d))))
    return float(np.median(per_target)) if per_target else float("nan")


# -----------------------------------------------------------------------------
# alpha_S matched-median derivation (in-basis).
# -----------------------------------------------------------------------------
def _draw_dense_J(d, seed, c=1.5):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    return G - c * np.eye(d)


def _matched_median_alpha(U, S_real, n_reps=N_REPS_ALPHA, seed=SEED):
    sim_norms = []
    for r in range(n_reps):
        J_ref = _draw_dense_J(U.shape[0], seed + r)
        S_true = -np.linalg.solve(J_ref, U)
        sim_norms.append(np.linalg.norm(S_true, axis=0))
    median_sim_norm = float(np.median(np.concatenate(sim_norms)))
    S_real_median = float(np.median(np.linalg.norm(S_real, axis=0)))
    return float(S_real_median / max(median_sim_norm, 1e-30))


# -----------------------------------------------------------------------------
# Nested-CV with per-fold SD (verbatim 40_nested_cv_rho.py).
# -----------------------------------------------------------------------------
def _fit_A_at_rank(S, U, r):
    r = int(r)
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]), dtype=float)
    Us, sv, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(sv))
    sv_inv = np.zeros_like(sv); sv_inv[:r] = 1.0 / sv[:r]
    S_pinv = (Vt.T * sv_inv) @ Us.T
    return -U @ S_pinv


def _mse_pooled(A, S_test, U_test):
    resid = A @ S_test + U_test
    return float(np.sum(resid ** 2))


def _target_grouped_folds(target_of_g, k, seed):
    rng = np.random.default_rng(seed)
    targets = sorted(set(target_of_g.tolist()))
    rng.shuffle(targets)
    tgroups = np.array_split(np.array(targets), k)
    folds = []
    for i in range(k):
        test_targets = set(tgroups[i].tolist())
        test_mask = np.array([t in test_targets for t in target_of_g])
        folds.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return folds


def _inner_pick_rank(S_tr, U_tr, target_of_tr, seed):
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
    best_r, best_mse = None, np.inf
    for r in RANK_GRID:
        pooled_num = 0.0; ok = True
        for tr, val in inner:
            if r > 0 and len(tr) <= r:
                ok = False; break
            A = _fit_A_at_rank(S_tr[:, tr], U_tr[:, tr], r)
            pooled_num += _mse_pooled(A, S_tr[:, val], U_tr[:, val])
        if ok and pooled_num < best_mse:
            best_mse, best_r = pooled_num, r
    return best_r if best_r is not None else 0


def _nested_cv_rho(S, U, target_of_g, seed):
    outer = _target_grouped_folds(target_of_g, N_OUTER, seed)
    num_sq = 0.0; den_sq = 0.0
    per_fold = []; picked = []
    for i, (tr, te) in enumerate(outer):
        if len(tr) < 3:
            continue
        r = _inner_pick_rank(S[:, tr], U[:, tr], target_of_g[tr], seed + i * 17)
        if r > 0 and len(tr) <= r:
            r = 0
        A = _fit_A_at_rank(S[:, tr], U[:, tr], r)
        resid = A @ S[:, te] + U[:, te]
        num_sq += float(np.sum(resid ** 2))
        den_sq += float(np.sum(U[:, te] ** 2))
        rho_fold = float(np.sqrt(np.sum(resid ** 2) / max(np.sum(U[:, te] ** 2), 1e-30)))
        per_fold.append({"fold": i, "picked_rank": int(r),
                         "n_train": int(len(tr)), "n_test": int(len(te)),
                         "rho_fold": rho_fold})
        picked.append(int(r))
    rho_pooled = float(np.sqrt(num_sq / max(den_sq, 1e-30)))
    per_fold_rhos = [pf["rho_fold"] for pf in per_fold]
    rho_fold_sd = float(np.std(per_fold_rhos, ddof=1)) if len(per_fold_rhos) > 1 else 0.0
    return {"rho_pooled": rho_pooled,
            "rho_fold_sd": rho_fold_sd,
            "per_fold": per_fold,
            "picked_ranks_summary": {
                "min": int(min(picked)) if picked else None,
                "median": int(np.median(picked)) if picked else None,
                "max": int(max(picked)) if picked else None,
            }}


def _linear_sim_nested(U, target_of_g, sigma, alpha_S, seed_base, n_reps):
    rhos = []; per_rep_picks = []
    for r in range(n_reps):
        rng = np.random.default_rng(seed_base + r + 5000)
        J_ref = rng.normal(size=(U.shape[0], U.shape[0])) / np.sqrt(U.shape[0]) \
                 - 1.5 * np.eye(U.shape[0])
        J = J_ref / max(alpha_S, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_obs = S_true + sigma * rng.normal(size=S_true.shape)
        out = _nested_cv_rho(S_obs, U, target_of_g, seed_base + r + 5000)
        rhos.append(out["rho_pooled"])
        per_rep_picks.append(out["picked_ranks_summary"]["median"])
    return {"n_reps": n_reps, "sigma": sigma, "alpha_S": alpha_S,
            "rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos, ddof=1)),
            "picked_rank_median_summary": [int(x) if x is not None else None
                                            for x in per_rep_picks]}


def _derive_targets(guide_names, stored_map=None):
    if stored_map:
        clean = {str(k): str(v) for k, v in dict(stored_map).items()}
        return np.array([clean.get(str(g), str(g)) for g in guide_names])
    out = []
    for g in guide_names:
        s = str(g)
        if s.startswith("guide_"):
            out.append(s[len("guide_"):])
        else:
            out.append(s.split("_")[0])
    return np.array(out)


# -----------------------------------------------------------------------------
# Driver.
# -----------------------------------------------------------------------------
def _one_screen(tag, bundle, adata, guide_key, control_label):
    meas = bundle["measurement"]; basis = bundle["basis"]
    S = meas.S; U = meas.U
    guides = list(meas.report.retained_guides)
    stored_map = getattr(meas.report, "guide_targets", None)
    target_of_g = _derive_targets(guides, stored_map)
    n_targets = len(set(target_of_g.tolist()))

    print(f"[{tag}] bootstrapping sigma in-basis …", flush=True)
    sigma = _bootstrap_sigma(adata, basis, guides, guide_key, control_label, seed=SEED)
    print(f"[{tag}] sigma = {sigma:.4f}", flush=True)

    print(f"[{tag}] deriving alpha_S (matched-median column-norm) …", flush=True)
    alpha_S = _matched_median_alpha(U, S)
    print(f"[{tag}] alpha_S = {alpha_S:.2f}", flush=True)

    print(f"[{tag}] nested CV (real) …", flush=True)
    real = _nested_cv_rho(S, U, target_of_g, SEED)
    print(f"[{tag}] real rho (nested-CV) = {real['rho_pooled']:.4f} "
          f"(fold SD {real['rho_fold_sd']:.4f})", flush=True)

    print(f"[{tag}] linear-truth matched-SNR …", flush=True)
    lin = _linear_sim_nested(U, target_of_g, sigma, alpha_S,
                             seed_base=SEED, n_reps=N_REPS_LIN_SIM)
    print(f"[{tag}] matched-linear rho = {lin['rho_mean']:.4f} +/- {lin['rho_std']:.4f}",
          flush=True)

    return {
        "n_sgRNA": int(len(guides)),
        "n_targets": int(n_targets),
        "sigma": sigma,
        "alpha_S": alpha_S,
        "real_nested_cv": real,
        "linear_truth_matched_alpha_nested_cv": lin,
        "basis_meta": bundle["basis_meta"],
    }


def _save_bundle(path, bundle):
    with Path(path).open("wb") as f:
        pickle.dump(bundle, f)
    print(f"  saved {path}", flush=True)


def main():
    payload = {"seed": SEED, "n_outer": N_OUTER, "n_inner": N_INNER,
               "rank_grid": RANK_GRID,
               "basis_variants": ["PCA(n_components=30)",
                                   "FA(n_components=30, QR-orthonormalized loadings)"],
               "datasets": {}}

    screens = [
        ("K562_essential", "replogle", K562_H5AD,
         RESULTS / "k562_essential_pca55_measurement.pkl",
         RESULTS / "k562_essential_fa_measurement.pkl"),
        ("RPE1_essential", "replogle", RPE1_H5AD,
         RESULTS / "rpe1_essential_pca55_measurement.pkl",
         RESULTS / "rpe1_essential_fa_measurement.pkl"),
        ("Jost_2020", "jost", None,
         RESULTS / "jost_pca55_measurement.pkl",
         RESULTS / "jost_fa_measurement.pkl"),
    ]

    for tag, kind, h5ad_path, pca_bundle_path, fa_bundle_path in screens:
        print("=" * 72); print(f"{tag}"); print("=" * 72)
        payload["datasets"][tag] = {}
        for basis_name, bundle_path in [("pca", pca_bundle_path),
                                         ("fa", fa_bundle_path)]:
            if kind == "replogle":
                bundle, adata = _build_replogle_measurement(h5ad_path, tag, basis_name)
            else:
                bundle, adata = _build_jost_measurement(basis_name)
            _save_bundle(bundle_path, bundle)
            payload["datasets"][tag][basis_name.upper()] = _one_screen(
                tag, bundle, adata, guide_key="guide", control_label="non-targeting",
            )
            del bundle, adata

    out_path = OUT / "F_pca_fa_nested_cv_rho.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {out_path}", flush=True)

    print("\n=== Summary: PCA(d=30) vs FA(d=30,QR) under the SAME pipeline ===",
          flush=True)
    for tag, info in payload["datasets"].items():
        for arm, data in info.items():
            print(f"  {tag}/{arm}: real = {data['real_nested_cv']['rho_pooled']:.4f} "
                  f"(fold SD {data['real_nested_cv']['rho_fold_sd']:.4f}), "
                  f"matched-linear = "
                  f"{data['linear_truth_matched_alpha_nested_cv']['rho_mean']:.4f} "
                  f"+/- {data['linear_truth_matched_alpha_nested_cv']['rho_std']:.4f} "
                  f"[sigma={data['sigma']:.4f}, alpha_S={data['alpha_S']:.2f}]",
                  flush=True)


if __name__ == "__main__":
    main()
