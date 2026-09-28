"""F3 on Jost 2020, end to end (reviewer request).

**Primary basis: PCA on log-normalized non-targeting-control cells**, to
match Replogle. NMF-then-QR was tried first and is kept as an optional
sensitivity check via ``--basis nmf_qr``, but any Jost-vs-Replogle
comparison must use the same-basis path or it mixes dataset with basis.

Steps, in order:

  1. Build the Jost measurement bundle end-to-end from raw counts:
     scanpy normalize_total + log1p → HVG selection → PCA on control
     cells → measure_operator at rank_tol = 1e-2.
  2. Bootstrap σ_Jost per-sgRNA in Jost's own PCA basis — do NOT
     transplant σ from Replogle's basis. Addresses the F1
     "units of σ" check for Jost.
  3. Compute F3 real ρ (held-out prediction) on the Jost measurement.
  4. Run a matched-SNR linear-truth simulation using **Jost's actual
     rank-24 U** (not a truncated or padded version), so the
     linear-truth baseline is like-for-like with real Jost.
  5. Report ρ *on the identified subspace* — restrict evaluation to
     range(U), the actuated input subspace at rank_tol — as well as the
     overall ρ. The identified-subspace ρ is the honest comparison
     under the corrected `full_domain_identified` gate.
  6. Direction-only ρ on the Jost measurement (independent replication
     of the reading-(c) verdict with Jost's count-based κ).

Outputs `results/jost_measurement.pkl` and
`results/recheck/F3_Jost.json`.

Not part of `run_all.sh`. Runs once, off-line, before the restructure
is signed off. Requires the Jost 2020 GSE132080 raw counts under
`examples/data/jost2020/`. Runtime ~10–20 min.

Usage:
    python reproduction/37_recheck_F3_jost_end_to_end.py                # PCA basis (default, matches Replogle)
    python reproduction/37_recheck_F3_jost_end_to_end.py --basis nmf_qr # sensitivity check with NMF+QR basis
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data" / "jost2020"
RESULTS = REPO / "results"
OUT = RESULTS / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260927
D = 30
RANK_TOL = 1e-2
MIN_CELLS_PER_GUIDE = 30
N_HVG = 3000
N_REPS_LIN_SIM = 15


def build_jost_adata():
    """Load the Jost raw 10x matrix + cell/target identities into AnnData.

    Jost's cell-identity table uses ``cell_barcode`` (with the 10x -1/-2/-3
    gemgroup suffix) and ``guide_identity`` (e.g. TARGET_TARGET_+_coord_00 for
    real guides, ``neg_ctrl_non-targeting_NNNNN`` for controls).
    """
    import anndata as ad
    import scipy.io as sio
    import pandas as pd
    import gzip

    print("[1/6] loading Jost 10x matrix …")
    mat = sio.mmread(DATA / "GSE132080_10X_matrix.mtx.gz").T.tocsr()  # (cells, genes)
    with gzip.open(DATA / "GSE132080_10X_barcodes.tsv.gz", "rt") as f:
        barcodes = [ln.strip() for ln in f]
    with gzip.open(DATA / "GSE132080_10X_genes.tsv.gz", "rt") as f:
        gene_rows = [ln.strip().split("\t") for ln in f]
    gene_ids = [row[0] for row in gene_rows]
    gene_syms = [row[1] if len(row) > 1 else row[0] for row in gene_rows]
    ids = pd.read_csv(DATA / "GSE132080_cell_identities.csv.gz")
    ids = ids.set_index("cell_barcode")
    obs = pd.DataFrame(index=barcodes)
    id_col = ids["guide_identity"].reindex(obs.index).astype(str)
    obs["sgRNA"] = id_col.fillna("NA")
    # Assigned/unassigned marker (Jost uses "*" for no-assignment cells).
    obs["assigned"] = ~obs["sgRNA"].isin({"NA", "*", "nan"})
    # Target = first underscore-separated token; controls have "neg" prefix.
    # Jost's target names are gene SYMBOLS (e.g. "GNB2L1", "MTOR").
    obs["target"] = obs["sgRNA"].str.split("_").str[0].fillna("NA")
    obs["is_control"] = obs["target"].isin({"neg"})
    # Use gene symbols as var_names so measure_operator's target lookup
    # (target_key values are symbols) matches the expression columns.
    var = pd.DataFrame({"gene_id": gene_ids, "gene_symbol": gene_syms})
    # Some symbols repeat; make unique by suffixing duplicates.
    from anndata.utils import make_index_unique
    var.index = make_index_unique(pd.Index(gene_syms))
    a = ad.AnnData(mat, obs=obs, var=var)
    # Drop unassigned cells (they carry no perturbation or NT label).
    keep = a.obs["assigned"].to_numpy()
    a = a[keep].copy()
    print(f"  loaded {a.shape}, control cells: {int(a.obs['is_control'].sum())}, "
          f"perturbed cells: {int((~a.obs['is_control']).sum())}")
    return a


def _hvg_and_normalize(adata):
    """Shared preprocessing before either basis is fit.

    Force-keeps Jost's target genes even if they miss the HVG cutoff — a
    target gene absent from the feature matrix causes measure_operator to
    drop that guide with 'target_gene_absent_from_response_matrix'.
    """
    import scanpy as sc
    sc.pp.filter_cells(adata, min_counts=200)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
    # Force-keep every Jost target gene present in var_names.
    targets = set(adata.obs.loc[~adata.obs["is_control"], "target"].astype(str).unique())
    force_keep = adata.var_names.isin(targets)
    n_forced = int(force_keep.sum())
    adata.var["highly_variable"] = adata.var["highly_variable"] | force_keep
    n_kept = int(adata.var["highly_variable"].sum())
    print(f"  HVG: {n_kept} (of which {n_forced} force-kept Jost targets; of {len(targets)} targets, {n_forced} are in var_names)")
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def fit_jost_pca_basis(adata, seed=SEED, d=D):
    """Primary basis: PCA on log-normalized non-targeting-control cells.

    This matches the Replogle pipeline in `01b`/`01c`, so a Jost-vs-Replogle
    comparison mixes only the dataset, not the basis.
    """
    from sklearn.decomposition import PCA
    print("[2/6] HVG + PCA on Jost controls (log-normalized) — primary basis …")
    adata = _hvg_and_normalize(adata)
    ctrl = adata[adata.obs["is_control"].to_numpy()].copy()
    X = np.asarray(ctrl.X.toarray() if hasattr(ctrl.X, "toarray") else ctrl.X, dtype=np.float32)
    pca = PCA(n_components=d, random_state=seed).fit(X)
    loadings = pca.components_.T.astype(np.float32)  # (genes, d), columns orthonormal
    print(f"  PCA fit, ||W^T W - I||_F = {np.linalg.norm(loadings.T @ loadings - np.eye(d)):.3e}")
    return loadings, adata


def fit_jost_nmf_basis(adata, seed=SEED, d=D):
    """Sensitivity check: NMF on log-normalized controls, then QR-orthonormalize.

    Not the primary Jost basis. Available under ``--basis nmf_qr`` for direct
    comparison with the PCA-primary run.
    """
    from sklearn.decomposition import NMF
    print("[2/6] HVG + NMF-then-QR on Jost controls — sensitivity basis …")
    adata = _hvg_and_normalize(adata)
    ctrl = adata[adata.obs["is_control"].to_numpy()].copy()
    X = np.asarray(ctrl.X.toarray() if hasattr(ctrl.X, "toarray") else ctrl.X, dtype=np.float32)
    model = NMF(n_components=d, init="nndsvda", random_state=seed, max_iter=400, tol=1e-4)
    _ = model.fit_transform(X)
    loadings = model.components_.T.astype(np.float32)
    Q, _ = np.linalg.qr(loadings, mode="reduced")
    print(f"  NMF converged, ||Q^T Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(d)):.3e}")
    return Q, adata


def build_S_U(adata, W, method_tag, seed=SEED):
    """Project per-sgRNA responses into program space; encode U from κ."""
    import anchorop as ao

    print("[3/6] measuring guide responses on Jost …")
    basis = ao.make_program_basis(W, adata.var_names, method=method_tag,
                                  control_count=int(adata.obs["is_control"].sum()),
                                  normalize=False)
    # Anchorop convention (see examples/01b): controls carry empty target_gene
    # and a guide value equal to control_label. Preserve per-sgRNA identity
    # for the perturbed cells so we get one guide per sgRNA (Jost's design).
    adata.obs["guide"] = adata.obs["sgRNA"].astype(str)
    adata.obs["target_gene"] = adata.obs["target"].astype(str)
    ctrl_mask = adata.obs["is_control"].to_numpy()
    adata.obs.loc[ctrl_mask, "guide"] = "non-targeting"
    adata.obs.loc[ctrl_mask, "target_gene"] = ""
    meas = ao.measure_operator(
        adata, basis,
        guide_key="guide", target_key="target_gene", control_label="non-targeting",
        min_cells_per_guide=MIN_CELLS_PER_GUIDE, min_knockdown_efficiency=0.05,
        reg="tsvd", reg_param="path", rank_tol=RANK_TOL, state_label="Jost_2020",
        aggregate_replicate_guides=False,  # keep per-sgRNA responses; Jost's design point
    )
    return meas, basis


def bootstrap_sigma_jost(adata, basis, meas, seed=SEED):
    """Within-sgRNA split-half bootstrap of σ in Jost's own basis."""
    print("[4/6] within-sgRNA split-half σ bootstrap on Jost basis …")
    rng = np.random.default_rng(seed)
    W = np.asarray(basis.loadings)
    Xg = adata.X
    to_dense = lambda x: np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)
    guide_col = adata.obs["guide"].to_numpy()
    ctrl_idx = np.where(guide_col == "non-targeting")[0]
    if ctrl_idx.size == 0:
        raise SystemExit("no control cells found")
    ctrl_mean = to_dense(Xg[ctrl_idx]).mean(axis=0) @ W  # (d,)
    dvals = []
    for gn in list(meas.report.retained_guides):
        rows = np.where(guide_col == gn)[0]
        if rows.size < 20:
            continue
        rng.shuffle(rows)
        half = rows.size // 2
        z1 = to_dense(Xg[rows[:half]]).mean(axis=0) @ W - ctrl_mean
        z2 = to_dense(Xg[rows[half : 2 * half]]).mean(axis=0) @ W - ctrl_mean
        # Var(d1 - d2) = 4 σ²_percell / N; per-entry σ = ||d1 - d2|| / (2 √d)
        dvals.append(float(np.linalg.norm(z1 - z2) / (2 * np.sqrt(W.shape[1]))))
    sigma = float(np.median(dvals)) if dvals else float("nan")
    print(f"  sigma_Jost_per_sgRNA (bootstrap in NMF/QR basis): {sigma:.4f}")
    return {"n_guides_used": len(dvals), "sigma_per_sgRNA": sigma}


def linear_sim_matched(meas, sigma, alpha_S, seed=SEED, n_reps=N_REPS_LIN_SIM):
    """Matched-SNR linear-truth simulation using Jost's actual U."""
    import anchorop as ao
    from anchorop.identifiability import regularized_pseudoinverse

    print("[5/6] matched-SNR linear-truth simulation on Jost's rank-24 U …")
    U = meas.U
    d, m = U.shape
    rhos_overall, rhos_subspace = [], []
    # Restrict-to-range(U) projector Q_U for the identified-subspace ρ.
    Uu, sv, _ = np.linalg.svd(U, full_matrices=False)
    keep = sv > (RANK_TOL * sv[0])
    Q_U = Uu[:, keep] @ Uu[:, keep].T
    for r in range(n_reps):
        rng = np.random.default_rng(seed + r)
        J_ref = rng.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
        J = J_ref / max(alpha_S, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_obs = S_true + sigma * rng.normal(size=S_true.shape)
        sim = ao.measure_from_sensitivity(
            S=S_obs, U=U,
            guide_names=list(meas.report.retained_guides),
            guide_efficiencies=dict(meas.report.guide_efficiencies),
            guide_targets=getattr(meas.report, "guide_targets", None),
            reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
        )
        hop = ao.held_out_prediction_check(sim, n_folds=5, seed=seed + 1000 + r)
        rhos_overall.append(float(hop.rho_pooled))
        # Identified-subspace ρ: project S_obs and U by Q_U, refit, evaluate.
        S_sub = Q_U @ S_obs
        U_sub = Q_U @ U
        sim_sub = ao.measure_from_sensitivity(
            S=S_sub, U=U_sub,
            guide_names=list(meas.report.retained_guides),
            guide_efficiencies=dict(meas.report.guide_efficiencies),
            guide_targets=getattr(meas.report, "guide_targets", None),
            reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
        )
        hop_sub = ao.held_out_prediction_check(sim_sub, n_folds=5, seed=seed + 2000 + r)
        rhos_subspace.append(float(hop_sub.rho_pooled))
    return {
        "n_reps": n_reps,
        "sigma": sigma, "alpha_S": alpha_S,
        "rho_overall_mean": float(np.mean(rhos_overall)),
        "rho_overall_std": float(np.std(rhos_overall)),
        "rho_identified_subspace_mean": float(np.mean(rhos_subspace)),
        "rho_identified_subspace_std": float(np.std(rhos_subspace)),
        "identified_dim": int(keep.sum()),
    }


def main():
    import argparse
    import anchorop as ao

    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=("pca_controls", "nmf_qr"), default="pca_controls",
                    help="Basis to fit on Jost. Default 'pca_controls' matches Replogle.")
    args = ap.parse_args()

    a = build_jost_adata()
    if args.basis == "pca_controls":
        W, a = fit_jost_pca_basis(a)
        method_tag = "pca_controls"
    else:
        W, a = fit_jost_nmf_basis(a)
        method_tag = "nmf_qr"
    meas, basis = build_S_U(a, W, method_tag)
    out_pkl = RESULTS / ("jost_measurement.pkl" if args.basis == "pca_controls"
                         else "jost_measurement_nmf_qr.pkl")
    with out_pkl.open("wb") as f:
        pickle.dump({"measurement": meas, "basis": basis, "provenance": {
            "seed": SEED, "d": D, "rank_tol": RANK_TOL, "n_hvg": N_HVG,
            "min_cells_per_guide": MIN_CELLS_PER_GUIDE, "basis_method": method_tag,
        }}, f)
    print("  saved:", out_pkl)

    boot = bootstrap_sigma_jost(a, basis, meas)

    print("[6/6] F3 real vs matched-SNR simulation on Jost …")
    hop_real = ao.held_out_prediction_check(meas, n_folds=5, seed=SEED)
    linc_real = ao.linearity_check(meas)
    # α_S from data: same recipe as F2 — median ‖S_col‖ observed / at α=1.
    S_med = float(np.median(np.linalg.norm(meas.S, axis=0)))
    # Reference (α=1) sim uses the same seed + first draw only, to get one
    # reference ‖S‖ per (dataset, seed). Extend if you want an ensemble.
    rng = np.random.default_rng(SEED)
    d, m = meas.U.shape
    J_ref = rng.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
    S_ref = -np.linalg.solve(J_ref, meas.U)
    ref_med = float(np.median(np.linalg.norm(S_ref, axis=0)))
    alpha_hat = S_med / max(ref_med, 1e-30)
    lin_pub = linear_sim_matched(meas, boot["sigma_per_sgRNA"], alpha_S=1.0)
    lin_mat = linear_sim_matched(meas, boot["sigma_per_sgRNA"], alpha_S=alpha_hat)

    # Independent replication of the reading-(c) verdict on Jost's count-based
    # κ: column-normalize S and U, refit, compare direction-only ρ to baseline.
    S = meas.S; U = meas.U
    sn = np.linalg.norm(S, axis=0); un = np.linalg.norm(U, axis=0)
    keep = (sn > 0) & (un > 0)
    S_dir = (S[:, keep] / sn[keep]); U_dir = (U[:, keep] / un[keep])
    gnames = [g for i, g in enumerate(meas.report.retained_guides) if keep[i]]
    meas_dir = ao.measure_from_sensitivity(
        S=S_dir, U=U_dir,
        guide_names=gnames,
        guide_efficiencies={g: meas.report.guide_efficiencies[g] for g in gnames},
        guide_targets=getattr(meas.report, "guide_targets", None),
        reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
    )
    hop_dir = ao.held_out_prediction_check(meas_dir, n_folds=5, seed=SEED)

    payload = {
        "seed": SEED, "d": D, "rank_tol": RANK_TOL,
        "basis": args.basis,
        "n_guides_retained": int(len(meas.report.retained_guides)),
        "input_subspace_dim": int(meas.report.input_subspace_dim),
        "effective_response_rank": int(meas.report.effective_response_rank),
        "full_domain_identified": bool(meas.report.full_domain_identified),
        "sigma_per_sgRNA_bootstrap_in_jost_basis": boot,
        "S_median_col_norm_observed": S_med,
        "S_median_col_norm_sim_alpha1": ref_med,
        "alpha_hat_S_jost": alpha_hat,
        "rho_real_overall": float(hop_real.rho_pooled),
        "rho_real_direction_only": float(hop_dir.rho_pooled),
        "rho_direction_only_delta_vs_baseline": float(hop_dir.rho_pooled - hop_real.rho_pooled),
        "rel_diff_real": float(linc_real.relative_difference),
        "linear_sim_at_published_alpha": lin_pub,
        "linear_sim_at_matched_alpha": lin_mat,
    }
    print(json.dumps(payload, indent=2))
    out_json = OUT / f"F3_Jost_{args.basis}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print("saved:", out_json)


if __name__ == "__main__":
    main()
