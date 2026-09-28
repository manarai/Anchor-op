"""Footprint-encoding test on all four screens (preregistered 2026-09-27).

Encoding: u_g = −κ_g · Wᵀ · Σ_ctrl · δ_g, with Σ_ctrl the gene–gene
covariance of non-targeting-control cells only, computed on the same
HVG feature space that fits the program basis W. δ_g is a gene-space
one-hot; W is the same PCA-on-controls loadings as in the released
pipeline. Perturbed cells never enter Σ_ctrl.

Screens (four fits total; K562 top-200 and K562 random-200 both
report):
  - K562_top200   from results/k562_essential_measurement.pkl
                  Σ_ctrl reloaded from the K562 h5ad if not cached.
  - K562_random200 from results/k562_random200_measurement.pkl
                  Σ_ctrl loaded from results/k562_random200_sigma_ctrl.npz
                  (already produced by script 42).
  - RPE1          from results/rpe1_essential_measurement.pkl,
                  Σ_ctrl reloaded from the RPE1 h5ad if not cached.
  - Jost          from results/jost_measurement.pkl, Σ_ctrl reloaded
                  from the Jost raw counts if not cached.

Recipe: target-grouped nested cross-validation matching Table 1 in
MANUSCRIPT.md §2.1. Rank grid {0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}.
Four controls per screen:
  (a) current-encoding baseline (u_g = −κ_g Wᵀδ_g, same recipe);
  (b) shuffled-footprint null with N_PERM = 100 permutations of the
      target→footprint mapping;
  (c) random-direction null with N_RAND = 100 draws of unit vectors
      rescaled to the column norms of the footprint U;
  (d) matched-SNR linear-truth on footprint U, α_S re-derived, N=15.

Direction-only nested-CV (footprint, column-normalized) is also
reported per screen, alongside the primary metric.

Success criterion, per screen: footprint nested-CV ρ ≤ 0.90 AND
below the 2.5th percentile of the shuffled-footprint null. Both
conditions must hold.

Output: results/recheck/F_footprint.json.
"""
from __future__ import annotations
import argparse, json, pickle, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260927
RANK_TOL = 1e-2
N_OUTER = 5
N_INNER = 3
N_REPS_LIN_SIM = 15
N_PERM = 100
N_RAND = 100
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]

SUCCESS_RHO_THRESHOLD = 0.90
SUCCESS_PERCENTILE = 2.5  # footprint ρ must be below the 2.5th percentile of shuffled null


# ==============================================================================
# Nested-CV helpers (identical to reproduction/40_nested_cv_rho.py)
# ==============================================================================

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
    tg = np.array_split(np.array(targets, dtype=object), k)
    folds = []
    for i in range(k):
        te = set(tg[i].tolist())
        m = np.array([t in te for t in target_of_g])
        folds.append((np.where(~m)[0], np.where(m)[0]))
    return folds


def _inner_pick(S_tr, U_tr, target_of_tr, seed):
    ts = sorted(set(target_of_tr.tolist()))
    k = min(N_INNER, len(ts))
    if k < 2:
        return 0
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
    num, den, picks, per_fold = 0.0, 0.0, [], []
    for i, (tr, te) in enumerate(outer):
        r = _inner_pick(S[:, tr], U[:, tr], target_of_g[tr], seed + i * 17)
        if r > 0 and len(tr) <= r:
            r = 0
        A = _fit_A_at_rank(S[:, tr], U[:, tr], r)
        resid = A @ S[:, te] + U[:, te]
        num += float(np.sum(resid ** 2))
        den += float(np.sum(U[:, te] ** 2))
        picks.append(int(r))
        per_fold.append(float(np.sqrt(np.sum(resid ** 2) / max(np.sum(U[:, te] ** 2), 1e-30))))
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_rho": per_fold,
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
            "picked_rank_median": int(np.median(picks)) if picks else 0}


# ==============================================================================
# Σ_ctrl computation from h5ad (skipped when a cache exists)
# ==============================================================================

def _compute_sigma_ctrl_replogle(h5ad_path, basis_gene_names, keep_targets, cache_path):
    """Load Replogle h5ad, restrict to (controls, basis genes), compute Σ_ctrl."""
    if cache_path.exists():
        print(f"    Σ_ctrl cached at {cache_path}, loading …")
        return np.load(cache_path)["sigma_ctrl"]
    print(f"    Σ_ctrl not cached, reloading h5ad {h5ad_path} …")
    import anchorop as ao
    adata = ao.load_replogle_h5ad(str(h5ad_path))
    # Restrict to controls + basis genes (already the HVG selection used to fit W).
    # If any basis gene missing from adata.var_names (rare edge case), zero it out.
    gene_index = {g: i for i, g in enumerate(adata.var_names)}
    idx = [gene_index[g] for g in basis_gene_names if g in gene_index]
    ctrl_mask = (adata.obs["target_gene"] == "").to_numpy()
    print(f"    control cells: {int(ctrl_mask.sum())}; basis genes present: {len(idx)}/{len(basis_gene_names)}")
    X = adata.X[ctrl_mask][:, idx]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    finite = np.isfinite(X).all(axis=0)
    if not finite.all():
        X[:, ~finite] = 0.0  # zero out any residual NaN gene columns
    sigma_ctrl = np.cov(X, rowvar=False).astype(np.float32)
    # Pad missing genes with zeros to keep the shape matching basis_gene_names.
    if len(idx) < len(basis_gene_names):
        full = np.zeros((len(basis_gene_names), len(basis_gene_names)), dtype=np.float32)
        present_positions = [i for i, g in enumerate(basis_gene_names) if g in gene_index]
        for i_local, i_full in enumerate(present_positions):
            full[i_full, present_positions] = sigma_ctrl[i_local]
        sigma_ctrl = full
    np.savez_compressed(cache_path, sigma_ctrl=sigma_ctrl, gene_names=np.array(basis_gene_names))
    print(f"    saved Σ_ctrl {sigma_ctrl.shape} to {cache_path}")
    return sigma_ctrl


def _compute_sigma_ctrl_jost(basis_gene_names, cache_path):
    """Compute Σ_ctrl for Jost from GSE132080 raw counts on controls."""
    if cache_path.exists():
        print(f"    Σ_ctrl cached at {cache_path}, loading …")
        return np.load(cache_path)["sigma_ctrl"]
    print(f"    Σ_ctrl not cached, reloading Jost raw counts …")
    import scanpy as sc, anndata as ad, scipy.io as sio, pandas as pd, gzip
    from anndata.utils import make_index_unique
    JD = DATA / "jost2020"
    mat = sio.mmread(JD / "GSE132080_10X_matrix.mtx.gz").T.tocsr()
    with gzip.open(JD / "GSE132080_10X_barcodes.tsv.gz", "rt") as f:
        barcodes = [ln.strip() for ln in f]
    with gzip.open(JD / "GSE132080_10X_genes.tsv.gz", "rt") as f:
        gene_rows = [ln.strip().split("\t") for ln in f]
    gene_ids = [r[0] for r in gene_rows]
    gene_syms = [r[1] if len(r) > 1 else r[0] for r in gene_rows]
    ids = pd.read_csv(JD / "GSE132080_cell_identities.csv.gz").set_index("cell_barcode")
    obs = pd.DataFrame(index=barcodes)
    id_col = ids["guide_identity"].reindex(obs.index).astype(str)
    obs["assigned"] = ~id_col.isin({"NA", "*", "nan"})
    obs["target"] = id_col.str.split("_").str[0].fillna("NA")
    obs["is_control"] = obs["target"].isin({"neg"})
    var = pd.DataFrame({"gene_id": gene_ids, "gene_symbol": gene_syms})
    var.index = make_index_unique(pd.Index(gene_syms))
    a = ad.AnnData(mat, obs=obs, var=var)
    a = a[a.obs["assigned"].to_numpy()].copy()
    # Same preprocessing as script 37: filter, normalize, log1p, then restrict to basis genes.
    sc.pp.filter_cells(a, min_counts=200)
    sc.pp.filter_genes(a, min_cells=10)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    gene_index = {g: i for i, g in enumerate(a.var_names)}
    idx = [gene_index[g] for g in basis_gene_names if g in gene_index]
    ctrl_mask = a.obs["is_control"].to_numpy()
    print(f"    control cells: {int(ctrl_mask.sum())}; basis genes present: {len(idx)}/{len(basis_gene_names)}")
    X = a[ctrl_mask].X[:, idx]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    sigma_ctrl_partial = np.cov(X, rowvar=False).astype(np.float32)
    if len(idx) < len(basis_gene_names):
        full = np.zeros((len(basis_gene_names), len(basis_gene_names)), dtype=np.float32)
        present_positions = [i for i, g in enumerate(basis_gene_names) if g in gene_index]
        for i_local, i_full in enumerate(present_positions):
            full[i_full, present_positions] = sigma_ctrl_partial[i_local]
        sigma_ctrl = full
    else:
        sigma_ctrl = sigma_ctrl_partial
    np.savez_compressed(cache_path, sigma_ctrl=sigma_ctrl, gene_names=np.array(basis_gene_names))
    print(f"    saved Σ_ctrl {sigma_ctrl.shape} to {cache_path}")
    return sigma_ctrl


# ==============================================================================
# Footprint-U construction and per-screen driver
# ==============================================================================

def _footprint_U(sigma_ctrl, W, kappas_by_guide, guide_targets, gene_names):
    """u_g = -κ_g · Wᵀ · Σ_ctrl · δ_g. Returns U (d × m) and a boolean mask
    marking guides whose target was found in gene_names (others get zero column)."""
    gene_index = {g: i for i, g in enumerate(gene_names)}
    d = W.shape[1]
    guides = list(kappas_by_guide.keys())
    m = len(guides)
    U = np.zeros((d, m), dtype=np.float64)
    kept = np.zeros(m, dtype=bool)
    for j, gn in enumerate(guides):
        tgt = guide_targets.get(gn) if guide_targets else None
        if tgt is None:
            # Fall back to guide name minus "guide_" prefix, or first underscore token.
            s = str(gn)
            tgt = s[len("guide_"):] if s.startswith("guide_") else s.split("_")[0]
        if tgt in gene_index:
            i_gene = gene_index[tgt]
            foot = sigma_ctrl[:, i_gene]  # Σ_ctrl · δ_g, gene-space column
            U[:, j] = -kappas_by_guide[gn] * (W.T @ foot)
            kept[j] = True
    return U, np.array(guides), kept


def _column_normalize(S, U):
    sn = np.linalg.norm(S, axis=0); un = np.linalg.norm(U, axis=0)
    keep = (sn > 0) & (un > 0)
    return S[:, keep] / sn[keep], U[:, keep] / un[keep], keep


def _linear_truth_matched(U, target_of_g, sigma, seed_base, n_reps=N_REPS_LIN_SIM):
    """Matched-SNR linear-truth simulation on the given U (footprint or current)."""
    d, m = U.shape
    S_med_obs = None  # will be filled by caller if used; here we compute α from a canonical scale
    # α_S set so median col-norm of noise-free S matches the SCREEN's real S median.
    # The caller passes sigma; α_S is looked up externally to keep this function pure.
    raise NotImplementedError  # placeholder — actual α_S derivation handled in run_one_screen


def _run_screen(name, S_obs, U_current, U_footprint, kept_guides,
                target_of_g, kappa_seq_footprint, sigma, alpha_S_current, alpha_S_footprint,
                seed_base=SEED):
    """Run the four preregistered analyses on one screen."""
    d = S_obs.shape[0]
    result = {"screen": name, "n_guides_footprint": int(U_footprint.shape[1]),
              "sigma": sigma}

    # Baseline (a) — current encoding: already reported in Table 1 per Methods §4.5,
    # but recomputed here on the same (target_of_g) to keep the comparison local.
    print(f"  [{name}] (a) current-encoding nested-CV baseline …")
    real_current = _nested_rho(S_obs, U_current, target_of_g, seed=seed_base)
    result["current_encoding_real"] = real_current

    # Primary — footprint encoding
    print(f"  [{name}] primary footprint-encoding nested-CV …")
    real_foot = _nested_rho(S_obs, U_footprint, target_of_g, seed=seed_base)
    result["footprint_real"] = real_foot

    # Direction-only footprint
    print(f"  [{name}] direction-only footprint nested-CV …")
    Sn, Un, keep2 = _column_normalize(S_obs, U_footprint)
    dir_out = _nested_rho(Sn, Un, target_of_g[keep2], seed=seed_base + 1)
    result["footprint_direction_only"] = dir_out

    # (b) Shuffled-footprint null
    print(f"  [{name}] (b) shuffled-footprint null (N={N_PERM}) …")
    perm_rhos = []
    for r in range(N_PERM):
        rng = np.random.default_rng(seed_base + 100 + r)
        perm = rng.permutation(U_footprint.shape[1])
        U_perm = U_footprint[:, perm]
        perm_rhos.append(_nested_rho(S_obs, U_perm, target_of_g, seed=seed_base + 100 + r)["rho_pooled"])
    result["shuffled_null"] = {
        "n_perm": N_PERM,
        "mean": float(np.mean(perm_rhos)),
        "std": float(np.std(perm_rhos, ddof=1)),
        "p025": float(np.percentile(perm_rhos, 2.5)),
        "p50": float(np.percentile(perm_rhos, 50)),
        "p975": float(np.percentile(perm_rhos, 97.5)),
        "min": float(np.min(perm_rhos)),
        "max": float(np.max(perm_rhos)),
        "all_rhos": perm_rhos,
    }

    # (c) Random-direction null: unit vectors rescaled to footprint U's column norms
    print(f"  [{name}] (c) random-direction null (N={N_RAND}) …")
    col_norms = np.linalg.norm(U_footprint, axis=0)
    rand_rhos = []
    for r in range(N_RAND):
        rng = np.random.default_rng(seed_base + 200 + r)
        R = rng.standard_normal(size=U_footprint.shape)
        R /= np.linalg.norm(R, axis=0, keepdims=True)
        R *= col_norms
        rand_rhos.append(_nested_rho(S_obs, R, target_of_g, seed=seed_base + 200 + r)["rho_pooled"])
    result["random_direction_null"] = {
        "n_draws": N_RAND,
        "mean": float(np.mean(rand_rhos)),
        "std": float(np.std(rand_rhos, ddof=1)),
        "p025": float(np.percentile(rand_rhos, 2.5)),
        "p50": float(np.percentile(rand_rhos, 50)),
        "p975": float(np.percentile(rand_rhos, 97.5)),
        "all_rhos": rand_rhos,
    }

    # (d) Matched-SNR linear-truth on footprint U with re-derived α_S
    print(f"  [{name}] (d) matched-SNR linear-truth on footprint U (N={N_REPS_LIN_SIM}) …")
    S_med_foot = float(np.median(np.linalg.norm(S_obs, axis=0)))
    # α_S so median col-norm of -J^{-1} U_footprint matches S_med_foot for the screen.
    # Use one reference draw to calibrate.
    rng0 = np.random.default_rng(seed_base)
    J_ref0 = rng0.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
    S_ref = -np.linalg.solve(J_ref0, U_footprint)
    ref_med = float(np.median(np.linalg.norm(S_ref, axis=0)))
    alpha_S_foot_data = S_med_foot / max(ref_med, 1e-30)
    lin_rhos = []
    for r in range(N_REPS_LIN_SIM):
        rng = np.random.default_rng(seed_base + 400 + r)
        J_ref = rng.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
        J = J_ref / max(alpha_S_foot_data, 1e-30)
        S_true = -np.linalg.solve(J, U_footprint)
        S_lin = S_true + sigma * rng.normal(size=S_true.shape)
        lin_rhos.append(_nested_rho(S_lin, U_footprint, target_of_g, seed=seed_base + 400 + r)["rho_pooled"])
    result["footprint_linear_truth_matched"] = {
        "n_reps": N_REPS_LIN_SIM,
        "alpha_S": float(alpha_S_foot_data),
        "rho_mean": float(np.mean(lin_rhos)),
        "rho_std": float(np.std(lin_rhos, ddof=1)),
    }

    # Verdict per preregistered rule
    foot_rho = real_foot["rho_pooled"]
    p025 = result["shuffled_null"]["p025"]
    passes_threshold = foot_rho <= SUCCESS_RHO_THRESHOLD
    below_p025 = foot_rho < p025
    result["verdict"] = {
        "foot_rho": float(foot_rho),
        "threshold_0_90_passed": bool(passes_threshold),
        "below_2p5_percentile_shuffled_null": bool(below_p025),
        "success": bool(passes_threshold and below_p025),
    }
    print(f"  [{name}] verdict: foot_rho={foot_rho:.4f}, "
          f"threshold_pass={passes_threshold}, below_p025={below_p025} (p025={p025:.4f}) — "
          f"success={result['verdict']['success']}")

    return result


# ==============================================================================
# Driver
# ==============================================================================

REPLOGLE = {
    "K562_top200": {
        "pkl": "k562_essential_measurement.pkl",
        "h5ad": "K562_essential_normalized_singlecell_01.h5ad",
        "sigma_cache": "k562_top200_sigma_ctrl.npz",
        "sigma": 0.240,
        "alpha_S": 369.0,
    },
    "K562_random200": {
        "pkl": "k562_random200_measurement.pkl",
        "h5ad": "K562_essential_normalized_singlecell_01.h5ad",
        "sigma_cache": "k562_random200_sigma_ctrl.npz",  # already exists
        "sigma": 0.240,
        "alpha_S": 165.5,
    },
    "RPE1": {
        "pkl": "rpe1_essential_measurement.pkl",
        "h5ad": "rpe1_normalized_singlecell_01.h5ad",
        "sigma_cache": "rpe1_sigma_ctrl.npz",
        "sigma": 0.352,
        "alpha_S": 199.0,
    },
}

JOST = {
    "Jost_2020": {
        "pkl": "jost_measurement.pkl",
        "sigma_cache": "jost_sigma_ctrl.npz",
        "sigma": 0.066,
        "alpha_S": 29.6,
    },
}


def _load_measurement(pkl_name):
    with (RESULTS / pkl_name).open("rb") as f:
        return pickle.load(f)


def _guide_targets_map(meas):
    stored = getattr(meas.report, "guide_targets", None)
    if stored:
        return {str(k): str(v) for k, v in dict(stored).items()}
    out = {}
    for g in meas.report.retained_guides:
        s = str(g)
        out[s] = s[len("guide_"):] if s.startswith("guide_") else s.split("_")[0]
    return out


def run_screen(name, cfg, kind):
    print(f"\n### {name}")
    bundle = _load_measurement(cfg["pkl"])
    meas = bundle["measurement"]
    basis = bundle["basis"]
    W = np.asarray(basis.loadings)  # (n_genes, d)
    S_obs = meas.S
    U_current = meas.U
    guide_targets = _guide_targets_map(meas)
    guides = list(meas.report.retained_guides)
    kappas = {str(g): float(meas.report.guide_efficiencies[g]) for g in guides}

    cache_path = RESULTS / cfg["sigma_cache"]
    if kind == "replogle":
        h5ad_path = DATA / cfg["h5ad"]
        sigma_ctrl = _compute_sigma_ctrl_replogle(h5ad_path, list(basis.gene_names), None, cache_path)
    else:
        sigma_ctrl = _compute_sigma_ctrl_jost(list(basis.gene_names), cache_path)

    # Build footprint U on the same guide order as the pkl
    U_foot, guides_arr, kept = _footprint_U(sigma_ctrl, W, kappas, guide_targets, list(basis.gene_names))
    print(f"  footprint U built: {int(kept.sum())}/{U_foot.shape[1]} guides had target in gene_names")
    # Restrict all matrices to guides whose target is in gene_names
    S_obs_k = S_obs[:, kept]
    U_current_k = U_current[:, kept]
    U_foot_k = U_foot[:, kept]
    target_of_g = np.array([str(guide_targets.get(str(g), str(g))) for g in np.array(guides)[kept]])

    return _run_screen(name, S_obs_k, U_current_k, U_foot_k, kept,
                       target_of_g, kappas, cfg["sigma"], cfg["alpha_S"], None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional list of screen names to run (default: all four).")
    args = ap.parse_args()

    all_screens = [(name, cfg, "replogle") for name, cfg in REPLOGLE.items()] + \
                  [(name, cfg, "jost") for name, cfg in JOST.items()]
    if args.only:
        all_screens = [(n, c, k) for (n, c, k) in all_screens if n in args.only]

    payload = {"seed": SEED, "n_perm": N_PERM, "n_rand": N_RAND,
               "n_reps_lin_sim": N_REPS_LIN_SIM, "rank_grid": RANK_GRID,
               "success_threshold_rho": SUCCESS_RHO_THRESHOLD,
               "success_percentile": SUCCESS_PERCENTILE, "screens": {}}
    for name, cfg, kind in all_screens:
        result = run_screen(name, cfg, kind)
        payload["screens"][name] = result

    # Trim large arrays before saving to keep JSON size reasonable.
    for s in payload["screens"].values():
        for k in ("shuffled_null", "random_direction_null"):
            arr = s[k].pop("all_rhos", None)
            if arr is not None:
                s[k]["all_rhos_sha256"] = None  # placeholder — omit raw array from JSON
                s[k]["all_rhos_min5"] = sorted(arr)[:5]
                s[k]["all_rhos_max5"] = sorted(arr)[-5:]

    out_path = OUT / "F_footprint.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {out_path}")

    # Final verdict summary
    print("\n=== SUMMARY ===")
    succ = 0
    for name, s in payload["screens"].items():
        v = s["verdict"]
        print(f"  {name}: foot_rho={v['foot_rho']:.4f}  threshold_pass={v['threshold_0_90_passed']}  "
              f"below_p025={v['below_2p5_percentile_shuffled_null']}  success={v['success']}")
        if v["success"]: succ += 1
    print(f"\n# successes: {succ}/{len(payload['screens'])}")
    if succ >= 2:
        print("  → Decision rule: projection mechanism becomes a Result.")
    elif succ == 0:
        print("  → Decision rule: footprint encoding does not rescue held-out prediction; projection failure stays open.")
    else:
        print("  → Decision rule: mixed outcome; report per screen.")


if __name__ == "__main__":
    main()
