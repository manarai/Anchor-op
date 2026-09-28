"""Direct Jost matched-geometry operator-recovery empirical null + paired bootstrap.

Closes the §2.7 audit item: "K562-geometry extrapolation of the Jost recovery
figure was illustrative; a direct simulation under Jost's own U and κ is
outstanding." This script does the direct run.

Pipeline:
  1. Load Jost 2020 (GSE132080) counts, sgRNA phenotypes, cell identities.
  2. Normalize + HVG + include target genes (same recipe as script 09).
  3. Fit a d=30 PCA basis on all cells (all_cells scope; the anchor-op measurement
     of Jost in script 09 uses this).
  4. Build per-sgRNA U columns as -κ_g × W_target_row, matching the pipeline.
  5. For each interaction ensemble ∈ {dense, sparse_10pct, sparse_2pct, low_rank_5}
     draw N=200 J_true = G + shift and simulate S_obs = S_true + σ · noise with
     σ_Jost = 0.08 per sgRNA (variance-scaling approximation from §2.7; a direct
     per-sgRNA bootstrap σ would be preferable, this is what is currently
     available).
  6. Report per-dataset paired-bootstrap cross-replicate and shuffled-U tests
     (cos_full, cos_1, cos_5).

Sensitivity: also run at σ = 0.036 (target-aggregate, best-case for Jost) and
σ = 0.240 (K562 anchor, for direct cross-dataset comparison).

Data dependencies: examples/data/jost2020/ (as in script 09).
Runtime ~15 min (Jost load ~5 min + 4 ensembles × 3 σ × N=200 fits).
"""
import warnings; warnings.filterwarnings("ignore")
import gzip, json, pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
from anchorop.identifiability import regularized_pseudoinverse

JOST_DIR = Path(__file__).resolve().parents[1] / "examples/data/jost2020"
OUT_FIG = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_RES = Path(__file__).resolve().parents[1] / "results"

D = 30
N_HVG = 2000
N_MIN_CELLS_SGRNA = 20
RANK_TOL = 1e-2
STRUCTURES = ["dense", "sparse_10pct", "sparse_2pct", "low_rank_5"]
N_REPS = 200
SEED_BASE = 20260810
# Three σ values:
#   σ=0.036 — Jost target-aggregate σ (per Fig. S19), best-case Jost noise.
#   σ=0.08  — Jost per-sgRNA variance-scaling approximation (§2.7).
#   σ=0.240 — K562 conditional response-noise anchor, for cross-dataset comparison.
SIGMAS = [0.036, 0.08, 0.240]

JOST_U_PICKLE = OUT_RES / "jost_u_at_d30.pkl"


def build_jost_U():
    """Load Jost data and construct the anchor-op U matrix (d=30, all_cells PCA)."""
    print("Loading Jost 2020 (GSE132080) — this can take ~5 min ...")
    with gzip.open(JOST_DIR / "GSE132080_10X_matrix.mtx.gz", "rt") as f:
        X = sio.mmread(f).tocsc().T.tocsr()
    with gzip.open(JOST_DIR / "GSE132080_10X_barcodes.tsv.gz", "rt") as f:
        barcodes = np.array([line.strip() for line in f])
    with gzip.open(JOST_DIR / "GSE132080_10X_genes.tsv.gz", "rt") as f:
        gene_df = pd.read_csv(f, sep="\t", header=None, names=["ensembl", "symbol"])
        gene_names = gene_df["symbol"].to_numpy()
    sg = pd.read_csv(JOST_DIR / "GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz")
    ci = pd.read_csv(JOST_DIR / "GSE132080_cell_identities.csv.gz")

    def to_sg(gid):
        if pd.isna(gid):
            return None
        if "non-targeting" in gid or "neg_ctrl" in gid:
            return None
        parts = gid.split("_", 1)
        return parts[1] if len(parts) == 2 else gid

    ci["sgRNA_name"] = ci["guide_identity"].map(to_sg)
    sg_to_gene = dict(zip(sg["sgRNA_name"], sg["gene"]))
    sg_to_activity = dict(zip(sg["sgRNA_name"], sg["relative_activity_day5"]))
    bc_to_ci = {bc: i for i, bc in enumerate(ci["cell_barcode"].to_numpy())}
    in_ci = np.array([b in bc_to_ci for b in barcodes])
    X = X[in_ci, :]
    bcs_kept = barcodes[in_ci]
    ci_sub = ci.iloc[[bc_to_ci[b] for b in bcs_kept]].reset_index(drop=True)
    gid_col = ci_sub["guide_identity"].to_numpy()
    sgn = ci_sub["sgRNA_name"].to_numpy()
    is_ctrl = pd.Series(gid_col).str.contains("non-targeting|neg_ctrl", na=False).to_numpy()
    print(
        f"  cells: {is_ctrl.sum()} ctrl + {(~is_ctrl).sum()} pert, "
        f"{len(sg)} sgRNAs, {sg['gene'].nunique()} targets"
    )

    X_dense = X.toarray().astype(np.float32)
    counts = X_dense.sum(axis=1)
    X_norm = np.log1p(X_dense * (1e4 / np.maximum(counts, 1))[:, None])
    top_hvg = np.argsort(-X_norm.var(axis=0))[:N_HVG]
    target_gene_idx = np.array([i for i, g in enumerate(gene_names) if g in set(sg["gene"])])
    keep_feats = np.union1d(top_hvg, target_gene_idx)
    X_hvg = X_norm[:, keep_feats]
    var_names_hvg = gene_names[keep_feats]

    sgrna_cells = defaultdict(list)
    for i, s in enumerate(sgn):
        if s is None or is_ctrl[i]:
            continue
        if s in sg_to_gene and s in sg_to_activity and not np.isnan(sg_to_activity[s]):
            sgrna_cells[s].append(i)

    # all_cells PCA basis at d=30
    center = X_hvg.mean(0)
    pca = PCA(n_components=D, random_state=0).fit(X_hvg - center)
    W = pca.components_.T
    z_ctrl_mean = ((X_hvg[is_ctrl] - center) @ W).mean(0)

    U_cols, names, effs = [], [], {}
    for s, cells in sgrna_cells.items():
        if len(cells) < N_MIN_CELLS_SGRNA:
            continue
        tgt = sg_to_gene[s]
        tidx = np.where(var_names_hvg == tgt)[0]
        if len(tidx) != 1:
            continue
        k = float(np.clip(sg_to_activity[s], 0.0, 1.0))
        if k < 0.02:
            continue
        loading = W[int(tidx[0])]
        if np.linalg.norm(loading) < 1e-8:
            continue
        U_cols.append(-k * loading)
        names.append(s)
        effs[s] = k
    U = np.column_stack(U_cols)
    print(f"  Jost U built: d={U.shape[0]}, n_sgRNA={U.shape[1]}")
    return U, names, effs


if JOST_U_PICKLE.exists():
    print(f"Loading cached Jost U from {JOST_U_PICKLE}")
    with JOST_U_PICKLE.open("rb") as f:
        cached = pickle.load(f)
    U_jost, jost_names, jost_effs = cached["U"], cached["names"], cached["effs"]
else:
    U_jost, jost_names, jost_effs = build_jost_U()
    with JOST_U_PICKLE.open("wb") as f:
        pickle.dump({"U": U_jost, "names": jost_names, "effs": jost_effs}, f)
    print(f"Cached Jost U to {JOST_U_PICKLE}")


def draw_J(d, seed, structure):
    rng = np.random.default_rng(seed)
    if structure == "dense":
        G = rng.normal(size=(d, d)) / np.sqrt(d)
        return G - 1.5 * np.eye(d)
    if structure == "sparse_10pct":
        mask = rng.random((d, d)) < 0.10
        G = np.zeros((d, d))
        G[mask] = rng.normal(size=int(mask.sum())) / np.sqrt(max(mask.sum(), 1) / d)
        return G - 1.5 * np.eye(d)
    if structure == "sparse_2pct":
        mask = rng.random((d, d)) < 0.02
        G = np.zeros((d, d))
        G[mask] = rng.normal(size=int(mask.sum())) / np.sqrt(max(mask.sum(), 1) / d)
        return G - 1.5 * np.eye(d)
    if structure == "low_rank_5":
        Ul = rng.normal(size=(d, 5)) / np.sqrt(d)
        Vl = rng.normal(size=(5, d)) / np.sqrt(d)
        return Ul @ Vl - 1.5 * np.eye(d)
    raise ValueError(structure)


def fit_operator(S, U):
    S_pinv, *_ = regularized_pseudoinverse(S, method="tsvd", parameter="path", rank_tol=RANK_TOL)
    return -U @ S_pinv


def cos_full(A, J):
    return float(np.sum(A * J)) / max(np.linalg.norm(A) * np.linalg.norm(J), 1e-30)


def cos_topk(A, J, S_true, k):
    Us, _, _ = np.linalg.svd(S_true, full_matrices=False)
    Uk = Us[:, :k]
    AUk = A @ Uk
    JUk = J @ Uk
    return float(np.sum(AUk * JUk)) / max(np.linalg.norm(AUk) * np.linalg.norm(JUk), 1e-30)


def paired_bootstrap(real_arr, null_paired_arr, n_boot=10_000, seed=42):
    n = len(real_arr)
    diff = np.asarray(real_arr) - np.asarray(null_paired_arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    mean_d = float(diff.mean())
    p_greater = float(np.mean(boot_means <= 0.0))
    p_less = float(np.mean(boot_means >= 0.0))
    p_two = 2.0 * min(p_greater, p_less)
    p_two = max(p_two, 1.0 / n_boot)
    d_z = mean_d / max(float(np.std(diff, ddof=1)), 1e-30)
    return {
        "mean_paired_diff": mean_d,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p_two_sided": p_two,
        "p_one_sided_diff_gt_0": 1.0 - p_greater,
        "d_z": float(d_z),
        "n_pairs": int(n),
        "n_bootstrap": int(n_boot),
    }


def run_one(U_real, sigma, structure):
    d, n_guides = U_real.shape
    Js, S_trues, A_fits = [], [], []
    rng_n = np.random.default_rng(SEED_BASE + 999)
    for r in range(N_REPS):
        J = draw_J(d, seed=SEED_BASE + r, structure=structure)
        S_true = -np.linalg.solve(J, U_real)
        S_obs = S_true + sigma * rng_n.normal(size=S_true.shape)
        Js.append(J); S_trues.append(S_true); A_fits.append(fit_operator(S_obs, U_real))
    real_cos = np.array([cos_full(A_fits[r], Js[r]) for r in range(N_REPS)])
    real_cos1 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 1) for r in range(N_REPS)])
    real_cos5 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 5) for r in range(N_REPS)])
    magr = np.array([np.linalg.norm(A_fits[r]) / np.linalg.norm(Js[r]) for r in range(N_REPS)])
    frob = np.array([np.linalg.norm(A_fits[r] - Js[r]) / np.linalg.norm(Js[r]) for r in range(N_REPS)])

    # Cross-rep null: shift=1 paired; also 10-shift expansion for distributions.
    null_cross_cos, null_cross_cos1, null_cross_cos5 = [], [], []
    paired_cross_cos, paired_cross_cos1, paired_cross_cos5 = [], [], []
    for shift in range(1, 11):
        for r in range(N_REPS):
            rp = (r + shift) % N_REPS
            c = cos_full(A_fits[r], Js[rp])
            c1 = cos_topk(A_fits[r], Js[rp], S_trues[rp], 1)
            c5 = cos_topk(A_fits[r], Js[rp], S_trues[rp], 5)
            null_cross_cos.append(c); null_cross_cos1.append(c1); null_cross_cos5.append(c5)
            if shift == 1:
                paired_cross_cos.append(c); paired_cross_cos1.append(c1); paired_cross_cos5.append(c5)
    # Shuffled-U null (paired by r).
    null_shuf_cos, null_shuf_cos1, null_shuf_cos5 = [], [], []
    rng_su = np.random.default_rng(SEED_BASE + 888)
    for r in range(N_REPS):
        perm = rng_su.permutation(n_guides)
        S_ts = -np.linalg.solve(Js[r], U_real[:, perm])
        S_obs = S_ts + sigma * rng_su.normal(size=S_ts.shape)
        A_sh = fit_operator(S_obs, U_real)
        null_shuf_cos.append(cos_full(A_sh, Js[r]))
        null_shuf_cos1.append(cos_topk(A_sh, Js[r], S_trues[r], 1))
        null_shuf_cos5.append(cos_topk(A_sh, Js[r], S_trues[r], 5))

    reals = {"cos_full": real_cos, "cos_1": real_cos1, "cos_5": real_cos5}
    nc = {"cos_full": null_cross_cos, "cos_1": null_cross_cos1, "cos_5": null_cross_cos5}
    ns = {"cos_full": null_shuf_cos, "cos_1": null_shuf_cos1, "cos_5": null_shuf_cos5}
    paired_c = {"cos_full": np.asarray(paired_cross_cos),
                "cos_1":    np.asarray(paired_cross_cos1),
                "cos_5":    np.asarray(paired_cross_cos5)}
    paired_s = {"cos_full": np.asarray(null_shuf_cos),
                "cos_1":    np.asarray(null_shuf_cos1),
                "cos_5":    np.asarray(null_shuf_cos5)}

    def summ(name):
        r = reals[name]; c = nc[name]; s = ns[name]
        return {
            "real_mean": float(r.mean()), "real_std": float(r.std()),
            "real_SE":   float(r.std() / np.sqrt(len(r))),
            "cross_rep_null_mean": float(np.mean(c)), "cross_rep_null_std": float(np.std(c)),
            "shuf_U_null_mean":    float(np.mean(s)), "shuf_U_null_std":    float(np.std(s)),
            "z_per_rep_cross_rep": float((r.mean() - np.mean(c)) / max(np.std(c), 1e-9)),
            "z_per_rep_shuf_U":    float((r.mean() - np.mean(s)) / max(np.std(s), 1e-9)),
            "paired_bootstrap_cross_rep": paired_bootstrap(r, paired_c[name]),
            "paired_bootstrap_shuf_U":    paired_bootstrap(r, paired_s[name]),
        }

    return {
        "sigma": sigma, "n_reps": N_REPS, "n_guides": int(n_guides),
        "structure": structure,
        "frob_rel_err_mean": float(frob.mean()), "frob_rel_err_std": float(frob.std()),
        "magnitude_ratio_mean": float(magr.mean()), "magnitude_ratio_std": float(magr.std()),
        "cos_full": summ("cos_full"),
        "cos_1":    summ("cos_1"),
        "cos_5":    summ("cos_5"),
    }


def _fmt(pb):
    n_boot = pb["n_bootstrap"]
    p = pb["p_two_sided"]
    p_str = f"p<{1/n_boot:g}" if p <= 1/n_boot else f"p={p:.4f}"
    return (f"paired-diff={pb['mean_paired_diff']:+.4f} "
            f"[95% CI {pb['ci95_lo']:+.4f},{pb['ci95_hi']:+.4f}] d_z={pb['d_z']:+.3f} {p_str}")


results = {}
for sigma in SIGMAS:
    print(f"\n{'='*72}\nJost @ σ={sigma}, U shape={U_jost.shape}, N={N_REPS}\n{'='*72}")
    per_sigma = {}
    for structure in STRUCTURES:
        print(f"\n  {structure} ...")
        row = run_one(U_jost, sigma, structure)
        per_sigma[structure] = row
        s = row["cos_full"]
        print(f"    cos_full  real={s['real_mean']:+.4f}   cross-rep={s['cross_rep_null_mean']:+.4f}   "
              f"shuf-U={s['shuf_U_null_mean']:+.4f}")
        print(f"      cross-rep {_fmt(s['paired_bootstrap_cross_rep'])}")
        s1 = row["cos_1"]
        print(f"    cos_1     real={s1['real_mean']:+.4f}   cross-rep={s1['cross_rep_null_mean']:+.4f}")
        print(f"      cross-rep {_fmt(s1['paired_bootstrap_cross_rep'])}")
    results[f"sigma_{sigma}"] = per_sigma

(OUT_FIG / "jost_matched_geometry_nulls.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT_FIG / 'jost_matched_geometry_nulls.json'}")

print("\n" + "="*72)
print("SUMMARY — Jost full-operator paired-bootstrap cross-rep null")
print("="*72)
print(f"{'sigma':>8s} {'structure':>14s}  {'paired-diff':>12s} {'95% CI':>22s} {'d_z':>7s} {'p':>10s}")
for sig_key in results:
    for st in STRUCTURES:
        pb = results[sig_key][st]["cos_full"]["paired_bootstrap_cross_rep"]
        p = pb["p_two_sided"]
        p_str = f"<{1/pb['n_bootstrap']:g}" if p <= 1/pb["n_bootstrap"] else f"{p:.4f}"
        print(f"{sig_key:>8s} {st:>14s}  {pb['mean_paired_diff']:>+12.4f}  "
              f"[{pb['ci95_lo']:+.4f},{pb['ci95_hi']:+.4f}]  {pb['d_z']:>+7.3f}  {p_str:>10s}")
