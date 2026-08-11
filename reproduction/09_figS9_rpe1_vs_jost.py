"""Fig S9: RPE1 essential vs Jost 2020 mismatched-sgRNA titration.

Runs anchor-op pipeline on Jost's dose-response Perturb-seq screen (GSE132080),
and produces a side-by-side comparison with the RPE1 essential measurement
(bundle loaded from ../results/).

Data dependencies:
  - `examples/data/rpe1_normalized_singlecell_01.h5ad` (Replogle 2022 RPE1 essential)
  - `examples/data/jost2020/GSE132080_10X_{matrix,barcodes,genes}.tsv.gz`
  - `examples/data/jost2020/GSE132080_{cell_identities,sgRNA_barcode_sequences_and_phenotypes}.csv.gz`

Jost data can be downloaded from GEO:
  curl -O https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132080/suppl/GSE132080_{10X_matrix.mtx.gz,10X_barcodes.tsv.gz,10X_genes.tsv.gz,cell_identities.csv.gz,sgRNA_barcode_sequences_and_phenotypes.csv.gz}

Runtime ~10 min.
"""
import warnings; warnings.filterwarnings("ignore")
import gzip
import pickle
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import anchorop as ao
from anchorop.identifiability import regularized_pseudoinverse
from sklearn.decomposition import PCA

JOST_DIR = Path(__file__).resolve().parents[1] / "examples/data/jost2020"
OUT_FIG = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_RES = Path(__file__).resolve().parents[1] / "results"
OUT_FIG.mkdir(exist_ok=True); OUT_RES.mkdir(exist_ok=True)

D = 30
N_HVG = 2000
N_MIN_CELLS_SGRNA = 20

# Load Jost
print("Loading Jost 2020 (GSE132080)...")
with gzip.open(JOST_DIR / "GSE132080_10X_matrix.mtx.gz", "rt") as f:
    X = sio.mmread(f).tocsc().T.tocsr()  # cells × genes
with gzip.open(JOST_DIR / "GSE132080_10X_barcodes.tsv.gz", "rt") as f:
    barcodes = np.array([line.strip() for line in f])
with gzip.open(JOST_DIR / "GSE132080_10X_genes.tsv.gz", "rt") as f:
    gene_df = pd.read_csv(f, sep="\t", header=None, names=["ensembl", "symbol"])
    gene_names = gene_df["symbol"].to_numpy()
sg = pd.read_csv(JOST_DIR / "GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz")
ci = pd.read_csv(JOST_DIR / "GSE132080_cell_identities.csv.gz")

def to_sg(gid):
    if pd.isna(gid): return None
    if "non-targeting" in gid or "neg_ctrl" in gid: return None
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
gid = ci_sub["guide_identity"].to_numpy()
sgn = ci_sub["sgRNA_name"].to_numpy()
is_ctrl = pd.Series(gid).str.contains("non-targeting|neg_ctrl", na=False).to_numpy()
print(f"  ctrl {is_ctrl.sum()} + pert {(~is_ctrl).sum()} cells, {len(sg)} sgRNAs, {sg['gene'].nunique()} targets")

# Normalize + HVG + include target genes
X_dense = X.toarray().astype(np.float32)
counts = X_dense.sum(axis=1)
X_norm = np.log1p(X_dense * (1e4 / np.maximum(counts, 1))[:, None])
top_hvg = np.argsort(-X_norm.var(axis=0))[:N_HVG]
target_gene_idx = np.array([i for i, g in enumerate(gene_names) if g in set(sg["gene"])])
keep_feats = np.union1d(top_hvg, target_gene_idx)
X_hvg = X_norm[:, keep_feats]
var_names_hvg = gene_names[keep_feats]

# Per-sgRNA cell groups
sgrna_cells = defaultdict(list)
for i, s in enumerate(sgn):
    if s is None or is_ctrl[i]: continue
    if s in sg_to_gene and s in sg_to_activity and not np.isnan(sg_to_activity[s]):
        sgrna_cells[s].append(i)

# Sweep d × basis_scope: hold-out ρ and per-target linearity
def analyze_target(delta_zs, kappas):
    norms = np.linalg.norm(delta_zs, axis=1)
    if norms.min() < 1e-12 or norms.std() < 1e-12: return None
    unit = delta_zs / norms[:, None]
    gram = unit @ unit.T
    off = gram[~np.eye(len(delta_zs), dtype=bool)]
    slope, intercept = np.polyfit(kappas, norms, 1)
    ss_tot = float(np.sum((norms - norms.mean())**2))
    ss_res = float(np.sum((norms - (slope*kappas + intercept))**2))
    r2_free = 1 - ss_res / max(ss_tot, 1e-12)
    sp, _ = spearmanr(kappas, norms)
    return {"cos": float(off.mean()), "r2_free": r2_free, "spearman": float(sp)}

results_jost = {}
for d in [5, 10, 20, 30]:
    for scope in ["controls_only", "all_cells"]:
        X_basis = X_hvg[is_ctrl] if scope == "controls_only" else X_hvg
        center = X_basis.mean(0)
        pca = PCA(n_components=d, random_state=0).fit(X_basis - center)
        W = pca.components_.T
        def to_prog(x): return (x - center) @ W
        z_ctrl_mean = to_prog(X_hvg[is_ctrl]).mean(0)
        target_to_sg = defaultdict(list)
        for s in sgrna_cells:
            target_to_sg[sg_to_gene[s]].append(s)
        per_t = []
        for tgt, srs in target_to_sg.items():
            if len(srs) < 3: continue
            ks, dzs = [], []
            for s in srs:
                cells = sgrna_cells[s]
                if len(cells) < N_MIN_CELLS_SGRNA: continue
                ks.append(float(sg_to_activity[s]))
                dzs.append(to_prog(X_hvg[cells]).mean(0) - z_ctrl_mean)
            if len(dzs) < 3: continue
            r = analyze_target(np.array(dzs), np.array(ks))
            if r: per_t.append(r)
        cos_m = float(np.median([r["cos"] for r in per_t])) if per_t else float("nan")
        r2_m = float(np.median([r["r2_free"] for r in per_t])) if per_t else float("nan")
        # Operator + held-out ρ (aggregated per-sgRNA)
        S_cols, U_cols, names, effs = [], [], [], {}
        for s, cells in sgrna_cells.items():
            if len(cells) < N_MIN_CELLS_SGRNA: continue
            tgt = sg_to_gene[s]
            tidx = np.where(var_names_hvg == tgt)[0]
            if len(tidx) != 1: continue
            k = float(np.clip(sg_to_activity[s], 0.0, 1.0))
            if k < 0.02: continue
            dz = to_prog(X_hvg[cells]).mean(0) - z_ctrl_mean
            loading = W[int(tidx[0])]
            if np.linalg.norm(loading) < 1e-8: continue
            S_cols.append(dz); U_cols.append(-k * loading); names.append(s); effs[s] = k
        if len(S_cols) < d + 2:
            rho = float("nan"); cond = float("nan"); rank = 0
        else:
            S = np.column_stack(S_cols); U = np.column_stack(U_cols)
            m = ao.measure_from_sensitivity(
                S, U, guide_names=tuple(names), guide_efficiencies=effs,
                reg="tsvd", reg_param="path", rank_tol=1e-2)
            rho = ao.held_out_prediction_check(m, n_folds=5, seed=0, n_permutation_null=15).rho_pooled
            cond = float(m.report.condition_number); rank = int(m.report.effective_response_rank)
        results_jost[f"d{d}_{scope}"] = {"cos": cos_m, "r2_free": r2_m, "rho": rho,
                                          "cond": cond, "rank": rank, "n_guides": len(S_cols)}
        print(f"  d={d} {scope}: cos={cos_m:+.3f} r2_free={r2_m:+.3f} ρ={rho:.3f}")

# Load RPE1 for the comparison (needs the bundle from script 04)
rpe1_bundle = OUT_RES / "rpe1_essential_measurement.pkl"
if rpe1_bundle.exists():
    with rpe1_bundle.open("rb") as f:
        rpe1 = pickle.load(f)
    rpe1_linearity = rpe1["linearity"]
    print(f"\nRPE1 (from bundle): rel_diff={rpe1_linearity.relative_difference:.3f}, "
          f"ρ available from bundle")
else:
    print(f"\nWARNING: {rpe1_bundle} not found — skipping RPE1 side of comparison")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
ds = [5, 10, 20, 30]; scopes = ["controls_only", "all_cells"]
for ax, metric, ylabel in [(axes[0], "cos", "direction cosine"),
                             (axes[1], "r2_free", r"$R^2_{\rm free}$"),
                             (axes[2], "rho", r"held-out $\rho$")]:
    for scope, color in [("controls_only", "#c65a30"), ("all_cells", "#1f4e79")]:
        vals = [results_jost[f"d{d}_{scope}"][metric] for d in ds]
        ax.plot(ds, vals, "o-", color=color, lw=2, ms=8, label=f"Jost {scope}")
    ax.set_xlabel("d"); ax.set_ylabel(ylabel)
    if metric == "cos": ax.set_title("(a) direction cosine (Jost per-target)"); ax.set_ylim(0, 1.05)
    if metric == "r2_free": ax.set_title(r"(b) $R^2_{\rm free}$ (Jost per-target)"); ax.set_ylim(0, 1.05)
    if metric == "rho": ax.set_title(r"(c) global held-out $\rho$"); ax.set_ylim(0, 1.3); ax.axhline(1.0, ls=":", color="gray")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("Fig S9: Jost 2020 mismatched-sgRNA titration diagnostics (RPE1 comparison in text)",
             fontsize=11.5, y=1.02)
out = OUT_FIG / "figS9_rpe1_vs_jost.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out}")
(OUT_FIG / "jost_linearity.json").write_text(json.dumps(results_jost, indent=2))
