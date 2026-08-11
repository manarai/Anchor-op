"""Fig 2a/b: K562 84K noncoding-element aggregate — worked example of the estimator filter.

Loads the K562 aggregate raw 10x + CRISPR analysis, runs the full pipeline with
`min_control_detection_rate=0.05`, and produces:
  fig2a_k562_aggregate_diagnostics.png — diagnostic panel (partial rank 11/30)
  fig2b_efficiency_comparison.png — side-by-side estimator comparison histograms

Data dependency: local K562 aggregate 10x tarballs. Set DATA_ROOT below to
point at your local copy. The Replogle K562 raw data is available from the
Weissman lab / GEO under the aggregate 10x design.

Runtime ~5 min.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import pickle
from pathlib import Path
import numpy as np
import anchorop as ao

# EDIT THIS PATH to point at your local K562 aggregate:
DATA_ROOT = Path(os.environ.get(
    "K562_AGG_DATA_ROOT",
    "/Users/terooatt/Documents/Project_scQDiff/02_scQDiff/scIDIFF_anndata/data/pertubeseq_10x"
))
OUT_FIG = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_RES = Path(__file__).resolve().parents[1] / "results"

if not (DATA_ROOT / "extracted" / "filtered_feature_bc_matrix").exists():
    raise SystemExit(
        f"K562 aggregate 10x data not found at {DATA_ROOT}.\n"
        "Set K562_AGG_DATA_ROOT env var to your local path, or edit this script."
    )

MTX_DIR = DATA_ROOT / "extracted" / "filtered_feature_bc_matrix"
CRISPR_DIR = DATA_ROOT / "extracted" / "crispr_analysis"
NT_LABEL = "Non-Targeting"
MIN_CELLS_PER_TARGET = 60
MIN_CELLS_PER_GUIDE = 30
N_TARGETS_KEEP = 120

import scanpy as sc
import pandas as pd

print("Loading K562 aggregate 10x + CRISPR analysis...")
adata_full = sc.read_10x_mtx(str(MTX_DIR), gex_only=False, var_names="gene_symbols", cache=True)
adata_full.var_names_make_unique()
is_gex = (adata_full.var["feature_types"] == "Gene Expression").values
adata = adata_full[:, is_gex].copy()

feat = pd.read_csv(CRISPR_DIR / "feature_reference.csv")
gt = dict(zip(feat["id"], feat["target_gene_name"]))
calls = pd.read_csv(CRISPR_DIR / "protospacer_calls_per_cell.csv")
single = calls[calls["num_features"] == 1].copy()
single["target"] = single["feature_call"].map(gt)
single = single[single["target"].notna() & (single["target"] != "Ignore")]

# Pick top-N targets
counts = single["target"].value_counts()
qual = counts[counts >= MIN_CELLS_PER_TARGET].index.tolist()
non_nt = [t for t in qual if t != NT_LABEL][:N_TARGETS_KEEP - 1]
keep = {NT_LABEL, *non_nt}
single_kept = single[single["target"].isin(keep)]
cell_to_target = dict(zip(single_kept["cell_barcode"], single_kept["target"]))
cell_to_guide = dict(zip(single_kept["cell_barcode"], single_kept["feature_call"]))
mask = adata.obs_names.isin(cell_to_target)
adata = adata[mask].copy()
adata.obs["target_gene"] = adata.obs_names.map(cell_to_target).astype(str)
adata.obs["guide"] = adata.obs_names.map(cell_to_guide).astype(str)
nt_cells = adata.obs["target_gene"] == NT_LABEL
adata.obs.loc[nt_cells, "guide"] = NT_LABEL
adata.obs.loc[nt_cells, "target_gene"] = ""
# Aggregate to target level
perturbed = ~nt_cells
adata.obs.loc[perturbed, "guide"] = "guide_" + adata.obs.loc[perturbed, "target_gene"].astype(str)

# QC + normalization
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
adata = adata[(adata.obs["pct_counts_mt"] < 20) & (adata.obs["total_counts"] > 500)].copy()
adata.layers["raw_counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")

# HVG + target genes
target_syms = {t for t in adata.obs["target_gene"].astype(str).unique() if t}
in_matrix = target_syms & set(adata.var_names)
retain = adata.var["highly_variable"].to_numpy() | adata.var_names.isin(in_matrix)
adata_hvg = adata[:, retain].copy()

# Basis (cNMF)
control_mask = (adata_hvg.obs["target_gene"] == "").to_numpy()
basis = ao.fit_programs(adata_hvg, d=30, method="cnmf", control_mask=control_mask,
                         n_seeds=5, seed=20260729, max_iter=200)

# Measurement — this is where the estimator regime shows its filtering
measurement = ao.measure_operator(
    adata_hvg, basis,
    guide_key="guide", target_key="target_gene", control_label=NT_LABEL,
    min_cells_per_guide=MIN_CELLS_PER_GUIDE, min_knockdown_efficiency=0.05,
    reg="tsvd", reg_param="path", rank_tol=1e-2,
    bootstrap=50, bootstrap_seed=20260729, state_label="K562",
)
r = measurement.report
print(f"retained {r.n_guides_retained}/{r.n_guides_input}, rank {r.effective_response_rank}/{r.d}")
print(f"cond number {r.condition_number:.2f}")

# Save bundle
OUT_RES.mkdir(exist_ok=True)
with (OUT_RES / "k562_measurement.pkl").open("wb") as f:
    pickle.dump({"measurement": measurement, "basis": basis}, f)

# Figures
OUT_FIG.mkdir(exist_ok=True)
report_dict = ao.analyses.measurement_report(measurement, save_dir=str(OUT_FIG))
for src_name, dst_name in [("diagnostics", "fig2a_k562_aggregate_diagnostics.png"),
                             ("guide_drops", "fig2c_k562_aggregate_drops.png")]:
    src = OUT_FIG / f"{src_name}.png"
    if src.exists():
        src.rename(OUT_FIG / dst_name)
        print(f"wrote {OUT_FIG / dst_name}")

# Fig 2b: efficiency estimator comparison (mean_ratio vs detection_rate on the same targets)
# Reload raw counts for the estimator comparison
print("\nBuilding efficiency-comparison figure (2b)...")
import scanpy as sc
adata_raw = sc.read_10x_mtx(str(MTX_DIR), gex_only=True, var_names="gene_symbols", cache=True)
adata_raw.var_names_make_unique()
X_raw = adata_raw.X.toarray() if hasattr(adata_raw.X, "toarray") else np.asarray(adata_raw.X)
gene_idx = {g: i for i, g in enumerate(adata_raw.var_names)}
nt_bc = set(single.loc[single["target"] == NT_LABEL, "cell_barcode"])
nt_mask_full = np.asarray(adata_raw.obs_names.isin(nt_bc))

def one_target(t):
    if t not in gene_idx: return None
    col = X_raw[:, gene_idx[t]]
    pert = np.asarray(adata_raw.obs_names.isin([bc for bc, tgt in cell_to_target.items() if tgt == t]))
    if pert.sum() < 10: return None
    return {
        "mean_ratio": float(np.clip(1.0 - col[pert].mean() / max(col[nt_mask_full].mean(), 1e-8), 0.0, 1.0)),
        "detection_rate": float(np.clip((col[nt_mask_full] > 0).mean() - (col[pert] > 0).mean(), 0.0, 1.0)),
    }

both = [b for b in (one_target(t) for t in non_nt) if b is not None]
eff_mr = np.array([b["mean_ratio"] for b in both])
eff_dr = np.array([b["detection_rate"] for b in both])
print(f"  mean_ratio ≥0.99: {(eff_mr >= 0.99).sum()}/{len(both)}")
print(f"  detection_rate ≥0.99: {(eff_dr >= 0.99).sum()}/{len(both)}")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, vals, name, color in [(axes[0], eff_mr, "mean_ratio", "#c65a30"),
                                (axes[1], eff_dr, "detection_rate", "#1f4e79")]:
    ax.hist(vals, bins=25, color=color, alpha=0.85)
    ax.axvline(0.99, ls=":", color="red")
    ax.set_xlabel("efficiency estimate"); ax.set_ylabel("targets")
    ax.set_title(f"{name}  (n({name}≥0.99) = {(vals >= 0.99).sum()})")
fig.suptitle(f"Fig 2b: K562 aggregate efficiency estimators on {len(both)} candidate targets", fontsize=11, y=1.05)
out = OUT_FIG / "fig2b_efficiency_comparison.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out}")
