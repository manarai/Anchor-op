"""Fig S19: independent per-dataset σ estimation for K562, RPE1, and Jost 2020.

The paper's noise anchor σ = 0.266 is measured on K562 essential and applied to
RPE1 and Jost without independent estimation. This script runs the same
within-guide cell-level split-half bootstrap on RPE1 essential and Jost 2020,
so §2.2 and §2.7 can be reported at each dataset's own σ.

Method (identical to Methods §4.4):
  For each dataset:
    1. Build a d=30 program basis from NT controls.
    2. For each target with ≥20 cells:
       - Randomly split cells into two equal halves.
       - Compute half-Δz vectors d₁, d₂ in the program basis (each half's mean
         minus the full-control mean).
       - Estimate per-entry std of the full-data (N cells) Δz as
         ‖d₁ − d₂‖_F / (2√d) (the √2-correction from §4.4).
    3. Report the median across targets.

Also reports: median cells/guide, the per-cell z-std of NT controls
(σ_percell), and the direct-model prediction σ_percell/√N to make the
1.47× discrepancy (K562) visible for each dataset.

Runtime ~5 min.
"""
import warnings; warnings.filterwarnings("ignore")
import gzip, json, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import anchorop as ao

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "examples/data"
OUT_DIR = ROOT / "manuscript_figures"
RESULTS = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

D = 30
N_HVG = 3000
N_MIN_CELLS_PER_TARGET = 20
SEED = 20260810


def bootstrap_sigma(Z, guide_labels, ctrl_label, min_cells=N_MIN_CELLS_PER_TARGET, seed=SEED):
    """Split-half bootstrap on (Z, labels). Returns per-target sigma estimates."""
    rng = np.random.default_rng(seed)
    ctrl_mask = np.array([g == ctrl_label for g in guide_labels])
    z_ctrl_mean = Z[ctrl_mask].mean(axis=0)
    d = Z.shape[1]
    per_target = []
    unique = list(set(guide_labels))
    for g in unique:
        if g == ctrl_label:
            continue
        idx = np.where([lbl == g for lbl in guide_labels])[0]
        if len(idx) < min_cells:
            continue
        rng.shuffle(idx)
        half = len(idx) // 2
        d1 = Z[idx[:half]].mean(axis=0) - z_ctrl_mean
        d2 = Z[idx[half:2*half]].mean(axis=0) - z_ctrl_mean
        # per-entry std of full-data Δz = ‖d₁-d₂‖_F / (2√d)
        sigma_full = float(np.linalg.norm(d1 - d2) / (2 * np.sqrt(d)))
        per_target.append({"target": g, "n_cells": int(len(idx)), "sigma": sigma_full})
    return per_target


def per_cell_z_std(Z, guide_labels, ctrl_label):
    ctrl_mask = np.array([g == ctrl_label for g in guide_labels])
    Z_ctrl = Z[ctrl_mask]
    z_ctrl_mean = Z_ctrl.mean(axis=0)
    per_cell_dev = Z_ctrl - z_ctrl_mean
    return float(per_cell_dev.std())


def project_h5ad(h5ad_path, d=D):
    """Load an h5ad, fit d-dim PCA on NT controls, project all cells."""
    print(f"  loading {h5ad_path.name}...")
    adata = ao.load_replogle_h5ad(str(h5ad_path))
    guide_labels = adata.obs["gene"].astype(str).to_list()
    ctrl_label = "non-targeting"
    # HVG selection on log-normalized counts (adata is already normalized by load_replogle_h5ad)
    X = adata.X
    if sp.issparse(X):
        # Compute variance without densifying full matrix
        n = X.shape[0]
        mean = np.asarray(X.mean(axis=0)).flatten()
        sq_mean = np.asarray(X.multiply(X).mean(axis=0)).flatten()
        var = sq_mean - mean**2
        top_hvg = np.argsort(-var)[:N_HVG]
        X_hvg = X[:, top_hvg]
        X_arr = np.asarray(X_hvg.todense()) if X_hvg.shape[0] < 400_000 else X_hvg
    else:
        var = X.var(axis=0)
        top_hvg = np.argsort(-var)[:N_HVG]
        X_arr = X[:, top_hvg]
    ctrl_mask = np.array([g == ctrl_label for g in guide_labels])
    if not sp.issparse(X_arr):
        X_ctrl = X_arr[ctrl_mask]
    else:
        X_ctrl = np.asarray(X_arr[ctrl_mask].todense())
    print(f"    {X_ctrl.shape[0]} controls, fitting {d}-dim PCA...")
    center = X_ctrl.mean(axis=0)
    pca = PCA(n_components=d, random_state=SEED).fit(X_ctrl - center)
    W = pca.components_.T
    # Project all cells
    if sp.issparse(X_arr):
        Z = np.asarray((X_arr - center) @ W)
    else:
        Z = (X_arr - center) @ W
    return Z, guide_labels, ctrl_label


def load_jost():
    """Load Jost 2020 as normalized log1p, HVG-selected, program-projected."""
    jd = DATA_DIR / "jost2020"
    print(f"  loading Jost 2020 from {jd.name}...")
    with gzip.open(jd / "GSE132080_10X_matrix.mtx.gz", "rt") as f:
        X = sio.mmread(f).tocsc().T.tocsr()
    with gzip.open(jd / "GSE132080_10X_barcodes.tsv.gz", "rt") as f:
        barcodes = np.array([l.strip() for l in f])
    with gzip.open(jd / "GSE132080_10X_genes.tsv.gz", "rt") as f:
        gene_df = pd.read_csv(f, sep="\t", header=None, names=["ensembl", "symbol"])
        gene_names = gene_df["symbol"].to_numpy()
    ci = pd.read_csv(jd / "GSE132080_cell_identities.csv.gz")
    sg = pd.read_csv(jd / "GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz")

    def to_sg(gid):
        if pd.isna(gid): return None
        if "non-targeting" in gid or "neg_ctrl" in gid: return None
        parts = gid.split("_", 1)
        return parts[1] if len(parts) == 2 else gid
    ci["sgRNA_name"] = ci["guide_identity"].map(to_sg)
    sg_to_gene = dict(zip(sg["sgRNA_name"], sg["gene"]))
    bc_to_ci = {bc: i for i, bc in enumerate(ci["cell_barcode"].to_numpy())}
    in_ci = np.array([b in bc_to_ci for b in barcodes])
    X = X[in_ci, :]
    bcs_kept = barcodes[in_ci]
    ci_sub = ci.iloc[[bc_to_ci[b] for b in bcs_kept]].reset_index(drop=True)
    gid = ci_sub["guide_identity"].to_numpy()

    guide_labels = []
    for i, g in enumerate(gid):
        if isinstance(g, str) and ("non-targeting" in g or "neg_ctrl" in g):
            guide_labels.append("non-targeting")
        else:
            sname = ci_sub["sgRNA_name"].iloc[i]
            # aggregate to TARGET gene rather than per-sgRNA to make cells/guide sufficient
            g_target = sg_to_gene.get(sname, None)
            guide_labels.append(g_target if g_target is not None else "unmapped")

    X_dense = X.toarray().astype(np.float32)
    counts = X_dense.sum(axis=1)
    X_norm = np.log1p(X_dense * (1e4 / np.maximum(counts, 1))[:, None])
    var = X_norm.var(axis=0)
    top = np.argsort(-var)[:N_HVG]
    X_hvg = X_norm[:, top]

    ctrl_mask = np.array([g == "non-targeting" for g in guide_labels])
    X_ctrl = X_hvg[ctrl_mask]
    center = X_ctrl.mean(axis=0)
    print(f"    {X_ctrl.shape[0]} controls, fitting {D}-dim PCA...")
    pca = PCA(n_components=D, random_state=SEED).fit(X_ctrl - center)
    W = pca.components_.T
    Z = (X_hvg - center) @ W
    return Z, guide_labels, "non-targeting"


DATASETS = {
    "K562_essential": (project_h5ad, DATA_DIR / "K562_essential_normalized_singlecell_01.h5ad"),
    "RPE1_essential": (project_h5ad, DATA_DIR / "rpe1_normalized_singlecell_01.h5ad"),
    "Jost_2020":       (load_jost,    None),
}

summary = {}
for name, (loader, arg) in DATASETS.items():
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    Z, labels, ctrl_label = loader(arg) if arg is not None else loader()
    n_cells, d = Z.shape
    print(f"  Z shape: {Z.shape}, ctrl label: {ctrl_label}")
    ctrl_n = sum(1 for g in labels if g == ctrl_label)
    print(f"  {ctrl_n} control cells")

    per_target = bootstrap_sigma(Z, labels, ctrl_label)
    sigmas = np.array([t["sigma"] for t in per_target])
    ncells = np.array([t["n_cells"] for t in per_target])
    percell = per_cell_z_std(Z, labels, ctrl_label)
    median_sigma = float(np.median(sigmas))
    median_n = float(np.median(ncells))
    predicted_sigma = percell / np.sqrt(median_n)
    ratio = median_sigma / predicted_sigma

    print(f"  n_targets with ≥{N_MIN_CELLS_PER_TARGET} cells: {len(per_target)}")
    print(f"  median cells/target: {median_n:.0f}")
    print(f"  per-cell z-std σ_percell (NT controls): {percell:.3f}")
    print(f"  predicted σ = σ_percell/√N: {predicted_sigma:.3f}")
    print(f"  measured σ (bootstrap median): {median_sigma:.3f}")
    print(f"  ratio measured/predicted: {ratio:.2f}x")
    print(f"  σ distribution: p25={np.percentile(sigmas,25):.3f}, p50={median_sigma:.3f}, p75={np.percentile(sigmas,75):.3f}")

    summary[name] = {
        "n_cells_total": int(n_cells),
        "n_control_cells": int(ctrl_n),
        "n_targets_with_ncells_min": int(len(per_target)),
        "median_cells_per_target": float(median_n),
        "sigma_percell_nt": float(percell),
        "sigma_predicted": float(predicted_sigma),
        "sigma_measured_median": float(median_sigma),
        "sigma_measured_p25": float(np.percentile(sigmas, 25)),
        "sigma_measured_p75": float(np.percentile(sigmas, 75)),
        "ratio_measured_over_predicted": float(ratio),
    }

(OUT_DIR / "sigma_bootstrap_all.json").write_text(json.dumps(summary, indent=2))

# Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
ds_names = list(summary.keys())
colors = ["#1f4e79", "#c65a30", "#5a7a3a"]
xs = np.arange(len(ds_names))

ax = axes[0]
measured = [summary[n]["sigma_measured_median"] for n in ds_names]
predicted = [summary[n]["sigma_predicted"] for n in ds_names]
w = 0.35
ax.bar(xs - w/2, measured, w, color=colors, edgecolor="black", label="bootstrap σ")
ax.bar(xs + w/2, predicted, w, color=colors, alpha=0.5, edgecolor="black", hatch="//", label="σ_percell/√N")
for i, name in enumerate(ds_names):
    r = summary[name]["ratio_measured_over_predicted"]
    ax.text(i, max(measured[i], predicted[i]) + 0.02, f"{r:.2f}×", ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(ds_names, rotation=20, ha="right")
ax.set_ylabel("per-entry Δz noise σ")
ax.set_title("(a) per-dataset σ: bootstrap vs σ_percell/√N")
ax.legend(fontsize=9)

ax = axes[1]
for name, color in zip(ds_names, colors):
    ax.bar([name], [summary[name]["median_cells_per_target"]], color=color, edgecolor="black")
    ax.text(name, summary[name]["median_cells_per_target"] + 5,
             f"σ_percell={summary[name]['sigma_percell_nt']:.2f}", ha="center", fontsize=9)
ax.set_ylabel("median cells per target")
ax.set_title("(b) cells/target and per-cell noise per dataset")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

fig.suptitle("Fig S19: Independent per-dataset σ estimation via within-guide split-half bootstrap.",
             fontsize=11, y=1.03)
out = OUT_DIR / "figS19_sigma_bootstrap_all.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")
print("\nsummary:")
print(json.dumps(summary, indent=2))
