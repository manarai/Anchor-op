"""Fig 4a/b: Replogle RPE1 essential-gene measurement + save bundle.

Same code path as 03_fig3_k562_essential.py, applied to the RPE1 h5ad.
Requires `examples/data/rpe1_normalized_singlecell_01.h5ad`. Runtime ~8 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
from pathlib import Path
import numpy as np
import anchorop as ao
from sklearn.decomposition import PCA

DATA = Path(__file__).resolve().parents[1] / "examples/data/rpe1_normalized_singlecell_01.h5ad"
OUT_FIG = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_RES = Path(__file__).resolve().parents[1] / "results"
OUT_FIG.mkdir(exist_ok=True); OUT_RES.mkdir(exist_ok=True)

D_PROGRAMS = 30
N_HVG = 3000
N_TARGETS_KEEP = 200
SEED = 20260729

print("Loading RPE1 essential h5ad...")
adata = ao.load_replogle_h5ad(str(DATA))
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
extra_idx = np.array([i for i, g in enumerate(np.asarray(adata.var_names)) if g in target_syms])
keep_feats = np.union1d(top_hvg, extra_idx)
adata_hvg = adata[:, keep_feats].copy()

ctrl_mask = (adata_hvg.obs["guide"] == "non-targeting").to_numpy()
X_ctrl = adata_hvg.X.toarray() if hasattr(adata_hvg.X, "toarray") else adata_hvg.X
X_ctrl = X_ctrl[ctrl_mask]
pca = PCA(n_components=D_PROGRAMS, random_state=SEED).fit(X_ctrl - X_ctrl.mean(0))
basis = ao.make_program_basis(
    pca.components_.T.astype(np.float32), adata_hvg.var_names,
    method="pca_external", control_count=int(ctrl_mask.sum()), normalize=False,
)

measurement = ao.measure_operator(
    adata_hvg, basis,
    guide_key="guide", target_key="target_gene", control_label="non-targeting",
    min_cells_per_guide=30, min_knockdown_efficiency=0.05,
    reg="tsvd", reg_param="path", rank_tol=1e-2,
    bootstrap=100, bootstrap_seed=SEED, state_label="RPE1_essential",
)
r = measurement.report
print(f"retained {r.n_guides_retained}/{r.n_guides_input}, rank {r.effective_response_rank}/{r.d}, "
      f"cond {r.condition_number:.2f}")

linearity = ao.linearity_check(measurement, threshold=0.25, n_null=200, null_seed=42)
print(f"rel_diff = {linearity.relative_difference:.3f}, null_median = {linearity.null_median:.3f}")

BUNDLE = OUT_RES / "rpe1_essential_measurement.pkl"
with BUNDLE.open("wb") as f:
    pickle.dump({"measurement": measurement, "basis": basis, "linearity": linearity}, f)
print(f"saved {BUNDLE}")

report_dict = ao.analyses.measurement_report(measurement, save_dir=str(OUT_FIG))
for src_name, dst_name in [("diagnostics", "fig4a_rpe1_essential_diagnostics.png"),
                             ("guide_drops", "fig4b_rpe1_essential_drops.png")]:
    src_path = OUT_FIG / f"{src_name}.png"
    if src_path.exists():
        src_path.rename(OUT_FIG / dst_name)
        print(f"wrote {OUT_FIG / dst_name}")
