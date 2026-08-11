"""Fig 3a/b: Replogle K562 essential-gene measurement + save bundle.

Loads the K562 essential h5ad, runs the full pipeline (basis fit, measure_operator),
produces fig3a (diagnostics panel via ao.analyses.measurement_report) and
fig3b (guide-drop pareto), and pickles the measurement bundle to
`results/k562_essential_measurement.pkl` for downstream scripts.

Requires `examples/data/K562_essential_normalized_singlecell_01.h5ad`.
Runtime ~10 min. Uses the same code path as `examples/01b_measure_k562_replogle.ipynb`.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
from pathlib import Path
import numpy as np
import anchorop as ao
from sklearn.decomposition import PCA

DATA = Path(__file__).resolve().parents[1] / "examples/data/K562_essential_normalized_singlecell_01.h5ad"
OUT_FIG = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_RES = Path(__file__).resolve().parents[1] / "results"
OUT_FIG.mkdir(exist_ok=True); OUT_RES.mkdir(exist_ok=True)

D_PROGRAMS = 30
N_HVG = 3000
N_TARGETS_KEEP = 200
SEED = 20260729

print("Loading K562 essential h5ad...")
adata = ao.load_replogle_h5ad(str(DATA))
X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
finite = np.isfinite(X).all(axis=0)
adata = adata[:, finite].copy(); X = X[:, finite]

# Pick top-N targets by cell count (matches notebook config)
target_counts = adata.obs.loc[adata.obs["target_gene"] != "", "target_gene"].value_counts()
qualifying = target_counts[target_counts >= 60].index.tolist()[:N_TARGETS_KEEP]
keep_mask = adata.obs["target_gene"].isin(qualifying) | (adata.obs["target_gene"] == "")
adata = adata[keep_mask].copy()

# Aggregate to target level
perturbed = adata.obs["target_gene"] != ""
adata.obs.loc[perturbed, "guide"] = "guide_" + adata.obs.loc[perturbed, "target_gene"].astype(str)

# HVG by variance (Replogle ships z-scored residuals)
X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
gene_var = X_dense.var(axis=0)
top_hvg = np.argsort(-gene_var)[:N_HVG]
target_syms = {t for t in adata.obs["target_gene"].astype(str).unique() if t}
extra_idx = np.array([i for i, g in enumerate(np.asarray(adata.var_names)) if g in target_syms])
keep_feats = np.union1d(top_hvg, extra_idx)
adata_hvg = adata[:, keep_feats].copy()

# PCA basis on controls
ctrl_mask = (adata_hvg.obs["guide"] == "non-targeting").to_numpy()
X_ctrl = adata_hvg.X.toarray() if hasattr(adata_hvg.X, "toarray") else adata_hvg.X
X_ctrl = X_ctrl[ctrl_mask]
pca = PCA(n_components=D_PROGRAMS, random_state=SEED).fit(X_ctrl - X_ctrl.mean(0))
basis = ao.make_program_basis(
    pca.components_.T.astype(np.float32), adata_hvg.var_names,
    method="pca_external", control_count=int(ctrl_mask.sum()), normalize=False,
)

# Measurement
measurement = ao.measure_operator(
    adata_hvg, basis,
    guide_key="guide", target_key="target_gene", control_label="non-targeting",
    min_cells_per_guide=30, min_knockdown_efficiency=0.05,
    reg="tsvd", reg_param="path", rank_tol=1e-2,
    bootstrap=100, bootstrap_seed=SEED, state_label="K562_essential",
)
r = measurement.report
print(f"retained {r.n_guides_retained}/{r.n_guides_input}, rank {r.effective_response_rank}/{r.d}, "
      f"cond {r.condition_number:.2f}")

# Linearity check
linearity = ao.linearity_check(measurement, threshold=0.25, n_null=200, null_seed=42)
print(f"rel_diff = {linearity.relative_difference:.3f}, null_median = {linearity.null_median:.3f}")

# Save bundle for downstream scripts
BUNDLE = OUT_RES / "k562_essential_measurement.pkl"
with BUNDLE.open("wb") as f:
    pickle.dump({"measurement": measurement, "basis": basis, "linearity": linearity}, f)
print(f"saved {BUNDLE}")

# Figures via analyses report
report_dict = ao.analyses.measurement_report(measurement, save_dir=str(OUT_FIG))
# The analyses module saves standard figure names; rename to match manuscript convention
for src_name, dst_name in [("diagnostics", "fig3a_k562_essential_diagnostics.png"),
                             ("guide_drops", "fig3b_k562_essential_drops.png")]:
    src_path = OUT_FIG / f"{src_name}.png"
    if src_path.exists():
        src_path.rename(OUT_FIG / dst_name)
        print(f"wrote {OUT_FIG / dst_name}")
