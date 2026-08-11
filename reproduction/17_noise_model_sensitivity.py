"""Fig S17: noise-model sensitivity — residual-resampling vs i.i.d. Gaussian.

The recovery experiments in §2.2–2.5 use i.i.d. Gaussian noise on S entries.
Real per-entry Δz noise is heteroscedastic across programs and correlated
between them. This script rebuilds the noise term by bootstrap-resampling
real per-guide half-Δz residuals from K562 essential, matching Frobenius
norm to σ=0.266, and compares the recovery outcome to the i.i.d. Gaussian
baseline.

Construction of the residual bank:
  For each K562 target with ≥20 cells, split cells into two halves, compute
  half-Δz vectors d₁, d₂ ∈ ℝ^d. The signed half-difference (d₁ − d₂)/2 is
  a noise realization for a per-guide Δz (Methods §4.4). We accumulate one
  such d-vector per target → residual bank R ∈ ℝ^(N_targets × d).

Noise model:
  For each replicate: build a d×n noise matrix by sampling n residuals with
  replacement from R (columns), then rescale globally so that per-entry std
  matches σ=0.266. Preserves cross-program correlations and heteroscedasticity.

Comparison:
  Rerun the §2.2 recovery: draw J_true (dense/sparse-2%/rank-5), compute
  S_true = -J_true⁻¹ U_real, add either (a) i.i.d. Gaussian noise or (b) the
  residual-resampled noise, fit A, report cos and cos_1.

If the residual-resampled cosines differ from the Gaussian cosines by more
than one replicate SD, external validity of the density projections becomes
noise-model-dependent and the sensitivity is documented.

Runtime ~6 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse
import anchorop as ao

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1] / "examples/data/K562_essential_normalized_singlecell_01.h5ad"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

N_REPS = 30  # heavier than 15, lighter than the empirical-null 200 — this is a sensitivity check
SIGMA = 0.266
RANK_TOL = 1e-2
STRUCTURES = ["dense", "sparse_2pct", "rank_5"]
SEED_BASE = 20260810


def draw_J(d, seed, structure):
    rng = np.random.default_rng(seed)
    if structure == "dense":
        G = rng.normal(size=(d, d)) / np.sqrt(d)
        return G - 1.5 * np.eye(d)
    if structure == "sparse_2pct":
        mask = rng.random((d, d)) < 0.02
        G = np.zeros((d, d))
        G[mask] = rng.normal(size=int(mask.sum())) / np.sqrt(max(mask.sum(), 1) / d)
        return G - 1.5 * np.eye(d)
    if structure == "rank_5":
        Ul = rng.normal(size=(d, 5)) / np.sqrt(d)
        Vl = rng.normal(size=(5, d)) / np.sqrt(d)
        return Ul @ Vl - 1.5 * np.eye(d)
    raise ValueError


def fit_operator(S, U):
    S_pinv, *_ = regularized_pseudoinverse(S, method="tsvd", parameter="path", rank_tol=RANK_TOL)
    return -U @ S_pinv


def cos_full(A, J):
    inner = float(np.sum(A * J))
    return inner / max(np.linalg.norm(A) * np.linalg.norm(J), 1e-30)


def cos_topk(A, J, S_true, k):
    Us, _, _ = np.linalg.svd(S_true, full_matrices=False)
    Uk = Us[:, :k]
    AUk = A @ Uk; JUk = J @ Uk
    return float(np.sum(AUk * JUk)) / max(np.linalg.norm(AUk) * np.linalg.norm(JUk), 1e-30)


# ============================================================================
# Build the residual bank from K562 essential
# ============================================================================
print("Building residual bank from K562 essential h5ad...")
with (RESULTS / "k562_essential_measurement.pkl").open("rb") as f:
    state = pickle.load(f)
m = state["measurement"]
basis = state["basis"]
W = basis.loadings  # (n_genes, d)
gene_names_basis = basis.gene_names
d, n_guides = m.S.shape

if not DATA.exists():
    raise SystemExit(f"missing {DATA}; needed to build residual bank")

print("  loading h5ad (this takes a minute)...")
adata = ao.load_replogle_h5ad(str(DATA))

# Restrict expression matrix to genes present in the basis
gene_index = {g: i for i, g in enumerate(adata.var_names)}
basis_gene_positions = np.array([gene_index[g] for g in gene_names_basis if g in gene_index])
if len(basis_gene_positions) != W.shape[0]:
    raise SystemExit(f"gene mismatch: basis has {W.shape[0]} genes, matched {len(basis_gene_positions)}")

from scipy import sparse as sp
E = adata.X
if sp.issparse(E):
    E_basis = E[:, basis_gene_positions]
    Z = np.asarray(E_basis @ W)  # (n_cells, d)
else:
    Z = E[:, basis_gene_positions] @ W

# Iterate over guides that made it to the measurement; compute split-half residual per target
guide_col = None
for candidate in ["gene", "target_gene", "gene_id", "sgRNA_target_gene", "guide_identity"]:
    if candidate in adata.obs.columns:
        guide_col = candidate; break
if guide_col is None:
    raise SystemExit("could not find guide/target column in adata.obs; check schema")

print(f"  using guide column: {guide_col}")

# NT control mean
ctrl_label = None
for guess in ["non-targeting", "NT", "safe_targeting", "control", "Non-Targeting"]:
    if guess in adata.obs[guide_col].unique().tolist():
        ctrl_label = guess; break
if ctrl_label is None:
    counts = adata.obs[guide_col].value_counts()
    for guess in counts.index:
        gl = str(guess).lower()
        if "non" in gl and ("target" in gl or "-t" in gl):
            ctrl_label = guess; break
if ctrl_label is None:
    ctrl_label = basis.metadata.get("control_label") if isinstance(basis.metadata, dict) else None
if ctrl_label is None:
    raise SystemExit("could not identify control label; check schema")
print(f"  control label: {ctrl_label}")
ctrl_mask = (adata.obs[guide_col] == ctrl_label).values
if ctrl_mask.sum() < 100:
    raise SystemExit(f"only {ctrl_mask.sum()} control cells found; check labels")
z_ctrl_mean = Z[ctrl_mask].mean(axis=0)

# Build residuals for each target with ≥20 cells (subset to ~500 targets to keep runtime bounded)
targets = [g for g, c in adata.obs[guide_col].value_counts().items() if g != ctrl_label and c >= 20]
np.random.default_rng(SEED_BASE).shuffle(targets)
targets = targets[:800]
print(f"  building bank from {len(targets)} targets (each contributes one d-vector)...")

R_bank = []
rng_bank = np.random.default_rng(SEED_BASE + 77)
for g in targets:
    mask_g = (adata.obs[guide_col] == g).values
    idx = np.where(mask_g)[0]
    rng_bank.shuffle(idx)
    half = len(idx) // 2
    if half < 10:
        continue
    z1 = Z[idx[:half]].mean(axis=0) - z_ctrl_mean
    z2 = Z[idx[half:2*half]].mean(axis=0) - z_ctrl_mean
    R_bank.append((z1 - z2) / 2.0)
R_bank = np.array(R_bank)
print(f"  R_bank shape: {R_bank.shape}, per-entry std: {R_bank.std():.3f}")

# Rescale bank so its per-entry std = SIGMA (matches the noise anchor)
R_bank *= (SIGMA / R_bank.std())
print(f"  rescaled per-entry std to {R_bank.std():.3f}")


def resample_noise(d, n, rng):
    """Draw n residual columns from R_bank; each column is a d-vector."""
    idx = rng.integers(0, R_bank.shape[0], size=n)
    return R_bank[idx].T  # d×n


# ============================================================================
# Run recovery under both noise models
# ============================================================================
U_real = m.U
results = {}
for structure in STRUCTURES:
    print(f"\n{'-'*72}\nstructure: {structure}\n{'-'*72}")
    per_struct = {"gaussian": [], "residual": []}
    for rep in range(N_REPS):
        J = draw_J(d, seed=SEED_BASE + rep, structure=structure)
        S_true = -np.linalg.solve(J, U_real)
        # Gaussian
        rng_g = np.random.default_rng(rep + 5000)
        S_g = S_true + SIGMA * rng_g.normal(size=S_true.shape)
        A_g = fit_operator(S_g, U_real)
        per_struct["gaussian"].append({
            "cos": cos_full(A_g, J), "cos_1": cos_topk(A_g, J, S_true, 1),
            "cos_5": cos_topk(A_g, J, S_true, 5),
        })
        # Residual-resampled
        rng_r = np.random.default_rng(rep + 6000)
        S_r = S_true + resample_noise(d, n_guides, rng_r)
        A_r = fit_operator(S_r, U_real)
        per_struct["residual"].append({
            "cos": cos_full(A_r, J), "cos_1": cos_topk(A_r, J, S_true, 1),
            "cos_5": cos_topk(A_r, J, S_true, 5),
        })
    for model in ["gaussian", "residual"]:
        arr = per_struct[model]
        cos_mean = np.mean([x["cos"] for x in arr]); cos_std = np.std([x["cos"] for x in arr])
        cos1_mean = np.mean([x["cos_1"] for x in arr]); cos1_std = np.std([x["cos_1"] for x in arr])
        cos5_mean = np.mean([x["cos_5"] for x in arr]); cos5_std = np.std([x["cos_5"] for x in arr])
        print(f"  {model:>10s}: cos = {cos_mean:+.4f} ± {cos_std:.3f}   "
              f"cos_1 = {cos1_mean:+.3f} ± {cos1_std:.3f}   "
              f"cos_5 = {cos5_mean:+.3f} ± {cos5_std:.3f}")
    results[structure] = per_struct

# Save JSON
summary = {}
for structure in STRUCTURES:
    summary[structure] = {}
    for model in ["gaussian", "residual"]:
        arr = results[structure][model]
        summary[structure][model] = {
            "n_reps": N_REPS,
            "cos_mean": float(np.mean([x["cos"] for x in arr])),
            "cos_std": float(np.std([x["cos"] for x in arr])),
            "cos_1_mean": float(np.mean([x["cos_1"] for x in arr])),
            "cos_1_std": float(np.std([x["cos_1"] for x in arr])),
            "cos_5_mean": float(np.mean([x["cos_5"] for x in arr])),
            "cos_5_std": float(np.std([x["cos_5"] for x in arr])),
        }
(OUT_DIR / "noise_model_sensitivity.json").write_text(json.dumps(summary, indent=2))

# Figure: box plot per (structure, model) for cos, cos_1, cos_5
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
for ax_i, metric in enumerate(["cos", "cos_1", "cos_5"]):
    ax = axes[ax_i]
    positions, labels, data = [], [], []
    for si, structure in enumerate(STRUCTURES):
        for mj, model in enumerate(["gaussian", "residual"]):
            arr = [x[metric] for x in results[structure][model]]
            positions.append(3*si + mj)
            labels.append(f"{structure[:6]}\n{model[:5]}")
            data.append(arr)
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
    for patch, pos in zip(bp["boxes"], positions):
        patch.set_facecolor("#7a9ecc" if pos % 3 == 0 else "#c65a30")
        patch.set_alpha(0.6)
    ax.axhline(0, color="0.5", lw=0.6)
    if metric == "cos":
        ax.axhline(0.033, color="0.4", ls=":", lw=1)
        ax.axhline(-0.033, color="0.4", ls=":", lw=1)
    elif metric == "cos_1":
        ax.axhline(0.183, color="0.4", ls=":", lw=1)
        ax.axhline(-0.183, color="0.4", ls=":", lw=1)
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — Gaussian (blue) vs residual (orange)")

fig.suptitle(f"Fig S17: Noise-model sensitivity, N={N_REPS} replicates per cell, σ={SIGMA}, K562 geometry.\n"
             "Dotted grey = analytic random-matrix null ±1 SD.", fontsize=10.5, y=1.06)
out = OUT_DIR / "figS17_noise_model_sensitivity.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")
