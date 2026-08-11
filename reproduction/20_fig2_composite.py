"""Fig 2 (composite): recovery-vs-σ + empirical-null distributions.

Rewrites the paper's main-text Fig 2 to lead with the pipeline-matched
empirical null rather than the analytic random-matrix null. This is where
§2.2's central claim lives — full-operator cos indistinguishable from
cross-replicate null — and Fig 2 should show that directly.

Layout (4 columns × 3 rows):
  cols = dense, sparse_10pct, sparse_2pct, rank_5 (four ground-truth ensembles)
  row 0: cos(A, J) vs per-entry σ, K562 + RPE1, empirical cross-rep null overlay
  row 1: scale-sensitive ‖A-J‖_F/‖J‖_F vs σ, same overlay
  row 2: at K562 σ = 0.240, histogram of real cos, cross-rep null, shuffled-U null (dense col only shows all three; other cols show just real vs cross-rep for space)

Uses precomputed JSONs from scripts 13 and 16, so this is quick to redraw.
Runtime <30 s.
"""
import warnings; warnings.filterwarnings("ignore")
import json, pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"

# Load precomputed cosine sweeps
sweep_data = json.loads((OUT_DIR / "operator_recovery.json").read_text())
STRUCTURES = ["dense", "sparse_10pct", "sparse_2pct", "low_rank_5"]
struct_titles = {
    "dense":        "dense J (Ginibre)",
    "sparse_10pct": "sparse J (10%)",
    "sparse_2pct":  "very sparse J (2%)",
    "low_rank_5":   "low-rank J (rank 5)",
}

# Generate cross-replicate empirical null at K562 σ = 0.240 for cos_full, per structure
# The cross-rep null is computed on the fly for dense only (matches Fig S16).
# For other structures, we approximate the null band by using the same dense-J null width,
# since the null width is dominated by the pipeline geometry not the ground-truth structure.
SIGMA_MEASURED = 0.240
N_REPS_HIST = 200
RANK_TOL = 1e-2
SEED_BASE = 20260810


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


def fit_operator(S, U):
    S_pinv, *_ = regularized_pseudoinverse(S, method="tsvd", parameter="path", rank_tol=RANK_TOL)
    return -U @ S_pinv


def cos_full(A, J):
    return float(np.sum(A * J)) / max(np.linalg.norm(A) * np.linalg.norm(J), 1e-30)


# Empirical null distributions per (cell_line, structure) at σ = 0.240
print("Building empirical-null distributions per structure at σ=0.240 (K562)...")
with (RESULTS / "k562_essential_measurement.pkl").open("rb") as f:
    state = pickle.load(f)
U_real = state["measurement"].U
d, n_guides = state["measurement"].S.shape

null_data = {}
for structure in STRUCTURES:
    print(f"  {structure}...")
    Js, A_fits = [], []
    rng_n = np.random.default_rng(SEED_BASE + 999 + hash(structure) % 1000)
    for r in range(N_REPS_HIST):
        J = draw_J(d, seed=SEED_BASE + r + hash(structure) % 1000, structure=structure)
        S_true = -np.linalg.solve(J, U_real)
        S_obs = S_true + SIGMA_MEASURED * rng_n.normal(size=S_true.shape)
        Js.append(J); A_fits.append(fit_operator(S_obs, U_real))
    real = np.array([cos_full(A_fits[r], Js[r]) for r in range(N_REPS_HIST)])
    # Cross-replicate null with 5 shifts to get 1000 samples
    null_cr = []
    for shift in range(1, 6):
        for r in range(N_REPS_HIST):
            null_cr.append(cos_full(A_fits[r], Js[(r + shift) % N_REPS_HIST]))
    # Shuffled-U null with fresh noise
    null_su = []
    rng_su = np.random.default_rng(SEED_BASE + 888 + hash(structure) % 1000)
    for r in range(N_REPS_HIST):
        perm = rng_su.permutation(n_guides)
        U_shuf = U_real[:, perm]
        S_ts = -np.linalg.solve(Js[r], U_shuf)
        S_obs = S_ts + SIGMA_MEASURED * rng_su.normal(size=S_ts.shape)
        A_sh = fit_operator(S_obs, U_real)
        null_su.append(cos_full(A_sh, Js[r]))
    null_data[structure] = {"real": real, "cross_rep": np.array(null_cr), "shuf_U": np.array(null_su)}
    print(f"    real mean {real.mean():+.3f} ± {real.std():.3f}   "
          f"cross-rep null mean {np.mean(null_cr):+.3f} ± {np.std(null_cr):.3f}   "
          f"shuf-U null mean {np.mean(null_su):+.3f} ± {np.std(null_su):.3f}")

# Draw the composite
fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
for col, structure in enumerate(STRUCTURES):
    # ---- Row 0: cos vs σ ----
    ax = axes[0, col]
    for cell_line, color, marker in [("K562_essential", "#1f4e79", "o"),
                                       ("RPE1_essential", "#c65a30", "s")]:
        r = sweep_data[cell_line][structure]
        sig = [e["sigma"] for e in r]
        cos_m = [e["cosine_mean"] for e in r]
        cos_s = [e["cosine_std"] for e in r]
        ax.errorbar(sig, cos_m, yerr=cos_s, fmt=f"{marker}-", color=color, lw=2, ms=6,
                     capsize=3, label=cell_line.replace("_essential", ""))
    # Empirical cross-rep null band at σ = 0.240 (from null_data)
    nc = null_data[structure]["cross_rep"]
    nc_mean, nc_sd = float(nc.mean()), float(nc.std())
    ax.axhspan(nc_mean - nc_sd, nc_mean + nc_sd, alpha=0.18, color="#7a7a7a",
                label="cross-rep null ±1 SD" if col == 0 else None)
    ax.axhline(nc_mean, color="#4a4a4a", lw=0.8, ls="--")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axhline(1.0, color="green", ls=":", lw=1, alpha=0.5)
    ax.axhline(0.5, color="0.4", ls="--", lw=0.7, alpha=0.5)
    ax.axvspan(0.20, 0.30, alpha=0.15, color="#c65a30")
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_ylim(-0.15, 1.1)
    ax.set_xlabel("per-entry noise σ" if col in (0, 3) else "")
    if col == 0:
        ax.set_ylabel("cos(A, J_true)")
    ax.set_title(f"({['a','b','c','d'][col]}) {struct_titles[structure]}")
    if col == 0:
        ax.legend(loc="upper right", fontsize=8)

    # ---- Row 1: scale-sensitive error vs σ ----
    ax = axes[1, col]
    for cell_line, color, marker in [("K562_essential", "#1f4e79", "o"),
                                       ("RPE1_essential", "#c65a30", "s")]:
        r = sweep_data[cell_line][structure]
        sig = [e["sigma"] for e in r]
        frob_m = [e["frob_rel_err_mean"] for e in r]
        frob_s = [e["frob_rel_err_std"] for e in r]
        mag_m = [e["magnitude_ratio_mean"] for e in r]
        ax.errorbar(sig, frob_m, yerr=frob_s, fmt=f"{marker}-", color=color, lw=2, ms=6,
                     capsize=3, label=f"{cell_line.replace('_essential','')} ‖A-J‖/‖J‖")
        ax.plot(sig, mag_m, f"{marker}--", color=color, lw=1, ms=5, alpha=0.6,
                 label=f"{cell_line.replace('_essential','')} ‖A‖/‖J‖" if col == 0 else None)
    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax.axhline(np.sqrt(2), color="gray", ls="--", lw=1, alpha=0.4)
    ax.axhline(0.15, color="green", ls=":", lw=1, alpha=0.5)
    ax.axvspan(0.20, 0.30, alpha=0.15, color="#c65a30")
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("per-entry noise σ")
    if col == 0:
        ax.set_ylabel("‖A_fit − J_true‖_F/‖J‖ (solid)\n‖A‖/‖J‖ (dashed)")
        ax.legend(loc="center right", fontsize=7)
    ax.set_title(f"({['e','f','g','h'][col]}) magnitude at same σ sweep")

    # ---- Row 2: empirical distributions at σ = 0.240 (K562) ----
    ax = axes[2, col]
    real = null_data[structure]["real"]
    ncross = null_data[structure]["cross_rep"]
    nshuf = null_data[structure]["shuf_U"]
    bins = np.linspace(min(np.min(nshuf), np.min(ncross), np.min(real)) - 0.02,
                       max(np.max(nshuf), np.max(ncross), np.max(real)) + 0.02, 35)
    ax.hist(nshuf, bins=bins, color="#8899cc", alpha=0.55, density=True,
             label="shuffled-U null" if col == 0 else None)
    ax.hist(ncross, bins=bins, color="#7a7a7a", alpha=0.55, density=True,
             label="cross-replicate null" if col == 0 else None)
    ax.hist(real, bins=bins, color="#1f4e79", alpha=0.72, density=True,
             label="real cos(A_r, J_r)" if col == 0 else None)
    ax.axvline(float(np.mean(real)), color="#1f4e79", ls="--", lw=1.2)
    ax.axvline(0.0, color="0.5", lw=0.5)
    ax.axvline(0.5, color="green", ls=":", lw=0.7)
    real_mean, cross_mean = float(np.mean(real)), float(np.mean(ncross))
    cross_sd = float(np.std(ncross))
    zval = (real_mean - cross_mean) / max(cross_sd, 1e-9)
    ax.text(0.03, 0.95, f"real: {real_mean:+.3f}\ncross-rep null: {cross_mean:+.3f}\nz = {zval:+.2f}",
             transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))
    ax.set_xlabel("cos(A, J_true)")
    if col == 0:
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"({['i','j','k','l'][col]}) K562 σ = 0.240, N = {N_REPS_HIST}")

fig.suptitle(
    "Fig 2. Full-operator recovery at Replogle-matched geometry, with pipeline-matched empirical null.\n"
    "Top row: cos(A, J_true) vs per-entry σ, K562 blue circles / RPE1 orange squares, 15 reps; grey band = "
    "empirical cross-replicate null ±1 SD at σ = 0.240. Middle row: scale-sensitive ‖A-J‖_F/‖J‖_F (solid) "
    "and magnitude ratio ‖A‖/‖J‖ (dashed) on same sweep. Bottom row: empirical distributions at K562 σ = 0.240, "
    "N = 200 replicates: real cos(A_r, J_r) (blue) vs cross-replicate null cos(A_r, J_{r'}) (grey) "
    "vs shuffled-U null cos(A_shuf, J_r) (light blue). The real distribution is indistinguishable from the "
    "cross-replicate null for every ground-truth ensemble.",
    fontsize=10.5, y=1.02)

out = OUT_DIR / "fig2_operator_recovery.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")
