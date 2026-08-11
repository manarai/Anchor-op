"""Fig S13: operator-recovery check at Replogle-matched scale.

The paper's §3.3–3.4 report full-rank identification (K562 188/200, RPE1 153/200
guides at d=30), and §3.5–3.6 show the linearity diagnostics are noise-limited
at that scale. A reviewer's follow-up question: even setting linearity aside,
is the reported operator ITSELF recoverable at published Perturb-seq scale?

This script answers it directly. Under a synthetic LINEAR ground truth J drawn
fresh for each replicate, using each Replogle cell line's real U and real κ,
adds per-entry Gaussian noise at σ = 0.266 (measured on K562 essential; §3.5),
computes the fit `A = -U · pinv(S)`, and reports:

  ||A - J_true||_F / ||J_true||_F                (Frobenius recovery)
  |max Re(λ_A) - max Re(λ_J_true)|               (spectral abscissa error)
  median |Re(λ_A) - Re(λ_J_true)| / median|Re(λ_J_true)|   (eigenvalue drift)

Sweeps σ ∈ {0, 0.01, 0.05, 0.1, 0.266, 0.5} at each cell line's real (d, n_guides,
U, κ), 15 replicates per point. Also translates σ → required cells/guide via
σ = σ_percell/√n_cells with σ_percell ≈ 2.2.

Requires pickled measurement bundles from `../results/`. Runtime ~3 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

RANK_TOL = 1e-2
SIGMA_PERCELL = 2.2
N_REPS = 15
SIGMAS = [0.0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20, 0.266, 0.35, 0.5]


def draw_J(d, seed, structure="dense"):
    """Draw a random J with a specified sparsity structure. All variants are
    shifted to spectral radius ~−1.5 so they are stable and comparably conditioned."""
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
        U_lr = rng.normal(size=(d, 5)) / np.sqrt(d)
        V_lr = rng.normal(size=(5, d)) / np.sqrt(d)
        return U_lr @ V_lr - 1.5 * np.eye(d)
    raise ValueError(structure)


def fit_operator(S, U, rank_tol=RANK_TOL):
    """A = -U · pinv(S) via anchor-op's regularized TSVD path."""
    S_pinv, *_ = regularized_pseudoinverse(
        S, method="tsvd", parameter="path", rank_tol=rank_tol
    )
    return -U @ S_pinv


def recovery_metrics(A_fit, J_true):
    # Scale-sensitive metric (raw): ||A - J|| / ||J||. Equals 1 when A → 0.
    frob = float(np.linalg.norm(A_fit - J_true) / np.linalg.norm(J_true))
    # Scale-invariant metric 1: cosine similarity in Frobenius inner product.
    # Equals 1 iff A = c·J for some c > 0, equals 0 iff A ⟂ J.
    inner = float(np.sum(A_fit * J_true))
    norm_A = float(np.linalg.norm(A_fit))
    norm_J = float(np.linalg.norm(J_true))
    cos = inner / max(norm_A * norm_J, 1e-30)
    # Scale-invariant metric 2: best-scalar-rescaled error.
    # min_c ||cA - J||² = ||J||²(1 - cos²), so best_rescaled_err = sqrt(1 - cos²).
    best_rescaled = float(np.sqrt(max(0.0, 1.0 - cos**2)))
    # Optimal rescaling factor (would need to be known ex ante to apply)
    c_star = inner / max(norm_A**2, 1e-30)
    # Magnitude ratio: is the fit shrunk to zero or matched in magnitude?
    magnitude_ratio = norm_A / max(norm_J, 1e-30)
    # Eigenvalue drift (as before)
    eig_true = np.linalg.eigvals(J_true)
    eig_fit = np.linalg.eigvals(A_fit)
    remaining = list(range(len(eig_fit)))
    paired_true, paired_fit = [], []
    for et in eig_true:
        j = min(remaining, key=lambda i: abs(eig_fit[i] - et))
        paired_true.append(et); paired_fit.append(eig_fit[j]); remaining.remove(j)
    et_arr = np.array(paired_true); ef_arr = np.array(paired_fit)
    abscissa_err = float(abs(max(ef_arr.real) - max(et_arr.real)))
    real_drift = float(np.median(np.abs(ef_arr.real - et_arr.real)) /
                        max(np.median(np.abs(et_arr.real)), 1e-9))
    return {"frob_rel_err": frob,
             "cosine": cos,
             "best_rescaled_err": best_rescaled,
             "magnitude_ratio": magnitude_ratio,
             "c_star": c_star,
             "abscissa_err": abscissa_err,
             "real_eig_drift": real_drift}


STRUCTURES = ["dense", "sparse_10pct", "sparse_2pct", "low_rank_5"]

results = {}
for cell_line, filename in [("K562_essential", "k562_essential_measurement.pkl"),
                              ("RPE1_essential", "rpe1_essential_measurement.pkl")]:
    bundle_path = RESULTS / filename
    if not bundle_path.exists():
        raise SystemExit(f"missing {bundle_path} — run reproduction/03 and 04 first")
    with bundle_path.open("rb") as f:
        state = pickle.load(f)
    m = state["measurement"]
    U_real = m.U
    d, n_guides = m.S.shape
    print(f"\n{'=' * 70}")
    print(f"{cell_line} (d={d}, n_guides={n_guides})")
    print('=' * 70)
    per_line = {}
    for structure in STRUCTURES:
        print(f"\n  ground-truth J structure: {structure}")
        print(f"  {'σ':>8s} {'cells/guide':>13s}   {'‖A-J‖/‖J‖ mean±std':>25s}   {'|Δ abscissa|':>15s}   {'real eig drift':>18s}")
        rows = []
        for sigma in SIGMAS:
            n_cells = float("inf") if sigma == 0 else (SIGMA_PERCELL / sigma) ** 2
            frobs, coss, resc, magr, absc = [], [], [], [], []
            for rep in range(N_REPS):
                J_true = draw_J(d, seed=20260810 + rep, structure=structure)
                S_true = -np.linalg.solve(J_true, U_real)
                rng_noise = np.random.default_rng(rep + 10000)
                S_obs = S_true + sigma * rng_noise.normal(size=S_true.shape)
                A_fit = fit_operator(S_obs, U_real)
                met = recovery_metrics(A_fit, J_true)
                frobs.append(met["frob_rel_err"])
                coss.append(met["cosine"])
                resc.append(met["best_rescaled_err"])
                magr.append(met["magnitude_ratio"])
                absc.append(met["abscissa_err"])
            entry = {
                "sigma": sigma,
                "cells_per_guide": None if not np.isfinite(n_cells) else float(n_cells),
                "frob_rel_err_mean": float(np.mean(frobs)),
                "frob_rel_err_std": float(np.std(frobs)),
                "cosine_mean": float(np.mean(coss)),
                "cosine_std": float(np.std(coss)),
                "best_rescaled_err_mean": float(np.mean(resc)),
                "best_rescaled_err_std": float(np.std(resc)),
                "magnitude_ratio_mean": float(np.mean(magr)),
                "magnitude_ratio_std": float(np.std(magr)),
                "abscissa_err_mean": float(np.mean(absc)),
                "abscissa_err_std": float(np.std(absc)),
            }
            rows.append(entry)
            cells_str = "∞ (noise-free)" if not np.isfinite(n_cells) else f"{n_cells:>13,.0f}"
            print(f"  σ={sigma:>7.4f} cells/g={cells_str:>13s}  "
                  f"‖A-J‖/‖J‖={entry['frob_rel_err_mean']:.3f}±{entry['frob_rel_err_std']:.3f}  "
                  f"cos={entry['cosine_mean']:+.3f}±{entry['cosine_std']:.3f}  "
                  f"best-rescaled={entry['best_rescaled_err_mean']:.3f}  "
                  f"‖A‖/‖J‖={entry['magnitude_ratio_mean']:.3f}")
        per_line[structure] = rows
    results[cell_line] = per_line

(OUT_DIR / "operator_recovery.json").write_text(json.dumps(results, indent=2))

# 2x4 figure: rows = (raw scale-sensitive error, scale-invariant cosine), cols = structures
fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
structure_titles = {
    "dense":         "dense J (Ginibre)",
    "sparse_10pct":  "sparse J (10% density)",
    "sparse_2pct":   "very sparse J (2% density)",
    "low_rank_5":    "low-rank J (rank 5)",
}
for col, structure in enumerate(STRUCTURES):
    ax_frob = axes[0, col]; ax_cos = axes[1, col]
    for cell_line, color in [("K562_essential", "#1f4e79"), ("RPE1_essential", "#c65a30")]:
        r = results[cell_line][structure]
        sig = [e["sigma"] for e in r]
        frob_m = [e["frob_rel_err_mean"] for e in r]
        frob_s = [e["frob_rel_err_std"] for e in r]
        cos_m = [e["cosine_mean"] for e in r]
        cos_s = [e["cosine_std"] for e in r]
        mag_m = [e["magnitude_ratio_mean"] for e in r]
        # Frobenius panel
        ax_frob.errorbar(sig, frob_m, yerr=frob_s, fmt="o-", color=color, lw=2, ms=7, capsize=3,
                          label=cell_line.replace("_essential", ""))
        # Cosine panel
        ax_cos.errorbar(sig, cos_m, yerr=cos_s, fmt="o-", color=color, lw=2, ms=7, capsize=3,
                         label=cell_line.replace("_essential", ""))
        # Magnitude ratio as dashed on cosine panel (right axis via alpha)
        ax_cos.plot(sig, mag_m, "s--", color=color, lw=1, ms=5, alpha=0.5,
                     label=f"{cell_line.replace('_essential','')} ‖A‖/‖J‖")
    # Frobenius axis decorations
    ax_frob.axhline(np.sqrt(2), color="gray", ls="--", lw=1, alpha=0.6)
    ax_frob.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax_frob.axhline(0.15, color="green", ls=":", lw=1, alpha=0.6)
    ax_frob.axvspan(0.20, 0.30, alpha=0.15, color="#c65a30")
    ax_frob.set_xscale("symlog", linthresh=0.005)
    ax_frob.set_ylim(0, 1.6)
    ax_frob.set_title(f"({['a','b','c','d'][col]}) {structure_titles[structure]}\n‖A-J‖_F / ‖J‖_F  (scale-sensitive)")
    if col == 0:
        ax_frob.set_ylabel(r"$\|A_{\rm fit} - J_{\rm true}\|_F / \|J_{\rm true}\|_F$")
        ax_frob.legend(loc="lower right", fontsize=8)
    # Cosine axis decorations
    ax_cos.axhline(0, color="gray", lw=0.8)
    ax_cos.axhline(1.0, color="green", ls=":", lw=1, alpha=0.6)
    ax_cos.axvspan(0.20, 0.30, alpha=0.15, color="#c65a30")
    ax_cos.set_xscale("symlog", linthresh=0.005)
    ax_cos.set_ylim(-0.3, 1.15)
    ax_cos.set_xlabel(r"per-entry noise σ")
    ax_cos.set_title(f"({['e','f','g','h'][col]}) cos(A, J_true)  (scale-invariant)\n+ dashed: ‖A‖/‖J‖ (magnitude ratio)")
    if col == 0:
        ax_cos.set_ylabel(r"cos$(A_{\rm fit}, J_{\rm true})$ (solid); $\|A\|_F / \|J\|_F$ (dashed)")
        ax_cos.legend(loc="upper right", fontsize=7)

fig.suptitle(
    "Fig S13: Operator recovery at Replogle-matched scale, scale-sensitive AND scale-invariant metrics.\n"
    "Shaded band = measured Replogle σ ≈ 0.27. Top row: ‖A-J‖_F/‖J‖_F (scale-sensitive, ≈1 if A→0). "
    "Bottom row: cos(A, J) (scale-invariant) + ‖A‖/‖J‖ (dashed). "
    "Cosine near 0 → no directional recovery; near 1 → directionally correct.",
    fontsize=10.5, y=1.03)
out = OUT_DIR / "figS13_operator_recovery.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")

print("\n" + "=" * 88)
print("VERDICT (scale-invariant): cos ≈ 0 → no directional content; cos > 0.3 → 'shrunk-toward-zero' story")
print("=" * 88)
for cell_line in results:
    print(f"\n{cell_line}:")
    print(f"  {'structure':>16s}  {'σ':>7s}   {'‖A-J‖/‖J‖':>10s}  {'cos':>8s}  {'best-rescaled':>13s}  {'‖A‖/‖J‖':>10s}")
    for structure in STRUCTURES:
        r = results[cell_line][structure]
        for sigma_target in [0.005, 0.025, 0.266]:
            e = next(x for x in r if abs(x["sigma"] - sigma_target) < 1e-6)
            print(f"  {structure:>16s}  σ={sigma_target:.3f}   "
                  f"{e['frob_rel_err_mean']:>10.3f}  "
                  f"{e['cosine_mean']:>+8.3f}  "
                  f"{e['best_rescaled_err_mean']:>13.3f}  "
                  f"{e['magnitude_ratio_mean']:>10.4f}")
        print()

# Sanity check: signal-vs-noise per entry at measured σ
print("=" * 88)
print("Per-entry signal-vs-noise sanity check (using dense case, K562)")
print("=" * 88)
r = results["K562_essential"]["dense"]
noise_free = next(e for e in r if e["sigma"] == 0)
# Signal per entry: use ‖J_true‖ / sqrt(d²) as a rough per-entry scale
# We don't store J_true, so recompute a representative J to get its per-entry std
J_rep = draw_J(30, seed=20260810, structure="dense")
signal_per_entry_std = float(np.std(J_rep))
print(f"synthetic J_true per-entry std: {signal_per_entry_std:.3f}")
print(f"measured σ (per Δz entry, from within-guide bootstrap): 0.266")
print(f"→ noise-to-signal ratio at operator entry scale is NOT directly comparable")
print(f"  because operator entries are estimated from d² parameters × n·d observations.")
print(f"  What matters: cosine at measured σ (above). If cosine ≈ 0, no direction info regardless of magnitude.")
