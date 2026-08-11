"""Fig S15: per-direction operator recovery on the top-k singular directions of S.

Reviewer critique on §3.5's global cosine (≈0.04 at measured σ): real Δz noise is
heteroscedastic. If the signal concentrates in the top few singular directions of the
sensitivity matrix S, recovery in *those* directions could be much better than the
global cosine implies, and might yield a partial-positive result worth reporting.

This script measures exactly that. Given the fit `A = -U · pinv(S)` and the ground
truth `J_true`, we compute the SVD of the noise-free `S_true = -J_true^{-1} U`, take
the top-k left singular vectors `U_S(:, 1:k)` as the k best-illuminated response
directions, and compute:

  cos_k(A, J_true) = ⟨A U_S(:,1:k), J_true U_S(:,1:k)⟩_F
                     / (‖A U_S(:,1:k)‖_F · ‖J_true U_S(:,1:k)‖_F)

This is the cosine of the two operators' action on the k dominant response directions.
If cos_k → 1 for small k while global cos ≈ 0, the fit does have direction content in
the well-illuminated subspace (only the ill-conditioned directions are noise-dominated),
which is a partial-positive result.

Sweeps k ∈ {1, 2, 3, 5, 10, 15, 20, 25, 30} at σ ∈ {0.005, 0.05, 0.266} for the four
J_true structures on both cell lines.

Runtime ~2 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

RANK_TOL = 1e-2
N_REPS = 15
KS = [1, 2, 3, 5, 10, 15, 20, 25, 30]
SIGMAS = [0.005, 0.025, 0.10, 0.266]


def draw_J(d, seed, structure="dense"):
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


def fit_operator(S, U, rank_tol=RANK_TOL):
    S_pinv, *_ = regularized_pseudoinverse(S, method="tsvd", parameter="path", rank_tol=rank_tol)
    return -U @ S_pinv


def per_direction_cosine(A, J_true, S_true, k):
    """cos(A U_S(:,1:k), J_true U_S(:,1:k)) in Frobenius inner product."""
    Us, sv, _ = np.linalg.svd(S_true, full_matrices=False)
    Uk = Us[:, :k]
    AUk = A @ Uk
    JUk = J_true @ Uk
    inner = float(np.sum(AUk * JUk))
    nA = float(np.linalg.norm(AUk)); nJ = float(np.linalg.norm(JUk))
    return inner / max(nA * nJ, 1e-30)


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
    print(f"\n{'='*70}\n{cell_line} (d={d}, n_guides={n_guides})\n{'='*70}")
    per_line = {}
    for structure in STRUCTURES:
        print(f"\n  structure: {structure}")
        print(f"  {'σ':>7s} " + "".join(f"{'k='+str(k):>10s}" for k in KS))
        cell_struct = {}
        for sigma in SIGMAS:
            per_k = {k: [] for k in KS}
            for rep in range(N_REPS):
                J_true = draw_J(d, seed=20260810 + rep, structure=structure)
                S_true = -np.linalg.solve(J_true, U_real)
                rng = np.random.default_rng(rep + 10000)
                S_obs = S_true + sigma * rng.normal(size=S_true.shape)
                A_fit = fit_operator(S_obs, U_real)
                for k in KS:
                    per_k[k].append(per_direction_cosine(A_fit, J_true, S_true, k))
            entry = {"sigma": sigma,
                      "cos_k_mean": {k: float(np.mean(per_k[k])) for k in KS},
                      "cos_k_std":  {k: float(np.std(per_k[k])) for k in KS}}
            cell_struct[str(sigma)] = entry
            print(f"  σ={sigma:.3f}  " +
                  "".join(f"{entry['cos_k_mean'][k]:>+10.3f}" for k in KS))
        per_line[structure] = cell_struct
    results[cell_line] = per_line

(OUT_DIR / "per_direction_recovery.json").write_text(json.dumps(results, indent=2))

# Figure: for each structure (4 cols), plot cos_k vs k at each σ (K562 solid, RPE1 dashed)
fig, axes = plt.subplots(1, 4, figsize=(20, 4.7), constrained_layout=True)
sig_colors = {0.005: "#5aa02c", 0.025: "#e0a020", 0.10: "#c65a30", 0.266: "#7a0d0d"}
structure_titles = {
    "dense":         "dense J (Ginibre)",
    "sparse_10pct":  "sparse J (10%)",
    "sparse_2pct":   "very sparse J (2%)",
    "low_rank_5":    "low-rank J (rank 5)",
}
for col, structure in enumerate(STRUCTURES):
    ax = axes[col]
    for sigma in SIGMAS:
        color = sig_colors[sigma]
        for cell_line, marker, ls, lw in [("K562_essential", "o", "-", 2),
                                            ("RPE1_essential", "s", "--", 1.5)]:
            e = results[cell_line][structure][str(sigma)]
            ys = [e["cos_k_mean"][k] for k in KS]
            es = [e["cos_k_std"][k] for k in KS]
            label = f"{cell_line.replace('_essential','')}, σ={sigma}"
            ax.errorbar(KS, ys, yerr=es, fmt=marker + ls, color=color, lw=lw, ms=6,
                         alpha=0.9, capsize=3, label=label)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axhline(1.0, color="green", ls=":", lw=1, alpha=0.6)
    ax.axhline(0.5, color="0.6", ls=":", lw=0.7, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("k (top-k singular directions of S restricted)")
    if col == 0:
        ax.set_ylabel(r"cos$(A U_S^{(k)}, J_{\rm true} U_S^{(k)})$")
    ax.set_title(f"({['a','b','c','d'][col]}) {structure_titles[structure]}")
    ax.set_ylim(-0.15, 1.05)
    if col == 3:
        ax.legend(loc="lower left", fontsize=7, ncol=1)

fig.suptitle(
    "Fig S15: Per-direction operator recovery restricted to the top-k best-illuminated singular directions of S.\n"
    "If cos_k rises above the global cosine at small k, the fit HAS direction content in the well-conditioned subspace.\n"
    "K562 = solid + circles, RPE1 = dashed + squares. Colors = per-entry noise σ.",
    fontsize=10.5, y=1.06)
out = OUT_DIR / "figS15_per_direction_recovery.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")

print("\n" + "="*80)
print("SUMMARY AT MEASURED σ = 0.266 (K562, all structures)")
print("="*80)
for structure in STRUCTURES:
    e = results["K562_essential"][structure]["0.266"]
    top_k_str = ", ".join(f"k={k}:{e['cos_k_mean'][k]:+.3f}" for k in [1, 3, 5, 10, 30])
    print(f"  {structure:>16s}  {top_k_str}")
print("\nInterpretation:")
print("  cos_1 or cos_3 >> cos_30 → partial-positive: fit has direction content in top few directions")
print("  cos_k flat across k    → no per-direction rescue; §3.5 global claim generalizes")
