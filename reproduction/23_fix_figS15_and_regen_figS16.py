"""(a) Regenerate Fig. S15 with fixed suptitle/legend placement.
(b) Regenerate Fig. S16 at per-dataset σ (K562 0.240, RPE1 0.352).

The originals used σ = 0.266 (early K562 anchor); the manuscript now cites
per-dataset σ throughout. This script updates both PNGs to the values the
manuscript body claims.

Runtime ~6 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"

DATASET_SIGMA = {"K562_essential": 0.240, "RPE1_essential": 0.352}
BUNDLE = {"K562_essential": "k562_essential_measurement.pkl",
           "RPE1_essential": "rpe1_essential_measurement.pkl"}
RANK_TOL = 1e-2
SEED_BASE = 20260810


def draw_dense_J(d, seed):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    return G - 1.5 * np.eye(d)


def draw_J(d, seed, structure):
    rng = np.random.default_rng(seed)
    if structure == "dense":
        return draw_dense_J(d, seed)
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


def cos_topk(A, J, S_true, k):
    Us, _, _ = np.linalg.svd(S_true, full_matrices=False)
    Uk = Us[:, :k]
    AUk = A @ Uk; JUk = J @ Uk
    return float(np.sum(AUk * JUk)) / max(np.linalg.norm(AUk) * np.linalg.norm(JUk), 1e-30)


# =====================================================================
# (b) Fig S16 — pipeline-matched empirical null at per-dataset σ
# =====================================================================
N_REPS_S16 = 200
print("Regenerating Fig S16 at per-dataset σ...")

def empirical_null_data(cell_line, filename, sigma):
    with (RESULTS / filename).open("rb") as f:
        state = pickle.load(f)
    U_real = state["measurement"].U
    d, n_guides = U_real.shape[0], U_real.shape[1]
    Js, S_trues, A_fits = [], [], []
    rng_n = np.random.default_rng(SEED_BASE + 999)
    print(f"  {cell_line}: fitting N={N_REPS_S16} replicates at σ={sigma}...")
    for r in range(N_REPS_S16):
        J = draw_dense_J(d, seed=SEED_BASE + r)
        S_true = -np.linalg.solve(J, U_real)
        S_obs = S_true + sigma * rng_n.normal(size=S_true.shape)
        Js.append(J); S_trues.append(S_true); A_fits.append(fit_operator(S_obs, U_real))
    real = {"cos": np.array([cos_full(A_fits[r], Js[r]) for r in range(N_REPS_S16)]),
             "cos_1": np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 1) for r in range(N_REPS_S16)]),
             "cos_5": np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 5) for r in range(N_REPS_S16)])}
    null_cross = {"cos": [], "cos_1": [], "cos_5": []}
    for shift in range(1, 11):
        for r in range(N_REPS_S16):
            rp = (r + shift) % N_REPS_S16
            null_cross["cos"].append(cos_full(A_fits[r], Js[rp]))
            null_cross["cos_1"].append(cos_topk(A_fits[r], Js[rp], S_trues[rp], 1))
            null_cross["cos_5"].append(cos_topk(A_fits[r], Js[rp], S_trues[rp], 5))
    null_cross = {k: np.array(v) for k, v in null_cross.items()}
    null_shuf = {"cos": [], "cos_1": [], "cos_5": []}
    rng_su = np.random.default_rng(SEED_BASE + 888)
    for r in range(N_REPS_S16):
        perm = rng_su.permutation(n_guides)
        S_ts = -np.linalg.solve(Js[r], U_real[:, perm])
        S_obs = S_ts + sigma * rng_su.normal(size=S_ts.shape)
        A_sh = fit_operator(S_obs, U_real)
        null_shuf["cos"].append(cos_full(A_sh, Js[r]))
        null_shuf["cos_1"].append(cos_topk(A_sh, Js[r], S_trues[r], 1))
        null_shuf["cos_5"].append(cos_topk(A_sh, Js[r], S_trues[r], 5))
    null_shuf = {k: np.array(v) for k, v in null_shuf.items()}
    return real, null_cross, null_shuf


fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for row, (cell_line, filename) in enumerate(BUNDLE.items()):
    sigma = DATASET_SIGMA[cell_line]
    real, null_c, null_s = empirical_null_data(cell_line, filename, sigma)
    colors_real = {"K562_essential": "#1f4e79", "RPE1_essential": "#c65a30"}
    color = colors_real[cell_line]
    for col, metric in enumerate(["cos", "cos_1", "cos_5"]):
        ax = axes[row, col]
        rr = real[metric]; nc = null_c[metric]; ns = null_s[metric]
        bins = np.linspace(min(rr.min(), nc.min(), ns.min()) - 0.02,
                            max(rr.max(), nc.max(), ns.max()) + 0.02, 40)
        ax.hist(ns, bins=bins, color="#8899cc", alpha=0.55, density=True,
                 label="shuffled-U null" if (row == 0 and col == 0) else None)
        ax.hist(nc, bins=bins, color="#7a7a7a", alpha=0.55, density=True,
                 label="cross-rep null" if (row == 0 and col == 0) else None)
        ax.hist(rr, bins=bins, color=color, alpha=0.72, density=True,
                 label="real: cos(A_r, J_r)" if (row == 0 and col == 0) else None)
        ax.axvline(float(rr.mean()), color=color, ls="--", lw=1.3)
        ax.axvline(0.0, color="0.5", lw=0.5)
        title_label = {"cos": "cos(A, J)", "cos_1": "cos_1(A, J | U_S top-1)",
                        "cos_5": "cos_5(A, J | U_S top-5)"}[metric]
        ax.set_title(f"{cell_line.replace('_essential','')}: {title_label}")
        ax.set_xlabel("cosine"); ax.set_ylabel("density")
        ax.text(0.03, 0.97, f"σ = {sigma}\nreal: {rr.mean():+.3f}\ncross-rep: {nc.mean():+.3f}\nshuf-U: {ns.mean():+.3f}",
                 transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))
        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=8)

fig.suptitle(f"Fig S16. Pipeline-matched empirical null (N={N_REPS_S16} replicates, dense J_true, per-dataset σ)",
              fontsize=12, y=1.02)
out = OUT_DIR / "figS16_empirical_null.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")


# =====================================================================
# (a) Fig S15 — fix suptitle/legend overlap; redraw with per-dataset σ context
# =====================================================================
print("\nRegenerating Fig S15 with fixed layout...")
KS = [1, 2, 3, 5, 10, 15, 20, 25, 30]
SIGMAS_S15 = [0.005, 0.025, 0.10, 0.266]  # keep original grid for continuity
N_REPS_S15 = 15
STRUCTURES = ["dense", "sparse_10pct", "sparse_2pct", "low_rank_5"]

struct_titles = {"dense": "dense J (Ginibre)",
                  "sparse_10pct": "sparse J (10%)",
                  "sparse_2pct": "very sparse J (2%)",
                  "low_rank_5": "low-rank J (rank 5)"}
sig_colors = {0.005: "#5aa02c", 0.025: "#e0a020", 0.10: "#c65a30", 0.266: "#7a0d0d"}

results = {}
for cell_line, filename in BUNDLE.items():
    with (RESULTS / filename).open("rb") as f:
        state = pickle.load(f)
    U_real = state["measurement"].U
    d, n_guides = U_real.shape
    print(f"  {cell_line}: computing cos_k across structures and σ grid...")
    per_line = {}
    for structure in STRUCTURES:
        struct_res = {}
        for sigma in SIGMAS_S15:
            per_k = {k: [] for k in KS}
            for rep in range(N_REPS_S15):
                J = draw_J(d, seed=SEED_BASE + rep, structure=structure)
                S_true = -np.linalg.solve(J, U_real)
                rng = np.random.default_rng(rep + 10000)
                S_obs = S_true + sigma * rng.normal(size=S_true.shape)
                A = fit_operator(S_obs, U_real)
                for k in KS:
                    per_k[k].append(cos_topk(A, J, S_true, k))
            struct_res[sigma] = {"cos_k_mean": {k: float(np.mean(per_k[k])) for k in KS},
                                 "cos_k_std":  {k: float(np.std(per_k[k])) for k in KS}}
        per_line[structure] = struct_res
    results[cell_line] = per_line

# Draw with better layout: 1x4 grid, larger figsize, legend outside plot area, no overlapping suptitle
fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)
for col, structure in enumerate(STRUCTURES):
    ax = axes[col]
    for sigma in SIGMAS_S15:
        color = sig_colors[sigma]
        for cell_line, marker, ls, lw in [("K562_essential", "o", "-", 2),
                                            ("RPE1_essential", "s", "--", 1.5)]:
            m = results[cell_line][structure][sigma]["cos_k_mean"]
            std = results[cell_line][structure][sigma]["cos_k_std"]
            ys = [m[k] for k in KS]; es = [std[k] for k in KS]
            ax.errorbar(KS, ys, yerr=es, fmt=marker + ls, color=color, lw=lw, ms=6,
                        alpha=0.9, capsize=3,
                        label=f"{cell_line.replace('_essential','')}, σ={sigma}")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axhline(1.0, color="green", ls=":", lw=1, alpha=0.6)
    ax.axhline(0.5, color="0.4", ls=":", lw=0.7, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("k (top-k singular directions of S restricted)")
    if col == 0:
        ax.set_ylabel(r"cos$_k$(A, J_true) under oracle U_S decomposition")
    ax.set_title(f"({['a','b','c','d'][col]}) {struct_titles[structure]}")
    ax.set_ylim(-0.15, 1.05)

# Single legend in bottom-right panel, only once
axes[-1].legend(loc="upper right", fontsize=7, ncol=1, framealpha=0.9)

# Push suptitle above the axes so it does not overlap titles
fig.suptitle(
    "Fig S15. Per-direction operator recovery restricted to top-k best-illuminated singular directions of S "
    "(oracle decomposition; U_S from noise-free S_true).\n"
    "K562 = solid + circles, RPE1 = dashed + squares. Colors = per-entry noise σ.",
    fontsize=11, y=1.06)
out = OUT_DIR / "figS15_per_direction_recovery.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
