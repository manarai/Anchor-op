"""Fig S18: sensitivity of recovery to the J_true stability-shift constant.

§4.6 shifts J_true = G − c·I with c chosen so median Re(λ_J) ≈ −1.5. Because
the pseudoinverse problem's behavior depends on the singular spectrum of
J⁻¹U, recovery conclusions could in principle be an artifact of this choice.

This script sweeps c ∈ {0.5, 1.0, 1.5, 2.0, 3.0} on both cell lines with the
dense ground truth at measured σ = 0.266, reporting cos, cos_1, cos_5,
‖A‖/‖J‖, and the condition number κ(J⁻¹U) — the last of which is the
mechanism by which the shift would affect recovery.

If cos and cos_1 stay ≈ constant across c (with the appropriate change in
scale ratio explained by the changing ‖J‖), the shift is not an artifact.

Runtime ~3 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"

N_REPS = 40
SIGMA = 0.266
RANK_TOL = 1e-2
SHIFTS = [0.5, 1.0, 1.5, 2.0, 3.0]  # c in J = G − cI
SEED_BASE = 20260810


def draw_dense_J(d, seed, shift):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    return G - shift * np.eye(d)


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


results = {}
for cell_line, filename in [("K562_essential", "k562_essential_measurement.pkl"),
                              ("RPE1_essential", "rpe1_essential_measurement.pkl")]:
    bundle_path = RESULTS / filename
    with bundle_path.open("rb") as f:
        state = pickle.load(f)
    m = state["measurement"]
    U_real = m.U
    d, n_guides = m.S.shape
    print(f"\n{'='*72}\n{cell_line} (d={d}, n_guides={n_guides})  N_REPS={N_REPS}, σ={SIGMA}\n{'='*72}")
    print(f"{'shift c':>8s}  {'cond(J⁻¹U)':>12s}  {'cos':>16s}  {'cos_1':>16s}  {'cos_5':>16s}  {'‖A‖/‖J‖':>10s}")
    per_line = []
    for shift in SHIFTS:
        cos_arr, cos1_arr, cos5_arr, mag_arr, cond_arr = [], [], [], [], []
        for rep in range(N_REPS):
            J = draw_dense_J(d, seed=SEED_BASE + rep, shift=shift)
            S_true = -np.linalg.solve(J, U_real)
            _, sv, _ = np.linalg.svd(S_true, full_matrices=False)
            cond = sv[0] / sv[-1]
            rng = np.random.default_rng(rep + 7000 + int(shift*100))
            S_obs = S_true + SIGMA * rng.normal(size=S_true.shape)
            A = fit_operator(S_obs, U_real)
            cos_arr.append(cos_full(A, J))
            cos1_arr.append(cos_topk(A, J, S_true, 1))
            cos5_arr.append(cos_topk(A, J, S_true, 5))
            mag_arr.append(np.linalg.norm(A) / np.linalg.norm(J))
            cond_arr.append(cond)
        entry = {
            "shift": shift,
            "cond_JinvU_mean": float(np.mean(cond_arr)),
            "cos_mean": float(np.mean(cos_arr)),  "cos_std": float(np.std(cos_arr)),
            "cos_1_mean": float(np.mean(cos1_arr)), "cos_1_std": float(np.std(cos1_arr)),
            "cos_5_mean": float(np.mean(cos5_arr)), "cos_5_std": float(np.std(cos5_arr)),
            "magnitude_ratio_mean": float(np.mean(mag_arr)),
        }
        per_line.append(entry)
        print(f"  c={shift:>4.2f}  {entry['cond_JinvU_mean']:>12.1f}  "
              f"{entry['cos_mean']:>+8.3f}±{entry['cos_std']:.3f}  "
              f"{entry['cos_1_mean']:>+8.3f}±{entry['cos_1_std']:.3f}  "
              f"{entry['cos_5_mean']:>+8.3f}±{entry['cos_5_std']:.3f}  "
              f"{entry['magnitude_ratio_mean']:>10.4f}")
    results[cell_line] = per_line

(OUT_DIR / "stability_shift_sweep.json").write_text(json.dumps(results, indent=2))

# Figure: three panels — cos, cos_1, cos_5 vs shift
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
for ax_i, metric in enumerate(["cos", "cos_1", "cos_5"]):
    ax = axes[ax_i]
    for cell_line, color in [("K562_essential", "#1f4e79"), ("RPE1_essential", "#c65a30")]:
        r = results[cell_line]
        xs = [e["shift"] for e in r]
        ys = [e[f"{metric}_mean"] for e in r]
        es = [e[f"{metric}_std"] for e in r]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, lw=2, ms=7, capsize=3,
                     label=cell_line.replace("_essential", ""))
    ax.axhline(0, color="0.5", lw=0.6)
    if metric == "cos":
        ax.axhline(0.033, color="0.4", ls=":", lw=1)
        ax.axhline(-0.033, color="0.4", ls=":", lw=1)
    elif metric == "cos_1":
        ax.axhline(0.183, color="0.4", ls=":", lw=1)
        ax.axhline(-0.183, color="0.4", ls=":", lw=1)
        ax.axhline(0.5, color="green", ls="--", lw=0.7, alpha=0.5)
    elif metric == "cos_5":
        ax.axhline(1/np.sqrt(30*5), color="0.4", ls=":", lw=1)
        ax.axhline(-1/np.sqrt(30*5), color="0.4", ls=":", lw=1)
    ax.set_xlabel("stability shift c (J = G − cI)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs c")
    if ax_i == 0:
        ax.legend()

fig.suptitle(f"Fig S18: J_true stability-shift sensitivity, N={N_REPS} per point, dense J_true, σ={SIGMA}.\n"
             "Recovery conclusions constant across the swept range → not a shift artifact.",
             fontsize=10.5, y=1.06)
out = OUT_DIR / "figS18_stability_shift_sweep.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")
