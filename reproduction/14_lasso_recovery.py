"""Fig S14: does a sparsity-aware fit (matrix LASSO) recover a sparse J at Replogle scale?

The §3.5 result — that anchor-op's default TSVD fit collapses to zero at measured
noise — cannot rule out sparsity-aware fits in principle. This script tests the
natural counter-hypothesis directly: given a 2%-sparse ground truth `J_true`
(18 nonzero entries in a 30×30 matrix, comfortably below the compressed-sensing
regime with n·d ≈ 5,640 observations), does a row-wise L1-regularized fit
recover where TSVD fails?

Setup mirrors reproduction/13:
  - Real Replogle U (K562 n=188, RPE1 n=153) and real κ
  - Synthetic 2%-sparse J_true, scaled to spectral radius ≈ 1.5
  - S_obs = -J_true^{-1} U + σ · noise
  - Fit row-by-row: for each row j of J, solve  min_v ||v S + U[j,:]||² + λ ||v||₁
    (sklearn.linear_model.Lasso with fit_intercept=False)
  - λ swept over a grid; report best over λ

Metrics reported per (σ, λ):
  - ‖A_lasso − J_true‖_F / ‖J_true‖_F
  - cos(A_lasso, J_true)
  - support recovery: fraction of true-nonzero entries with |A[i,j]| > threshold,
    and fraction of true-zero entries with |A[i,j]| < threshold

Interpretation:
  - If cos jumps from 0.05 (TSVD) to > 0.5 (LASSO) at measured σ → sparsity-aware fits
    recover where TSVD cannot; §3.5 claim narrows to 'the tool's default TSVD fit'.
  - If cos stays < 0.1 → sparsity does not help at this noise budget; §3.5 claim
    generalizes across fitting approaches.

Requires pickled measurement bundles from `../results/`. Runtime ~5 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Lasso

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

N_REPS = 8
SPARSITY = 0.02
SIGMAS = [0.005, 0.025, 0.10, 0.266]
LAMBDAS = np.logspace(-3, 0, 12)


def draw_sparse_J(d, seed, density=0.02):
    rng = np.random.default_rng(seed)
    mask = rng.random((d, d)) < density
    G = np.zeros((d, d))
    G[mask] = rng.normal(size=int(mask.sum())) / np.sqrt(max(mask.sum(), 1) / d)
    return G - 1.5 * np.eye(d)


def lasso_fit(S, U, lam):
    """Row-wise L1 fit: for each row j of J, minimize ||v · S + U[j,:]||² + λ ||v||₁."""
    d = S.shape[0]
    A = np.zeros((d, d))
    for j in range(d):
        # Lasso solves: min (1/2n) ||y - X β||² + α ||β||_1
        # We want: min ||v · S + U[j,:]||² + λ ||v||_1
        #     ⇔  min ||S^T v^T - (-U[j,:])||²  + λ ||v||_1
        # sklearn Lasso: X = S.T (n_samples=n_guides × n_features=d), y = -U[j,:]
        X = S.T  # (n_guides, d)
        y = -U[j, :]  # (n_guides,)
        # sklearn α = λ / (2 n_samples) mapping to our λ
        alpha = lam / (2 * X.shape[0])
        clf = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000, tol=1e-5)
        clf.fit(X, y)
        A[j, :] = clf.coef_
    return A


def support_metrics(A, J_true, threshold=1e-3):
    true_nz = np.abs(J_true) > threshold
    est_nz = np.abs(A) > threshold
    tp = int(np.sum(true_nz & est_nz))
    fp = int(np.sum(~true_nz & est_nz))
    fn = int(np.sum(true_nz & ~est_nz))
    tn = int(np.sum(~true_nz & ~est_nz))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics(A, J_true):
    frob = float(np.linalg.norm(A - J_true) / np.linalg.norm(J_true))
    inner = float(np.sum(A * J_true))
    nA = float(np.linalg.norm(A)); nJ = float(np.linalg.norm(J_true))
    cos = inner / max(nA * nJ, 1e-30)
    return {"frob_rel_err": frob, "cosine": cos, "magnitude_ratio": nA / max(nJ, 1e-30)}


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
    print(f"\n=== {cell_line} (d={d}, n_guides={n_guides}, {SPARSITY*100:.0f}%-sparse J_true) ===")
    print(f"{'σ':>8s} {'best_λ':>10s}  {'cos':>10s}  {'‖A-J‖/‖J‖':>12s}  {'support prec/recall':>25s}")
    per_line = []
    for sigma in SIGMAS:
        # For each σ, sweep λ; report best (highest cos) result
        best = None
        for lam in LAMBDAS:
            coss, frobs, precs, recs = [], [], [], []
            for rep in range(N_REPS):
                J_true = draw_sparse_J(d, seed=20260810 + rep, density=SPARSITY)
                S_true = -np.linalg.solve(J_true, U_real)
                rng_n = np.random.default_rng(rep + 10000)
                S_obs = S_true + sigma * rng_n.normal(size=S_true.shape)
                A_lasso = lasso_fit(S_obs, U_real, lam)
                met = metrics(A_lasso, J_true)
                sup = support_metrics(A_lasso, J_true)
                coss.append(met["cosine"]); frobs.append(met["frob_rel_err"])
                precs.append(sup["precision"]); recs.append(sup["recall"])
            entry_lam = {
                "sigma": sigma, "lam": float(lam),
                "cosine_mean": float(np.mean(coss)),
                "cosine_std": float(np.std(coss)),
                "frob_rel_err_mean": float(np.mean(frobs)),
                "support_precision_mean": float(np.mean(precs)),
                "support_recall_mean": float(np.mean(recs)),
            }
            if best is None or entry_lam["cosine_mean"] > best["cosine_mean"]:
                best = entry_lam
        per_line.append(best)
        print(f"{sigma:>8.4f} {best['lam']:>10.4f}  {best['cosine_mean']:>+7.3f}±{best['cosine_std']:.3f}  "
              f"{best['frob_rel_err_mean']:>12.3f}  "
              f"{best['support_precision_mean']:>10.3f}/{best['support_recall_mean']:.3f}")
    results[cell_line] = per_line

(OUT_DIR / "lasso_recovery.json").write_text(json.dumps(results, indent=2))

# Plot: cos(LASSO) vs σ, both cell lines; overlay TSVD baseline (~0.05 at σ=0.266)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)

ax = axes[0]
for cell_line, color in [("K562_essential", "#1f4e79"), ("RPE1_essential", "#c65a30")]:
    r = results[cell_line]
    sig = [e["sigma"] for e in r]
    cos = [e["cosine_mean"] for e in r]
    cos_std = [e["cosine_std"] for e in r]
    ax.errorbar(sig, cos, yerr=cos_std, fmt="o-", color=color, lw=2, ms=7, capsize=3,
                 label=f"{cell_line.replace('_essential','')} — LASSO (best λ)")
    # TSVD reference: cosine at same σ for same 2% sparse structure (from Fig S13 numbers)
    tsvd_cos = {0.005: 0.71 if 'K562' in cell_line else 0.65,
                 0.025: 0.31 if 'K562' in cell_line else 0.25,
                 0.10: 0.075 if 'K562' in cell_line else 0.075,
                 0.266: 0.05 if 'K562' in cell_line else 0.031}
    ax.plot(sig, [tsvd_cos.get(s, np.nan) for s in sig], "s--", color=color, lw=1, ms=6, alpha=0.6,
             label=f"{cell_line.replace('_essential','')} — TSVD (default)")
ax.axhline(0, color="0.6", lw=0.5); ax.axhline(1.0, color="green", ls=":", lw=1)
ax.axvspan(0.20, 0.30, alpha=0.15, color="#c65a30", label="measured Replogle σ")
ax.set_xscale("symlog", linthresh=0.005)
ax.set_xlabel(r"per-entry noise σ")
ax.set_ylabel("cos(A, J_true) on 2%-sparse ground truth")
ax.set_title("(a) LASSO vs TSVD on sparse J_true\n(does sparsity-aware fit rescue recovery?)")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(-0.1, 1.05)

ax = axes[1]
for cell_line, color in [("K562_essential", "#1f4e79"), ("RPE1_essential", "#c65a30")]:
    r = results[cell_line]
    sig = [e["sigma"] for e in r]
    prec = [e["support_precision_mean"] for e in r]
    rec = [e["support_recall_mean"] for e in r]
    ax.plot(sig, prec, "o-", color=color, lw=2, ms=7, label=f"{cell_line.replace('_essential','')} — precision")
    ax.plot(sig, rec, "s--", color=color, lw=1.5, ms=6, alpha=0.7, label=f"{cell_line.replace('_essential','')} — recall")
ax.axvspan(0.20, 0.30, alpha=0.15, color="#c65a30")
ax.set_xscale("symlog", linthresh=0.005)
ax.set_xlabel(r"per-entry noise σ")
ax.set_ylabel("support recovery (precision, recall)")
ax.set_title("(b) LASSO support recovery on 2%-sparse J_true\n(does it find the right nonzero entries?)")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(-0.05, 1.05)

fig.suptitle("Fig S14: Does sparsity-aware (LASSO) fitting rescue operator recovery at Replogle-matched scale?",
             fontsize=11.5, y=1.03)
out = OUT_DIR / "figS14_lasso_recovery.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")

print("\n=== VERDICT ===")
for cell_line in results:
    at_measured = next(e for e in results[cell_line] if abs(e["sigma"] - 0.266) < 1e-6)
    print(f"{cell_line} at measured σ=0.266 (2%-sparse J_true):")
    print(f"  LASSO best cos = {at_measured['cosine_mean']:+.3f}  (TSVD baseline was ~0.05)")
    print(f"  LASSO support: precision={at_measured['support_precision_mean']:.3f}, recall={at_measured['support_recall_mean']:.3f}")
    if at_measured["cosine_mean"] > 0.3:
        print(f"  → SPARSITY-AWARE FIT RECOVERS. §3.5 claim narrows to 'the tool's default TSVD fit'.")
    elif at_measured["cosine_mean"] > 0.1:
        print(f"  → PARTIAL rescue. LASSO improves over TSVD but still below meaningful recovery.")
    else:
        print(f"  → NO RESCUE. Sparsity-aware fit also fails at measured noise. §3.5 claim generalizes.")
