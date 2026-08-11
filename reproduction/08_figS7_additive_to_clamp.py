"""Fig S7: additive→clamp interpolation sweep on synthetic d=6 ground truth.

For each α ∈ [0, 1], mixes an additive-input response with a hard-clamp Schur
response. Fits anchor-op, computes ||J_fit − J_true||_F, bin-split rel_diff,
and held-out ρ. Shows that both linearity diagnostics stay near zero across
the entire additive↔clamp axis (both endpoints are linear input→response maps),
while J_fit error rises from 0.02 to 0.78.

Reproduces Fig S7 in manuscript_figures/. No external data. Runtime ~2 min.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import anchorop as ao

OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)


def build_synthetic_J(d=6):
    J = np.zeros((d, d))
    J[0:2, 0:2] = np.array([[-0.4, 0.9], [-0.9, -0.4]])
    J[2:4, 2:4] = np.array([[-0.6, 1.2], [-1.2, -0.6]])
    J[4, 4] = -1.5
    J[5, 5] = -0.1
    rng_local = np.random.default_rng(7)
    J = J + 0.15 * rng_local.normal(size=(d, d))
    eig = np.linalg.eigvals(J).real
    if eig.max() > -0.05:
        J = J + (-0.05 - eig.max()) * np.eye(d)
    return J


def additive_response(J, kappa, target_idx):
    d = J.shape[0]
    u = np.zeros(d); u[target_idx] = -kappa
    return -np.linalg.solve(J, u)


def clamp_response(J, kappa, target_idx):
    d = J.shape[0]
    mask = np.ones(d, dtype=bool); mask[target_idx] = False
    J11 = J[np.ix_(mask, mask)]
    J12 = J[mask, target_idx]
    x = np.zeros(d)
    x[target_idx] = -kappa
    x[mask] = -np.linalg.solve(J11, J12 * x[target_idx])
    return x


def run_sweep(alpha_grid, J_true, n_guides=60, seed=20260806, noise=0.02):
    rng = np.random.default_rng(seed)
    d = J_true.shape[0]
    guide_targets = rng.integers(0, d, size=n_guides)
    kappas = 0.3 + 0.6 * rng.uniform(size=n_guides)
    x_add = np.column_stack([additive_response(J_true, k, t) for k, t in zip(kappas, guide_targets)])
    x_clm = np.column_stack([clamp_response(J_true, k, t) for k, t in zip(kappas, guide_targets)])
    U = np.column_stack([-k * np.eye(d)[t] for k, t in zip(kappas, guide_targets)])
    guide_names = [f"g{i}" for i in range(n_guides)]
    guide_effs = {n: float(kappas[i]) for i, n in enumerate(guide_names)}
    J_true_norm = float(np.linalg.norm(J_true))
    results = []
    for alpha in alpha_grid:
        S = (1 - alpha) * x_add + alpha * x_clm
        S = S + noise * rng.normal(size=S.shape)
        m = ao.measure_from_sensitivity(
            S, U, guide_names=guide_names, guide_efficiencies=guide_effs,
            reg="tsvd", reg_param="path", rank_tol=1e-6,
        )
        J_fit = -U @ np.linalg.pinv(S)
        J_err = float(np.linalg.norm(J_fit - J_true) / J_true_norm)
        lin = ao.linearity_check(m, threshold=0.25, n_null=100, null_seed=42)
        rho = ao.held_out_prediction_check(m, n_folds=5, seed=0, n_permutation_null=30, null_seed=100)
        results.append({
            "alpha": alpha,
            "J_err": J_err,
            "rel_diff": float(lin.relative_difference),
            "rel_diff_null": lin.null_median,
            "held_out_rho": rho.rho_pooled,
        })
    return results


J = build_synthetic_J(d=6)
alphas = np.concatenate([np.linspace(0, 0.5, 11), np.linspace(0.55, 1.0, 10)])
results = run_sweep(alphas.tolist(), J, n_guides=60, noise=0.02)

alpha_arr = np.array([r["alpha"] for r in results])
j_err = np.array([r["J_err"] for r in results])
rd = np.array([r["rel_diff"] for r in results])
rho = np.array([r["held_out_rho"] for r in results])

fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), constrained_layout=True)
ax.plot(alpha_arr, j_err, "o-", color="#c65a30", lw=2.2, ms=7,
        label=r"$||J_{\rm fit} - J_{\rm true}||_F / ||J_{\rm true}||_F$")
ax.plot(alpha_arr, rd, "s-", color="#1f4e79", lw=2.2, ms=7,
        label="bin-split rel_diff (linearity diagnostic)")
ax.plot(alpha_arr, rho, "^-", color="#4a2f71", lw=2.2, ms=7,
        label=r"held-out $\rho$ (linearity diagnostic)")
ax.axhline(0.25, color="red", ls=":", lw=1.2, label="preregistered linearity threshold (0.25)")
ax.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.6, label="zero-predictor baseline (ρ=1)")

# Overlay observed Replogle for reference
ax.axhline(1.466, color="#1f4e79", ls="-.", lw=1.2, alpha=0.6)
ax.text(1.02, 1.466, "K562 rel_diff = 1.47", color="#1f4e79", fontsize=8, va="center", ha="left")
ax.axhline(1.571, color="#3d78ac", ls="-.", lw=1.2, alpha=0.6)
ax.text(1.02, 1.571, "RPE1 rel_diff = 1.57", color="#3d78ac", fontsize=8, va="center", ha="left")

ax.annotate("pure additive input\n(anchor-op's assumption)", xy=(0.0, 0.03), xytext=(0.05, 0.30),
            fontsize=9, color="0.3", arrowprops=dict(arrowstyle="->", color="0.5", lw=1))
ax.annotate("pure hard clamp\n(CRISPRi mechanism\nat large κ)", xy=(1.0, 0.78), xytext=(0.72, 0.55),
            fontsize=9, color="0.3", arrowprops=dict(arrowstyle="->", color="0.5", lw=1))

ax.set_xlabel(r"$\alpha$ — additive↔clamp interpolation")
ax.set_ylabel("diagnostic / error value")
ax.set_title("Fig S7: additive→clamp sweep — J_fit bias up to 78%,\nbut linearity diagnostics stay near 0 (both endpoints are linear input→response maps)")
ax.set_ylim(-0.05, 1.75)
ax.set_xlim(-0.02, 1.30)
ax.legend(loc="center right", fontsize=8.5, framealpha=0.95)

out = OUT_DIR / "figS7_additive_to_clamp_sweep.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
