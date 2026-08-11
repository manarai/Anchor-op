"""Fig S3: cross-format estimator simulation on count data.

Sweeps (λ_ctrl, κ) under a Poisson observation model. For each grid point,
draws n_ctrl + n_pert cell counts and computes three efficiency estimators
(mean_ratio, detection_rate, poisson_mle). Reports bias and standard
deviation across trials.

Reproduces Fig S3 and `figS3_estimator_simulation.npz` in manuscript_figures/.
No external data required. Runtime ~2 minutes.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import anchorop as ao

OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(20260731)
n_ctrl = 300
n_pert = 80
B = 300

lambdas = np.logspace(-3, 1, 14)
kappas = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])


def trial(lam_ctrl, kappa, rng):
    ctrl = rng.poisson(lam_ctrl, size=n_ctrl).astype(float)
    lam_pert = max(1e-9, (1 - kappa) * lam_ctrl)
    pert = rng.poisson(lam_pert, size=n_pert).astype(float)
    X = np.zeros((n_ctrl + n_pert, 1))
    X[:n_ctrl, 0] = ctrl
    X[n_ctrl:, 0] = pert
    cm = np.zeros(n_ctrl + n_pert, dtype=bool); cm[:n_ctrl] = True
    pm = ~cm
    return (
        ao.estimate_knockdown_efficiency(X, target_index=0, perturbed_mask=pm, control_mask=cm),
        ao.estimate_knockdown_efficiency_detection_rate(X, target_index=0, perturbed_mask=pm, control_mask=cm),
        ao.estimate_knockdown_efficiency_poisson_mle(X, target_index=0, perturbed_mask=pm, control_mask=cm),
    )


est_mr = np.zeros((len(kappas), len(lambdas), B))
est_dr = np.zeros_like(est_mr)
est_pm = np.zeros_like(est_mr)
for ki, k in enumerate(kappas):
    for li, l in enumerate(lambdas):
        for b in range(B):
            m, d, p = trial(l, k, rng)
            est_mr[ki, li, b] = m
            est_dr[ki, li, b] = d
            est_pm[ki, li, b] = p

bias_mr = est_mr.mean(-1) - kappas[:, None]
bias_dr = est_dr.mean(-1) - kappas[:, None]
bias_pm = est_pm.mean(-1) - kappas[:, None]
std_mr = est_mr.std(-1)
std_dr = est_dr.std(-1)
std_pm = est_pm.std(-1)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
xlab = [f"{l:.0e}" if (l >= 10 or l < 0.01) else f"{l:.2f}" for l in lambdas]
ylab = [f"{k:.1f}" for k in kappas]
for col, (name, bias, std) in enumerate([
    ("mean_ratio", bias_mr, std_mr),
    ("detection_rate (raw shift)", bias_dr, std_dr),
    ("poisson_mle", bias_pm, std_pm),
]):
    ax = axes[0, col]
    im = ax.imshow(bias, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5, origin="lower")
    ax.set_xticks(range(len(lambdas))); ax.set_xticklabels(xlab, rotation=40, fontsize=8)
    ax.set_yticks(range(len(kappas))); ax.set_yticklabels(ylab)
    ax.set_xlabel(r"baseline $\lambda_{\rm ctrl}$")
    ax.set_ylabel(r"true $\kappa$" if col == 0 else "")
    ax.set_title(f"{name}\nbias  E[$\\hat\\kappa$] − $\\kappa$")
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, col]
    im = ax.imshow(std, aspect="auto", cmap="viridis", vmin=0, vmax=0.35, origin="lower")
    ax.set_xticks(range(len(lambdas))); ax.set_xticklabels(xlab, rotation=40, fontsize=8)
    ax.set_yticks(range(len(kappas))); ax.set_yticklabels(ylab)
    ax.set_xlabel(r"baseline $\lambda_{\rm ctrl}$")
    ax.set_ylabel(r"true $\kappa$" if col == 0 else "")
    ax.set_title(f"std($\\hat\\kappa$) across {B} trials")
    fig.colorbar(im, ax=ax, shrink=0.85)

fig.suptitle(
    f"Fig S3: Efficiency-estimator bias and variance under Poisson observation model\n"
    f"(n_ctrl={n_ctrl}, n_pert={n_pert}, {B} trials per grid point)",
    fontsize=12, y=1.03,
)
out = OUT_DIR / "figS3_estimator_simulation.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
np.savez(out.with_suffix(".npz"),
         lambdas=lambdas, kappas=kappas,
         est_mr=est_mr, est_dr=est_dr, est_pm=est_pm,
         n_ctrl=n_ctrl, n_pert=n_pert, B=B)

print(f"wrote {out}")
print(f"summary:  mean_ratio |bias| = {np.abs(bias_mr).mean():.4f}")
print(f"          detection_rate |bias| = {np.abs(bias_dr).mean():.4f}")
print(f"          poisson_mle |bias| = {np.abs(bias_pm).mean():.4f}")
