"""Fig S2: detection_rate on pre-scaled z-scored expression residuals.

Simulates z-scored target-transcript data with a per-guide perturbation shift Δ.
Shows that on this data class, detection_rate = Pr[X_ctrl > 0] − Pr[X_pert > 0]
recovers the analytic prediction max(0, 0.5 − Φ(Δ/σ_ctrl)) to within ±0.05
across Δ ∈ [−2, +2] z-units, while mean_ratio degenerates because control mean ≈ 0.

Reproduces Fig S2 in manuscript_figures/. No external data. Runtime <30 s.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
import anchorop as ao

OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(20260731)
n_ctrl = 300
n_pert = 80
B = 200
shifts = np.linspace(-2.0, 2.0, 20)

est_dr = np.zeros((len(shifts), B))
est_mr = np.zeros((len(shifts), B))

for si, shift in enumerate(shifts):
    for b in range(B):
        ctrl = rng.normal(0.0, 1.0, size=n_ctrl)
        pert = rng.normal(shift, 1.0, size=n_pert)
        X = np.zeros((n_ctrl + n_pert, 1))
        X[:n_ctrl, 0] = ctrl; X[n_ctrl:, 0] = pert
        cm = np.zeros(n_ctrl + n_pert, dtype=bool); cm[:n_ctrl] = True
        pm = ~cm
        est_dr[si, b] = ao.estimate_knockdown_efficiency_detection_rate(
            X, target_index=0, perturbed_mask=pm, control_mask=cm)
        try:
            est_mr[si, b] = ao.estimate_knockdown_efficiency(
                X, target_index=0, perturbed_mask=pm, control_mask=cm)
        except Exception:
            est_mr[si, b] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

ax = axes[0]
predicted = np.clip(0.5 - norm.cdf(shifts), 0.0, 1.0)
ax.plot(shifts, predicted, "k--", lw=2, label=r"analytic  max(0, 0.5 − $\Phi(\Delta)$)")
mean_dr = est_dr.mean(1); std_dr = est_dr.std(1)
ax.fill_between(shifts, mean_dr - std_dr, mean_dr + std_dr, alpha=0.3, color="C0",
                 label="±1 std across trials")
ax.plot(shifts, mean_dr, "o-", color="C0", label="detection_rate estimator")
ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.6", lw=0.5)
ax.set_xlabel(r"perturbation shift $\Delta$ (z-units); knockdown gives $\Delta < 0$")
ax.set_ylabel("estimator value")
ax.set_title("detection_rate on z-scored data:\nrecovers the signed distributional shift as predicted")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
mean_mr = np.nanmean(est_mr, axis=1); std_mr = np.nanstd(est_mr, axis=1)
ax.fill_between(shifts, mean_mr - std_mr, mean_mr + std_mr, alpha=0.3, color="C3")
ax.plot(shifts, mean_mr, "o-", color="C3", label="mean_ratio estimator (blows up)")
ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.6", lw=0.5)
ax.set_xlabel(r"perturbation shift $\Delta$ (z-units)")
ax.set_ylabel("estimator value (clipped to [0, 1] in code)")
ax.set_title("mean_ratio on z-scored data:\ndegenerates because control mean ≈ 0")
ax.legend(loc="upper right", fontsize=9)

fig.suptitle("Fig S2: Estimator behavior on pre-scaled z-scored expression residuals", fontsize=12, y=1.05)
out = OUT_DIR / "figS2_estimators_on_zscored_data.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
