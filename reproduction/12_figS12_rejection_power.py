"""Fig S12: rejection-power surface for both linearity diagnostics.

At each (n_guides, κ_range, σ) point, compares synthetic linear vs synthetic
saturating (tanh sat=0.5) ground truth. Reports gap in held-out ρ and whether
it is 95%-CI detectable.

Two panels:
  (a) 2D power surface (n_guides × κ_range) at fixed sat=0.5, σ=0.266 (measured).
  (b) Noise sweep at three representative configs, mapped to required cells/guide.

Reproduces Fig S12. No external data. Runtime ~5 min.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

D = 30
RANK_TOL = 1e-2
NOISE_PER_ENTRY_MEASURED = 0.266
SIGMA_PERCELL = 2.2


def draw_synth(d, n_guides, kappa_range, seed):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    J = G - 1.5 * np.eye(d)
    Wd = rng.normal(size=(d, n_guides))
    Wd /= np.linalg.norm(Wd, axis=0, keepdims=True)
    kappa = rng.uniform(kappa_range[0], kappa_range[1], size=n_guides)
    U = -kappa[np.newaxis, :] * Wd
    return J, U


def simulate_S(J, U, sat_scale, noise, seed):
    S_lin = -np.linalg.solve(J, U)
    S_signal = S_lin if sat_scale is None or sat_scale > 1e6 else sat_scale * np.tanh(S_lin / sat_scale)
    rng = np.random.default_rng(seed)
    return S_signal + noise * rng.normal(size=S_signal.shape)


def held_out_rho(S, U):
    d, m = S.shape
    rng = np.random.default_rng(0)
    fold_ids = rng.integers(0, 5, size=m)
    r2, t2 = 0.0, 0.0
    for k in range(5):
        tr = fold_ids != k; te = fold_ids == k
        if tr.sum() < d or te.sum() == 0: continue
        S_pinv, *_ = regularized_pseudoinverse(S[:, tr], method="tsvd", parameter="path", rank_tol=RANK_TOL)
        A = -U[:, tr] @ S_pinv
        res = A @ S[:, te] + U[:, te]
        r2 += float(np.linalg.norm(res))**2
        t2 += float(np.linalg.norm(U[:, te]))**2
    return float(np.sqrt(r2)/max(np.sqrt(t2), 1e-12))


def rho_dist(n_guides, kappa_range, sat_scale, noise, n_reps=15, seed_base=0):
    rhos = []
    for rep in range(n_reps):
        J, U = draw_synth(D, n_guides, kappa_range, seed=seed_base + rep)
        S = simulate_S(J, U, sat_scale, noise, seed=seed_base + rep + 10000)
        rhos.append(held_out_rho(S, U))
    return float(np.mean(rhos)), float(np.std(rhos))


# Panel (a): 2D power surface at sat=0.5, σ=0.266
print("Computing 2D power surface...")
surface = []
for n in [50, 100, 200, 500, 1000]:
    for lo, hi, lab in [(0.05, 0.50, "narrow"), (0.05, 0.75, "medium"), (0.05, 1.00, "wide")]:
        lin_m, lin_s = rho_dist(n, (lo, hi), None, NOISE_PER_ENTRY_MEASURED, n_reps=15, seed_base=100)
        non_m, non_s = rho_dist(n, (lo, hi), 0.5, NOISE_PER_ENTRY_MEASURED, n_reps=15, seed_base=100)
        gap = non_m - lin_m
        detectable = gap > 1.96 * np.sqrt(lin_s**2 + non_s**2)
        surface.append({"n_guides": n, "kappa_label": lab, "gap": gap, "detectable": detectable})
        print(f"  n={n:>5d} κ={lab:>6s}: gap={gap:+.3f} detectable={detectable}")

# Panel (b): noise sweep
print("\nComputing noise sweep...")
SIGMAS = [0.005, 0.010, 0.020, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.266, 0.30]
configs = [
    ("REPLOGLE (n=200, κ narrow)", 200, (0.05, 0.50)),
    ("JOST-extended (n=200, κ wide)", 200, (0.05, 1.00)),
    ("ASPIRATIONAL (n=500, κ wide)", 500, (0.05, 1.00)),
]
noise_results = {}
for label, n, kr in configs:
    cfg = []
    for sigma in SIGMAS:
        lin_m, lin_s = rho_dist(n, kr, None, sigma)
        non_m, non_s = rho_dist(n, kr, 0.5, sigma)
        gap = non_m - lin_m
        cfg.append({"sigma": sigma, "gap": gap,
                     "detectable": gap > 1.96 * np.sqrt(lin_s**2 + non_s**2)})
    noise_results[label] = cfg
    print(f"  {label}: min gap {min(c['gap'] for c in cfg):+.3f}, max gap {max(c['gap'] for c in cfg):+.3f}")

(OUT_DIR / "rejection_power.json").write_text(json.dumps(
    {"power_surface": surface, "noise_sweep": noise_results}, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

ax = axes[0]
ns = sorted(set(s["n_guides"] for s in surface))
ks = ["narrow", "medium", "wide"]
gap_mat = np.zeros((len(ks), len(ns)))
detect_mat = np.zeros((len(ks), len(ns)), dtype=bool)
for s in surface:
    i = ks.index(s["kappa_label"]); j = ns.index(s["n_guides"])
    gap_mat[i, j] = s["gap"]; detect_mat[i, j] = s["detectable"]
im = ax.imshow(gap_mat, aspect="auto", cmap="RdYlBu_r", vmin=-0.05, vmax=0.05)
for i in range(len(ks)):
    for j in range(len(ns)):
        marker = "★" if detect_mat[i, j] else ""
        ax.text(j, i, f"{gap_mat[i,j]:+.3f}\n{marker}", ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns)
ax.set_yticks(range(len(ks))); ax.set_yticklabels(["narrow\n[0.05, 0.50]", "medium\n[0.05, 0.75]", "wide\n[0.05, 1.00]"])
ax.set_xlabel("n_guides"); ax.set_ylabel("κ range")
ax.set_title("(a) Rejection power for sat=0.5 tanh nonlinearity\n(gap = ρ_nonlin − ρ_linear, measured σ = 0.266)\n★ = 95%-CI detectable")
fig.colorbar(im, ax=ax, shrink=0.85)

ax = axes[1]
for (label, color, marker) in [
    ("REPLOGLE (n=200, κ narrow)", "#c65a30", "o"),
    ("JOST-extended (n=200, κ wide)", "#1f4e79", "s"),
    ("ASPIRATIONAL (n=500, κ wide)", "#2b6a3f", "^"),
]:
    entries = noise_results[label]
    sig = [e["sigma"] for e in entries]
    gap = [e["gap"] for e in entries]
    detect = [e["detectable"] for e in entries]
    ax.plot(sig, gap, color=color, lw=2, marker=marker, ms=7, label=label)
    for s, g, d in zip(sig, gap, detect):
        if d:
            ax.scatter(s, g, s=200, facecolors="none", edgecolors=color, lw=2, zorder=5)
ax.axhline(0.028, color="gray", ls=":", lw=1, label="approx 95% CI (gap > 0.028)")
ax.axhline(0, color="0.7", lw=0.5)
ax.axvspan(0.20, 0.30, alpha=0.18, color="#c65a30", label="measured Replogle σ")
ax.set_xscale("log")
ax.set_xlabel(r"per-entry noise σ  (secondary axis: required cells/guide)")
ax.set_ylabel(r"ρ gap: $\rho_{\rm nonlinear} - \rho_{\rm linear}$")
ax.set_title("(b) Noise sweep: detection power for sat=0.5 nonlinearity\ncircled points = 95%-CI detectable")
ax.legend(loc="upper right", fontsize=8)
ax2 = ax.twiny()
sig_ticks = [0.005, 0.01, 0.025, 0.05, 0.1, 0.266]
cells_ticks = [(SIGMA_PERCELL/s)**2 for s in sig_ticks]
ax2.set_xscale("log"); ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(sig_ticks)
ax2.set_xticklabels([f"{int(c):,}" for c in cells_ticks], fontsize=8)
ax2.set_xlabel(r"required cells per guide  (σ$_{\rm percell}$ ≈ 2.2)")
ax.set_ylim(-0.015, 0.055)

fig.suptitle("Fig S12: Rejection power for tanh-saturating nonlinearity (sat=0.5)", fontsize=11.5, y=1.03)
out = OUT_DIR / "figS12_rejection_power_surface.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
