"""Fig S11: κ-range sweep at fixed real-scale (d=30, n=200 guides).

Under a random synthetic linear ground truth, sweeps κ distribution width from
Replogle-shape narrow [0.05, 0.50] to Jost-shape wide [0.05, 1.00]. Also sweeps
noise σ at fixed narrow and wide κ. Shows that wider κ improves the diagnostic's
discriminative dynamic range at every noise level, but narrow κ never reaches
the preregistered 0.25 threshold at any tested σ.

Reproduces Fig S11 in manuscript_figures/. No external data. Runtime ~2 min.
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


def draw_synthetic_setup(d, n_guides, kappa_range, seed=0, noise_sigma=0.025):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    J = G - 1.5 * np.eye(d)
    W_delta = rng.normal(size=(d, n_guides))
    W_delta /= np.linalg.norm(W_delta, axis=0, keepdims=True)
    lo, hi = kappa_range
    kappa = rng.uniform(lo, hi, size=n_guides)
    U = -kappa[np.newaxis, :] * W_delta
    S_true = -np.linalg.solve(J, U)
    S = S_true + noise_sigma * rng.normal(size=S_true.shape)
    return S, U, kappa


def _sym_rel_diff(A, B):
    nA = float(np.linalg.norm(A)); nB = float(np.linalg.norm(B))
    denom = 0.5 * (nA + nB)
    return float("nan") if denom <= 1e-12 else float(np.linalg.norm(A - B) / denom)


def bin_ap(S_bin, U_bin, rank_tol=RANK_TOL):
    S_pinv, _sel, _path, proj = regularized_pseudoinverse(S_bin, method="tsvd", parameter="path", rank_tol=rank_tol)
    return -U_bin @ S_pinv, proj


def linearity_test(S, U, kappa, rank_tol=RANK_TOL, n_null=200, seed=42):
    m = S.shape[1]
    median = float(np.median(kappa))
    weak = kappa <= median; strong = ~weak
    A_w, PX_w = bin_ap(S[:, weak], U[:, weak], rank_tol)
    A_s, PX_s = bin_ap(S[:, strong], U[:, strong], rank_tol)
    eig, evec = np.linalg.eigh(PX_w @ PX_s @ PX_w)
    or_rank = int((eig > 0.99).sum())
    if or_rank == 0: return float("inf"), 0, float("nan"), float("nan")
    V = evec[:, -or_rank:]; Pc = V @ V.T
    rel_diff = _sym_rel_diff(A_w @ Pc, A_s @ Pc)
    rng = np.random.default_rng(seed)
    null_vals = []
    for _ in range(n_null):
        perm = rng.permutation(m)
        mask = np.zeros(m, dtype=bool); mask[perm[:m//2]] = True
        A_a, PX_a = bin_ap(S[:, mask], U[:, mask], rank_tol)
        A_b, PX_b = bin_ap(S[:, ~mask], U[:, ~mask], rank_tol)
        eig, evec = np.linalg.eigh(PX_a @ PX_b @ PX_a)
        or_r = int((eig > 0.99).sum())
        if or_r == 0: continue
        V_ = evec[:, -or_r:]; Pc_ = V_ @ V_.T
        rd = _sym_rel_diff(A_a @ Pc_, A_b @ Pc_)
        if np.isfinite(rd): null_vals.append(rd)
    return rel_diff, or_rank, float(np.median(null_vals)), float(np.std(null_vals))


def held_out_rho(S, U, rank_tol=RANK_TOL, n_folds=5, seed=0):
    d, m = S.shape
    rng = np.random.default_rng(seed)
    fold_ids = rng.integers(0, n_folds, size=m)
    r2, t2 = 0.0, 0.0
    for k in range(n_folds):
        tr = fold_ids != k; te = fold_ids == k
        if tr.sum() < d or te.sum() == 0: continue
        S_pinv, *_ = regularized_pseudoinverse(S[:, tr], method="tsvd", parameter="path", rank_tol=rank_tol)
        A = -U[:, tr] @ S_pinv
        res = A @ S[:, te] + U[:, te]
        r2 += float(np.linalg.norm(res))**2; t2 += float(np.linalg.norm(U[:, te]))**2
    return float(np.sqrt(r2)/max(np.sqrt(t2), 1e-12))


scenarios = [
    ("narrow (Replogle)", (0.05, 0.50)),
    ("medium",            (0.05, 0.75)),
    ("wide (Jost)",       (0.05, 1.00)),
    ("very wide",         (0.01, 1.00)),
]

# Panel A: κ range sweep at n=200, σ=0.025
n200_results = []
for label, (lo, hi) in scenarios:
    S, U, kappa = draw_synthetic_setup(D, 200, (lo, hi), seed=20260810, noise_sigma=0.025)
    rd, or_r, nm, ns = linearity_test(S, U, kappa)
    rho = held_out_rho(S, U)
    n200_results.append({"label": label, "kappa_width": hi - lo, "rel_diff": rd,
                         "null_median": nm, "held_out_rho": rho})

# Panel B: noise sweep at fixed narrow and wide κ
noise_narrow, noise_wide = [], []
for noise in [0.005, 0.010, 0.025, 0.050, 0.100]:
    S, U, kappa = draw_synthetic_setup(D, 200, (0.05, 0.50), seed=20260810, noise_sigma=noise)
    rd, _, nm, _ = linearity_test(S, U, kappa)
    noise_narrow.append({"sigma": noise, "rel_diff": rd, "null_median": nm, "held_out_rho": held_out_rho(S, U)})
    S, U, kappa = draw_synthetic_setup(D, 200, (0.05, 1.00), seed=20260810, noise_sigma=noise)
    rd, _, nm, _ = linearity_test(S, U, kappa)
    noise_wide.append({"sigma": noise, "rel_diff": rd, "null_median": nm, "held_out_rho": held_out_rho(S, U)})

(OUT_DIR / "kappa_range_sweep.json").write_text(
    json.dumps({"n200": n200_results,
                 "noise_sweep_narrow_kappa_n200": noise_narrow,
                 "noise_sweep_wide_kappa_n200": noise_wide}, indent=2))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

ax = axes[0]
xs = np.arange(len(n200_results))
labels = [s["label"] for s in n200_results]
kappa_widths = [s["kappa_width"] for s in n200_results]
rd = [s["rel_diff"] for s in n200_results]
nulls = [s["null_median"] for s in n200_results]
rho = [s["held_out_rho"] for s in n200_results]
ax.bar(xs - 0.22, rd, width=0.20, color="#c65a30", label="rel_diff")
ax.bar(xs, nulls, width=0.20, color="#e8a288", label="null_median")
ax.bar(xs + 0.22, rho, width=0.20, color="#1f4e79", label="held-out ρ")
ax.axhline(0.25, color="red", ls=":", lw=1)
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("diagnostic value")
ax.set_title("(a) κ range sweep at n=200 guides, σ=0.025\n(much cleaner than measured Replogle σ=0.27)")
ax.legend(loc="upper right", fontsize=8); ax.set_ylim(0, 1.2)
for x, w in zip(xs, kappa_widths):
    ax.text(x, -0.06, f"κ span = {w:.2f}", ha="center", fontsize=7, color="0.4")

ax = axes[1]
sig_n = [e["sigma"] for e in noise_narrow]; rho_n = [e["held_out_rho"] for e in noise_narrow]
sig_w = [e["sigma"] for e in noise_wide]; rho_w = [e["held_out_rho"] for e in noise_wide]
ax.plot(sig_n, rho_n, "o-", color="#c65a30", lw=2, ms=7, label="narrow κ [0.05, 0.50] (Replogle-like)")
ax.plot(sig_w, rho_w, "s-", color="#1f4e79", lw=2, ms=7, label="wide κ [0.05, 1.00] (Jost-like)")
ax.axhline(0.25, color="red", ls=":", lw=1, label="preregistered 0.25 threshold")
ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6, label="zero-predictor baseline")
ax.axvspan(0.20, 0.30, alpha=0.18, color="#2b6a3f", label="measured Replogle σ")
ax.set_xscale("log")
ax.set_xlabel("per-entry noise σ")
ax.set_ylabel("held-out ρ")
ax.set_title("(b) held-out ρ vs σ, at n=200: wider κ shifts curve down\nbut narrow-κ never reaches preregistered threshold")
ax.legend(loc="lower right", fontsize=8); ax.set_ylim(0, 1.45)

fig.suptitle("Fig S11: κ-range sweep at fixed n=200 guides, d=30", fontsize=11.5, y=1.02)
out = OUT_DIR / "figS11_kappa_range_sweep.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
