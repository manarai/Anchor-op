"""Fig S10: real-scale linear positive control on the linearity diagnostics.

For each Replogle measurement bundle, extracts real U and κ, draws synthetic
linear ground truth with matched (d, n_guides), and simulates observed S under
a range of Gaussian noise levels. Reports rel_diff, null_median, and held-out
ρ vs σ, overlaying observed Replogle values as horizontal lines.

Establishes: at measured σ ≈ 0.266, synthetic linear reproduces Replogle
observations within 0.05 — Replogle is consistent with linear at its scale.

Requires the pickled measurement bundles from `../results/`. Runtime ~3 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import anchorop as ao
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)


def _sym_rel_diff(A, B):
    nA = float(np.linalg.norm(A)); nB = float(np.linalg.norm(B))
    denom = 0.5 * (nA + nB)
    return float("nan") if denom <= 1e-12 else float(np.linalg.norm(A - B) / denom)


def bin_ap(S_bin, U_bin, rank_tol):
    S_pinv, _sel, _path, proj = regularized_pseudoinverse(S_bin, method="tsvd", parameter="path", rank_tol=rank_tol)
    return -U_bin @ S_pinv, proj


def linearity_test(S, U, kappa, rank_tol, n_null=200, seed=42):
    m = S.shape[1]
    median = float(np.median(kappa))
    weak = kappa <= median; strong = ~weak
    A_w, PX_w = bin_ap(S[:, weak], U[:, weak], rank_tol)
    A_s, PX_s = bin_ap(S[:, strong], U[:, strong], rank_tol)
    eig, evec = np.linalg.eigh(PX_w @ PX_s @ PX_w)
    or_rank = int((eig > 0.99).sum())
    if or_rank == 0: return float("inf"), float("nan")
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
    return rel_diff, float(np.median(null_vals))


def held_out_rho(S, U, rank_tol, n_folds=5, seed=0):
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
        r2 += float(np.linalg.norm(res))**2
        t2 += float(np.linalg.norm(U[:, te]))**2
    return float(np.sqrt(r2)/max(np.sqrt(t2), 1e-12))


data = {}
for cell_line, filename in [("K562_essential", "k562_essential_measurement.pkl"),
                              ("RPE1_essential", "rpe1_essential_measurement.pkl")]:
    bundle_path = RESULTS / filename
    if not bundle_path.exists():
        raise SystemExit(f"missing {bundle_path} — run reproduction/03_fig3_k562_essential.py and 04_fig4_rpe1_essential.py first")
    with bundle_path.open("rb") as f:
        state = pickle.load(f)
    m = state["measurement"]; r = m.report
    U_real = m.U
    kappa_real = np.array([r.guide_efficiencies[g] for g in m.guide_names])
    rank_tol = r.rank_tol
    d, n_guides = m.S.shape

    real_lin = state.get("linearity")
    real_rho = held_out_rho(m.S, U_real, rank_tol)

    print(f"\n{cell_line}: d={d} n_guides={n_guides}")
    print(f"  REAL rel_diff={real_lin.relative_difference:.3f}, null_median={real_lin.null_median:.3f}, ρ={real_rho:.3f}")

    syn = []
    for sigma in [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.266, 0.30]:
        rng_j = np.random.default_rng(20260810 + hash(cell_line) % 1000)
        G = rng_j.normal(size=(d, d)) / np.sqrt(d)
        J_true = G - 1.5 * np.eye(d)
        S_true = -np.linalg.solve(J_true, U_real)
        rng_n = np.random.default_rng(0)
        S_syn = S_true + sigma * rng_n.normal(size=S_true.shape)
        rd, nm = linearity_test(S_syn, U_real, kappa_real, rank_tol)
        rho = held_out_rho(S_syn, U_real, rank_tol)
        syn.append({"sigma": sigma, "rel_diff": rd, "null_median": nm, "rho": rho})
        print(f"    σ={sigma:.4f}: rel_diff={rd:.3f}, null={nm:.3f}, ρ={rho:.3f}")

    data[cell_line] = {
        "real": {"rel_diff": float(real_lin.relative_difference),
                  "null_median": float(real_lin.null_median),
                  "rho": real_rho},
        "synthetic": syn,
    }

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
for ax, (cell_line, title) in zip(axes, [("K562_essential", "K562 essential"), ("RPE1_essential", "RPE1 essential")]):
    d = data[cell_line]
    syn = d["synthetic"]; real = d["real"]
    sigmas = [s["sigma"] for s in syn]
    ax.plot(sigmas, [s["rel_diff"] for s in syn], "o-", color="#c65a30", lw=2, ms=6,
            label="synthetic linear: rel_diff")
    ax.plot(sigmas, [s["null_median"] for s in syn], "s--", color="#e8a288", lw=1.5, ms=5,
            label="synthetic linear: null_median")
    ax.plot(sigmas, [s["rho"] for s in syn], "^-", color="#1f4e79", lw=2, ms=6,
            label="synthetic linear: held-out ρ")
    ax.axhline(real["rel_diff"], color="#c65a30", ls=":", lw=1.5, alpha=0.8,
               label=f"REAL rel_diff = {real['rel_diff']:.3f}")
    ax.axhline(real["null_median"], color="#e8a288", ls=":", lw=1.5, alpha=0.8,
               label=f"REAL null_median = {real['null_median']:.3f}")
    ax.axhline(real["rho"], color="#1f4e79", ls=":", lw=1.5, alpha=0.8,
               label=f"REAL ρ = {real['rho']:.3f}")
    ax.axvspan(0.20, 0.30, alpha=0.18, color="#2b6a3f", label="measured σ range (0.27 median)")
    ax.axhline(0.25, color="red", ls="-", lw=1, alpha=0.5, label="preregistered 0.25 threshold")
    ax.set_xscale("log")
    ax.set_xlabel(r"per-entry noise σ on synthetic $S$")
    ax.set_ylabel("diagnostic value")
    ax.set_title(f"{title}\n(d=30, real U, real κ)")
    ax.legend(loc="center right", fontsize=7.5)
    ax.set_ylim(-0.05, 1.75)

fig.suptitle("Fig S10: Real-scale positive control (synthetic linear at Replogle-matched (d, U, κ), noise sweep)",
             fontsize=11.5, y=1.02)
out = OUT_DIR / "figS10_realscale_positive_control.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")
