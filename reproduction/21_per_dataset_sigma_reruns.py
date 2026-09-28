"""Reruns operator recovery + empirical null at each dataset's own bootstrapped σ.

Replaces the manuscript's earlier σ = 0.266 anchor with the per-dataset σ values
established in Fig. S19 (§4.4): K562 σ = 0.240, RPE1 σ = 0.352. Provides the
actual numbers cited in §2.2 Table 1 and §2.4 per-direction discussion.

Outputs (JSON):
  - per_dataset_recovery.json: cos_full, cos_1, cos_5, magnitude ratio,
    per cell line at its own σ, N=200 replicates, dense J_true.
  - per_dataset_per_direction.json: cos_k for k ∈ {1, 2, 3, 5, 10, 15, 20, 30}
    per cell line at its own σ, 4 ground-truth ensembles, N=15 replicates.

Runtime ~5 min.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, json
import numpy as np
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"

# Per-dataset σ from Fig. S19 (Methods §4.4)
DATASET_SIGMA = {"K562_essential": 0.240, "RPE1_essential": 0.352}
BUNDLE = {"K562_essential": "k562_essential_measurement.pkl",
           "RPE1_essential": "rpe1_essential_measurement.pkl"}
RANK_TOL = 1e-2
STRUCTURES = ["dense", "sparse_10pct", "sparse_2pct", "low_rank_5"]
KS = [1, 2, 3, 5, 10, 15, 20, 30]
SEED_BASE = 20260810
N_REPS_EMPNULL = 200
N_REPS_PERDIR = 15


def draw_J(d, seed, structure):
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
    raise ValueError


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


# =============================================================================
# Part 1: N=200 empirical null on dense J_true at per-dataset σ
# =============================================================================
recovery = {}
for cell_line, filename in BUNDLE.items():
    with (RESULTS / filename).open("rb") as f:
        state = pickle.load(f)
    U_real = state["measurement"].U
    d, n_guides = state["measurement"].S.shape
    sigma = DATASET_SIGMA[cell_line]
    print(f"\n{'='*70}\n{cell_line} @ σ={sigma}, dense J_true, N={N_REPS_EMPNULL}\n{'='*70}")
    Js, S_trues, A_fits = [], [], []
    rng_n = np.random.default_rng(SEED_BASE + 999)
    for r in range(N_REPS_EMPNULL):
        J = draw_J(d, seed=SEED_BASE + r, structure="dense")
        S_true = -np.linalg.solve(J, U_real)
        S_obs = S_true + sigma * rng_n.normal(size=S_true.shape)
        Js.append(J); S_trues.append(S_true); A_fits.append(fit_operator(S_obs, U_real))
        if (r + 1) % 100 == 0:
            print(f"  fit {r+1}/{N_REPS_EMPNULL}")
    real_cos = np.array([cos_full(A_fits[r], Js[r]) for r in range(N_REPS_EMPNULL)])
    real_cos1 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 1) for r in range(N_REPS_EMPNULL)])
    real_cos5 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 5) for r in range(N_REPS_EMPNULL)])
    magr = np.array([np.linalg.norm(A_fits[r]) / np.linalg.norm(Js[r]) for r in range(N_REPS_EMPNULL)])
    frob = np.array([np.linalg.norm(A_fits[r] - Js[r]) / np.linalg.norm(Js[r]) for r in range(N_REPS_EMPNULL)])
    # Cross-rep null: full expansion (10 shifts) for distribution stats + paired
    # (shift = 1 only) arrays for the paired-bootstrap inference below.
    null_cross_cos, null_cross_cos1, null_cross_cos5 = [], [], []
    paired_cross_cos, paired_cross_cos1, paired_cross_cos5 = [], [], []
    for shift in range(1, 11):
        for r in range(N_REPS_EMPNULL):
            rp = (r + shift) % N_REPS_EMPNULL
            c  = cos_full(A_fits[r], Js[rp])
            c1 = cos_topk(A_fits[r], Js[rp], S_trues[rp], 1)
            c5 = cos_topk(A_fits[r], Js[rp], S_trues[rp], 5)
            null_cross_cos.append(c)
            null_cross_cos1.append(c1)
            null_cross_cos5.append(c5)
            if shift == 1:
                paired_cross_cos.append(c)
                paired_cross_cos1.append(c1)
                paired_cross_cos5.append(c5)
    null_shuf_cos, null_shuf_cos1, null_shuf_cos5 = [], [], []
    rng_su = np.random.default_rng(SEED_BASE + 888)
    for r in range(N_REPS_EMPNULL):
        perm = rng_su.permutation(n_guides)
        S_ts = -np.linalg.solve(Js[r], U_real[:, perm])
        S_obs = S_ts + sigma * rng_su.normal(size=S_ts.shape)
        A_sh = fit_operator(S_obs, U_real)
        null_shuf_cos.append(cos_full(A_sh, Js[r]))
        null_shuf_cos1.append(cos_topk(A_sh, Js[r], S_trues[r], 1))
        null_shuf_cos5.append(cos_topk(A_sh, Js[r], S_trues[r], 5))
    paired_cross_cos  = np.asarray(paired_cross_cos)
    paired_cross_cos1 = np.asarray(paired_cross_cos1)
    paired_cross_cos5 = np.asarray(paired_cross_cos5)
    paired_shuf_cos   = np.asarray(null_shuf_cos)
    paired_shuf_cos1  = np.asarray(null_shuf_cos1)
    paired_shuf_cos5  = np.asarray(null_shuf_cos5)
    def _paired_bootstrap(real_arr, null_paired_arr, n_boot=10_000, seed=42):
        """Paired-difference bootstrap on shared simulated draws.

        real_arr and null_paired_arr are aligned length-N vectors: entry r
        pairs the fitted operator A_r's cosine against the true J_r (real)
        and its cosine against a paired null (either J_{(r+1) mod N} for the
        cross-replicate null, or J_r under a shuffled-U refit).

        Returns mean paired difference, 95 % CI, one-sided p (paired diff
        larger than 0), two-sided p, and Cohen's d_z (paired effect size).
        """
        n = len(real_arr)
        diff = np.asarray(real_arr) - np.asarray(null_paired_arr)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = diff[idx].mean(axis=1)
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        mean_d = float(diff.mean())
        # Two-sided empirical p from bootstrap; guard against 0.
        p_greater = float(np.mean(boot_means <= 0.0))
        p_less    = float(np.mean(boot_means >= 0.0))
        p_two = 2.0 * min(p_greater, p_less)
        p_two = max(p_two, 1.0 / n_boot)  # floor at simulation resolution
        d_z = mean_d / max(float(np.std(diff, ddof=1)), 1e-30)
        return {
            "mean_paired_diff": mean_d,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "p_two_sided": p_two,
            "p_one_sided_diff_gt_0": 1.0 - p_greater,
            "d_z": float(d_z),
            "n_pairs": int(n),
            "n_bootstrap": int(n_boot),
        }

    def summ(real, nc, ns, real_arr=None, cross_paired=None, shuf_paired=None):
        out = {
            "real_mean": float(real.mean()), "real_std": float(real.std()),
            "real_SE": float(real.std() / np.sqrt(len(real))),
            "cross_rep_null_mean": float(np.mean(nc)), "cross_rep_null_std": float(np.std(nc)),
            "shuf_U_null_mean": float(np.mean(ns)), "shuf_U_null_std": float(np.std(ns)),
            "z_per_rep_cross_rep": float((real.mean() - np.mean(nc)) / max(np.std(nc), 1e-9)),
            "z_per_rep_shuf_U": float((real.mean() - np.mean(ns)) / max(np.std(ns), 1e-9)),
        }
        # Paired-bootstrap inference on shared simulated draws.
        # Cross-replicate pairing uses shift=1 (primary paired null; the
        # 10-shift expansion in nc is retained for distribution plots only).
        if real_arr is not None and cross_paired is not None:
            out["paired_bootstrap_cross_rep"] = _paired_bootstrap(real_arr, cross_paired)
        if real_arr is not None and shuf_paired is not None:
            out["paired_bootstrap_shuf_U"] = _paired_bootstrap(real_arr, shuf_paired)
        return out
    row = {
        "sigma": sigma, "n_reps": N_REPS_EMPNULL, "n_guides": int(n_guides),
        "frob_rel_err_mean": float(frob.mean()), "frob_rel_err_std": float(frob.std()),
        "magnitude_ratio_mean": float(magr.mean()), "magnitude_ratio_std": float(magr.std()),
        "cos_full": summ(real_cos,  null_cross_cos,  null_shuf_cos,
                         real_arr=real_cos,  cross_paired=paired_cross_cos,  shuf_paired=paired_shuf_cos),
        "cos_1":    summ(real_cos1, null_cross_cos1, null_shuf_cos1,
                         real_arr=real_cos1, cross_paired=paired_cross_cos1, shuf_paired=paired_shuf_cos1),
        "cos_5":    summ(real_cos5, null_cross_cos5, null_shuf_cos5,
                         real_arr=real_cos5, cross_paired=paired_cross_cos5, shuf_paired=paired_shuf_cos5),
    }
    def _fmt_pb(pb):
        p_str = f"p<{1/pb['n_bootstrap']:g}" if pb["p_two_sided"] <= 1/pb["n_bootstrap"] else f"p={pb['p_two_sided']:.4f}"
        return (f"paired-diff={pb['mean_paired_diff']:+.4f} "
                f"[95% CI {pb['ci95_lo']:+.4f},{pb['ci95_hi']:+.4f}] "
                f"d_z={pb['d_z']:+.3f} {p_str}")
    for name in ("cos_full", "cos_1", "cos_5"):
        s = row[name]
        print(f"  {name:>8s} real={s['real_mean']:+.4f} ± {s['real_std']:.3f}   "
              f"cross-rep={s['cross_rep_null_mean']:+.4f}   shuf-U={s['shuf_U_null_mean']:+.4f}   "
              f"z_cross={s['z_per_rep_cross_rep']:+.2f}")
        print(f"           vs cross-rep: {_fmt_pb(s['paired_bootstrap_cross_rep'])}")
        print(f"           vs shuf-U:    {_fmt_pb(s['paired_bootstrap_shuf_U'])}")
    print(f"  ‖A-J‖/‖J‖ = {row['frob_rel_err_mean']:.4f}   ‖A‖/‖J‖ = {row['magnitude_ratio_mean']:.4f}")
    recovery[cell_line] = row

(OUT_DIR / "per_dataset_recovery.json").write_text(json.dumps(recovery, indent=2))

# =============================================================================
# Part 2: per-direction cos_k across 4 structures at per-dataset σ, N=15
# =============================================================================
perdir = {}
for cell_line, filename in BUNDLE.items():
    with (RESULTS / filename).open("rb") as f:
        state = pickle.load(f)
    U_real = state["measurement"].U
    d, n_guides = state["measurement"].S.shape
    sigma = DATASET_SIGMA[cell_line]
    print(f"\n{'-'*70}\n{cell_line} @ σ={sigma}, per-direction cos_k (N={N_REPS_PERDIR})\n{'-'*70}")
    line = {"sigma": sigma}
    for structure in STRUCTURES:
        per_k = {k: [] for k in KS}
        for rep in range(N_REPS_PERDIR):
            J = draw_J(d, seed=SEED_BASE + rep, structure=structure)
            S_true = -np.linalg.solve(J, U_real)
            rng = np.random.default_rng(rep + 10000)
            S_obs = S_true + sigma * rng.normal(size=S_true.shape)
            A = fit_operator(S_obs, U_real)
            for k in KS:
                per_k[k].append(cos_topk(A, J, S_true, k))
        line[structure] = {"cos_k_mean": {str(k): float(np.mean(per_k[k])) for k in KS},
                            "cos_k_std":  {str(k): float(np.std(per_k[k])) for k in KS}}
        top_k_str = ", ".join(f"k={k}:{line[structure]['cos_k_mean'][str(k)]:+.3f}"
                              for k in [1, 3, 5, 10, 30])
        print(f"  {structure:>16s}  {top_k_str}")
    perdir[cell_line] = line

(OUT_DIR / "per_dataset_per_direction.json").write_text(json.dumps(perdir, indent=2))

print("\n" + "="*70)
print("SUMMARY for manuscript §2.2 Table 1 and §2.4 per-direction:")
print("="*70)
for cell_line in BUNDLE:
    r = recovery[cell_line]
    p = perdir[cell_line]
    print(f"\n{cell_line} @ σ = {r['sigma']}:")
    print(f"  Table 1 dense: cos_full = {r['cos_full']['real_mean']:+.3f} ± {r['cos_full']['real_std']:.3f}, "
          f"‖A-J‖/‖J‖ = {r['frob_rel_err_mean']:.3f}, ‖A‖/‖J‖ = {r['magnitude_ratio_mean']:.4f}")
    print(f"  §2.4 per-direction cos_1 across structures:")
    for structure in STRUCTURES:
        c1 = p[structure]["cos_k_mean"]["1"]
        c5 = p[structure]["cos_k_mean"]["5"]
        c30 = p[structure]["cos_k_mean"]["30"]
        print(f"    {structure:>16s}  cos_1={c1:+.3f}  cos_5={c5:+.3f}  cos_30={c30:+.3f}")
