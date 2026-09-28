"""All-ensemble empirical null (N=200) at per-dataset σ + paired-bootstrap inference.

Extends `21_per_dataset_sigma_reruns.py` (which runs the dense case only) to all
four ground-truth interaction ensembles: dense, sparse_10pct, sparse_2pct,
low_rank_5, each stacked on the common -1.5·I diagonal shift. For each
(cell line × ensemble) pair it produces:

  - N=200 real cos(A_r, J_r), cos_1, cos_5 arrays
  - Cross-replicate null via 10 shifts (2000 null samples per metric) plus
    a shift-1 paired null (N=200) for paired-bootstrap inference
  - Shuffled-U null (N=200) with paired-bootstrap inference

Writes `per_dataset_recovery_all_ensembles.json`. Feeds the four bottom-row
panels of Fig. 2 at the current dataset-specific σ anchors (K562 σ=0.240,
RPE1 σ=0.352), replacing the previous state where only the dense-interaction
panel was sourced from current-anchor data.

Runtime ~15 min (4 ensembles × 2 cell lines × N=200 fits + paired bootstrap).
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, json
import numpy as np
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"

DATASET_SIGMA = {"K562_essential": 0.240, "RPE1_essential": 0.352}
BUNDLE = {"K562_essential": "k562_essential_measurement.pkl",
          "RPE1_essential": "rpe1_essential_measurement.pkl"}
RANK_TOL = 1e-2
STRUCTURES = ["dense", "sparse_10pct", "sparse_2pct", "low_rank_5"]
SEED_BASE = 20260810
N_REPS = 200


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
    raise ValueError(structure)


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


def paired_bootstrap(real_arr, null_paired_arr, n_boot=10_000, seed=42):
    n = len(real_arr)
    diff = np.asarray(real_arr) - np.asarray(null_paired_arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    mean_d = float(diff.mean())
    p_greater = float(np.mean(boot_means <= 0.0))
    p_less = float(np.mean(boot_means >= 0.0))
    p_two = 2.0 * min(p_greater, p_less)
    p_two = max(p_two, 1.0 / n_boot)
    d_z = mean_d / max(float(np.std(diff, ddof=1)), 1e-30)
    return {
        "mean_paired_diff": mean_d, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "p_two_sided": p_two, "p_one_sided_diff_gt_0": 1.0 - p_greater,
        "d_z": float(d_z), "n_pairs": int(n), "n_bootstrap": int(n_boot),
    }


def run_one(cell_line, structure, U_real, n_guides, d, sigma):
    Js, S_trues, A_fits = [], [], []
    rng_n = np.random.default_rng(SEED_BASE + 999)
    for r in range(N_REPS):
        J = draw_J(d, seed=SEED_BASE + r, structure=structure)
        S_true = -np.linalg.solve(J, U_real)
        S_obs = S_true + sigma * rng_n.normal(size=S_true.shape)
        Js.append(J); S_trues.append(S_true); A_fits.append(fit_operator(S_obs, U_real))
    real_cos  = np.array([cos_full(A_fits[r], Js[r]) for r in range(N_REPS)])
    real_cos1 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 1) for r in range(N_REPS)])
    real_cos5 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 5) for r in range(N_REPS)])
    magr = np.array([np.linalg.norm(A_fits[r]) / np.linalg.norm(Js[r]) for r in range(N_REPS)])
    frob = np.array([np.linalg.norm(A_fits[r] - Js[r]) / np.linalg.norm(Js[r]) for r in range(N_REPS)])
    # Cross-rep null: 10 shifts for distribution, shift=1 for paired.
    null_cross_cos, null_cross_cos1, null_cross_cos5 = [], [], []
    paired_cross_cos, paired_cross_cos1, paired_cross_cos5 = [], [], []
    for shift in range(1, 11):
        for r in range(N_REPS):
            rp = (r + shift) % N_REPS
            c  = cos_full(A_fits[r], Js[rp])
            c1 = cos_topk(A_fits[r], Js[rp], S_trues[rp], 1)
            c5 = cos_topk(A_fits[r], Js[rp], S_trues[rp], 5)
            null_cross_cos.append(c);  null_cross_cos1.append(c1);  null_cross_cos5.append(c5)
            if shift == 1:
                paired_cross_cos.append(c);  paired_cross_cos1.append(c1);  paired_cross_cos5.append(c5)
    # Shuffled-U null (paired by r).
    null_shuf_cos, null_shuf_cos1, null_shuf_cos5 = [], [], []
    rng_su = np.random.default_rng(SEED_BASE + 888)
    for r in range(N_REPS):
        perm = rng_su.permutation(n_guides)
        S_ts = -np.linalg.solve(Js[r], U_real[:, perm])
        S_obs = S_ts + sigma * rng_su.normal(size=S_ts.shape)
        A_sh = fit_operator(S_obs, U_real)
        null_shuf_cos.append(cos_full(A_sh, Js[r]))
        null_shuf_cos1.append(cos_topk(A_sh, Js[r], S_trues[r], 1))
        null_shuf_cos5.append(cos_topk(A_sh, Js[r], S_trues[r], 5))
    paired_cross = {"cos_full": np.asarray(paired_cross_cos),
                    "cos_1":    np.asarray(paired_cross_cos1),
                    "cos_5":    np.asarray(paired_cross_cos5)}
    paired_shuf  = {"cos_full": np.asarray(null_shuf_cos),
                    "cos_1":    np.asarray(null_shuf_cos1),
                    "cos_5":    np.asarray(null_shuf_cos5)}
    reals = {"cos_full": real_cos, "cos_1": real_cos1, "cos_5": real_cos5}
    nc = {"cos_full": null_cross_cos, "cos_1": null_cross_cos1, "cos_5": null_cross_cos5}
    ns = {"cos_full": null_shuf_cos, "cos_1": null_shuf_cos1, "cos_5": null_shuf_cos5}

    def summ(name):
        r = reals[name]; c = nc[name]; s = ns[name]
        return {
            "real_mean": float(r.mean()), "real_std": float(r.std()),
            "real_SE":   float(r.std() / np.sqrt(len(r))),
            "cross_rep_null_mean": float(np.mean(c)), "cross_rep_null_std": float(np.std(c)),
            "shuf_U_null_mean":    float(np.mean(s)), "shuf_U_null_std":    float(np.std(s)),
            "z_per_rep_cross_rep": float((r.mean() - np.mean(c)) / max(np.std(c), 1e-9)),
            "z_per_rep_shuf_U":    float((r.mean() - np.mean(s)) / max(np.std(s), 1e-9)),
            "paired_bootstrap_cross_rep": paired_bootstrap(r, paired_cross[name]),
            "paired_bootstrap_shuf_U":    paired_bootstrap(r, paired_shuf[name]),
        }

    return {
        "sigma": sigma, "n_reps": N_REPS, "n_guides": int(n_guides),
        "structure": structure,
        "frob_rel_err_mean": float(frob.mean()), "frob_rel_err_std": float(frob.std()),
        "magnitude_ratio_mean": float(magr.mean()), "magnitude_ratio_std": float(magr.std()),
        "cos_full": summ("cos_full"),
        "cos_1":    summ("cos_1"),
        "cos_5":    summ("cos_5"),
    }


def _fmt(pb):
    n_boot = pb["n_bootstrap"]
    p = pb["p_two_sided"]
    p_str = f"p<{1/n_boot:g}" if p <= 1/n_boot else f"p={p:.4f}"
    return (f"paired-diff={pb['mean_paired_diff']:+.4f} "
            f"[95% CI {pb['ci95_lo']:+.4f},{pb['ci95_hi']:+.4f}] d_z={pb['d_z']:+.3f} {p_str}")


results = {}
for cell_line, filename in BUNDLE.items():
    with (RESULTS / filename).open("rb") as f:
        state = pickle.load(f)
    U_real = state["measurement"].U
    d, n_guides = state["measurement"].S.shape
    sigma = DATASET_SIGMA[cell_line]
    print(f"\n{'='*72}\n{cell_line} @ σ={sigma}, d={d}, n_guides={n_guides}, N={N_REPS}\n{'='*72}")
    per_line = {}
    for structure in STRUCTURES:
        print(f"\n  {structure} ...")
        row = run_one(cell_line, structure, U_real, n_guides, d, sigma)
        per_line[structure] = row
        s = row["cos_full"]
        print(f"    cos_full  real={s['real_mean']:+.4f}   cross-rep={s['cross_rep_null_mean']:+.4f}   "
              f"shuf-U={s['shuf_U_null_mean']:+.4f}")
        print(f"      cross-rep {_fmt(s['paired_bootstrap_cross_rep'])}")
        print(f"      shuf-U    {_fmt(s['paired_bootstrap_shuf_U'])}")
        s1 = row["cos_1"]
        print(f"    cos_1     real={s1['real_mean']:+.4f}   cross-rep={s1['cross_rep_null_mean']:+.4f}   "
              f"shuf-U={s1['shuf_U_null_mean']:+.4f}")
        print(f"      cross-rep {_fmt(s1['paired_bootstrap_cross_rep'])}")
    results[cell_line] = per_line

(OUT_DIR / "per_dataset_recovery_all_ensembles.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT_DIR/'per_dataset_recovery_all_ensembles.json'}")

print("\n" + "="*72)
print("SUMMARY — full-operator paired-bootstrap cross-rep null (headline test)")
print("="*72)
print(f"{'cell_line':>18s} {'structure':>14s}  {'paired-diff':>12s} {'95% CI':>22s} {'d_z':>7s} {'p':>10s}")
for cl in results:
    for st in STRUCTURES:
        pb = results[cl][st]["cos_full"]["paired_bootstrap_cross_rep"]
        p = pb["p_two_sided"]
        p_str = f"<{1/pb['n_bootstrap']:g}" if p <= 1/pb["n_bootstrap"] else f"{p:.4f}"
        print(f"{cl:>18s} {st:>14s}  {pb['mean_paired_diff']:>+12.4f}  "
              f"[{pb['ci95_lo']:+.4f},{pb['ci95_hi']:+.4f}]  {pb['d_z']:>+7.3f}  {p_str:>10s}")
