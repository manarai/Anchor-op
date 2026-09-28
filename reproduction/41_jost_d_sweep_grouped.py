"""Jost d-sweep under target-grouped folds.

Reviewer's design-vs-encoding argument: Jost has 25 targets and d = 30.
With 5 target-grouped folds, each training set spans ~20 of 30 program
directions, so held-out target directions mostly fall outside the
identified subspace. That's an underdetermined-problem property, not
an encoding failure statement.

Fix: truncate the coordinate system to the first d' rows of S and U
for d' ∈ {5, 10, 15, 20, 25, 30} (equivalently, project onto the top
d' PCs of the control basis, which are the first d' rows of z = Wᵀe).
At d' ≤ 15, 20 training targets exceed d', so Jost becomes
overdetermined and the encoding is tested fairly. Reports nested-CV
regularized ρ on real Jost and on a matched-SNR linear-truth
simulation at each d'.

Runs off ``results/jost_measurement.pkl``. Minutes.
Output: results/recheck/F_jost_d_sweep_grouped.json.
"""
from __future__ import annotations
import json, pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260927
N_OUTER = 5
N_INNER = 3
N_REPS_LIN_SIM = 15
D_GRID = [5, 10, 15, 20, 25, 30]
SIGMA = 0.06555229270524932    # bootstrapped in Jost PCA basis (F3-Jost)


def _fit_A_at_rank(S, U, r):
    r = int(r)
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]), dtype=float)
    Us, sv, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(sv))
    sv_inv = np.zeros_like(sv); sv_inv[:r] = 1.0 / sv[:r]
    S_pinv = (Vt.T * sv_inv) @ Us.T
    return -U @ S_pinv


def _mse(A, S_te, U_te):
    r = A @ S_te + U_te
    return float(np.sum(r ** 2))


def _rank_grid(d):
    # rank 0 = A ≡ 0 (predict-zero baseline; ρ = 1 exactly).
    return [r for r in [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30] if r <= d]


def _kfold(idx, k, seed):
    rng = np.random.default_rng(seed)
    return list(np.array_split(rng.permutation(idx), k))


def _target_folds(target_of_g, k, seed):
    rng = np.random.default_rng(seed)
    targets = sorted(set(target_of_g.tolist()))
    rng.shuffle(targets)
    tgroups = np.array_split(np.array(targets), k)
    folds = []
    for i in range(k):
        te_t = set(tgroups[i].tolist())
        te = np.array([t in te_t for t in target_of_g])
        folds.append((np.where(~te)[0], np.where(te)[0]))
    return folds


def _inner_pick(S_tr, U_tr, target_of_tr, d, seed):
    """Target-grouped inner CV (no sibling leakage). Return picked rank."""
    unique_targets = sorted(set(target_of_tr.tolist()))
    k = min(N_INNER, len(unique_targets))
    if k < 2:
        return 0
    rng = np.random.default_rng(seed)
    tgt_perm = np.array(unique_targets, dtype=object); rng.shuffle(tgt_perm)
    tgroups = np.array_split(tgt_perm, k)
    inner = []
    for i in range(k):
        te_t = set(tgroups[i].tolist())
        te_mask = np.array([t in te_t for t in target_of_tr])
        inner.append((np.where(~te_mask)[0], np.where(te_mask)[0]))
    best_r, best = None, np.inf
    for r in _rank_grid(d):
        pooled = 0.0; ok = True
        for tr, val in inner:
            if r > 0 and len(tr) <= r:
                ok = False; break
            A = _fit_A_at_rank(S_tr[:, tr], U_tr[:, tr], r)
            pooled += _mse(A, S_tr[:, val], U_tr[:, val])
        if ok and pooled < best:
            best, best_r = pooled, r
    return best_r if best_r is not None else 0


def _nested_rho(S, U, target_of_g, d, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, picks = 0.0, 0.0, []
    for i, (tr, te) in enumerate(outer):
        r = _inner_pick(S[:, tr], U[:, tr], target_of_g[tr], d, seed + i * 17)
        if r > 0 and len(tr) <= r:
            r = 0
        A = _fit_A_at_rank(S[:, tr], U[:, tr], r)
        resid = A @ S[:, te] + U[:, te]
        num += float(np.sum(resid ** 2))
        den += float(np.sum(U[:, te] ** 2))
        picks.append(int(r))
    return {"rho": float(np.sqrt(num / max(den, 1e-30))),
            "picked_rank_median": int(np.median(picks)) if picks else None,
            "picked_rank_min": int(min(picks)) if picks else None,
            "picked_rank_max": int(max(picks)) if picks else None}


def main():
    with (RESULTS / "jost_measurement.pkl").open("rb") as f:
        b = pickle.load(f)
    meas = b["measurement"]
    S = meas.S; U = meas.U
    guides = list(meas.report.retained_guides)
    stored = getattr(meas.report, "guide_targets", None)
    if stored:
        stored = {str(k): str(v) for k, v in dict(stored).items()}
        target_of_g = np.array([stored.get(str(g), str(g)) for g in guides])
    else:
        target_of_g = np.array([str(g).split("_")[0] for g in guides])
    d0, m = U.shape
    n_targets = len(set(target_of_g.tolist()))

    payload = {"seed": SEED, "n_outer": N_OUTER, "n_inner": N_INNER,
               "n_reps_lin_sim": N_REPS_LIN_SIM, "d_grid": D_GRID,
               "sigma_per_sgRNA_jost": SIGMA,
               "n_sgRNA": m, "n_targets": n_targets, "d_full": d0,
               "note": ("d' truncation takes the first d' rows of S and U — "
                        "equivalently, projects into the top d' PCs of the "
                        "control basis. At each d', we compute alpha_hat_S "
                        "from the median column norm of the truncated real S "
                        "vs a fresh alpha=1 sim on the truncated U, so the "
                        "matched linear-truth control tracks the same SNR "
                        "at every d'."),
               "results": {}}
    for d in D_GRID:
        Sp, Up = S[:d, :], U[:d, :]
        real = _nested_rho(Sp, Up, target_of_g, d, SEED)
        # α̂ at this d
        rng0 = np.random.default_rng(SEED)
        J_ref0 = rng0.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
        S_ref = -np.linalg.solve(J_ref0, Up)
        alpha_hat = float(np.median(np.linalg.norm(Sp, axis=0))
                          / max(np.median(np.linalg.norm(S_ref, axis=0)), 1e-30))

        rhos = []; picks = []
        for r in range(N_REPS_LIN_SIM):
            rng = np.random.default_rng(SEED + r + 9000)
            J_ref = rng.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
            J = J_ref / max(alpha_hat, 1e-30)
            S_true = -np.linalg.solve(J, Up)
            S_obs = S_true + SIGMA * rng.normal(size=S_true.shape)
            out = _nested_rho(S_obs, Up, target_of_g, d, SEED + r + 9000)
            rhos.append(out["rho"]); picks.append(out["picked_rank_median"])
        lin = {"n_reps": N_REPS_LIN_SIM, "alpha_S_hat": alpha_hat,
               "rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
               "picked_rank_median_across_reps": int(np.median(picks)),
               "regime": ("overdetermined" if n_targets - (n_targets // N_OUTER) >= d
                          else "underdetermined")}
        payload["results"][f"d={d}"] = {"d": d, "real_nested_cv": real,
                                         "linear_truth_matched_nested_cv": lin,
                                         "alpha_S_hat_at_this_d": alpha_hat}
        print(f"d={d:>2}: real ρ={real['rho']:.4f} (rank med {real['picked_rank_median']}), "
              f"lin-truth ρ={lin['rho_mean']:.4f}±{lin['rho_std']:.4f} "
              f"(α̂={alpha_hat:.1f}; {lin['regime']})")

    (OUT / "F_jost_d_sweep_grouped.json").write_text(json.dumps(payload, indent=2))
    print("\nsaved:", OUT / "F_jost_d_sweep_grouped.json")


if __name__ == "__main__":
    main()
