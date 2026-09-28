"""Nested-CV regularized ρ on all three Perturb-seq screens.

Reviewer's point: ρ well above 1 mostly measures estimator variance
(A shrunk to zero gives ρ = 1 exactly), so it's an estimator property
as much as a biological one. Fix: choose the TSVD rank inside each
training fold by held-out MSE, then evaluate the picked model on the
outer test fold. Report the best achievable ρ. Do the same for the
matched-SNR linear-truth simulation so the comparison is like-for-like.

Outer folds: target-grouped 5-fold (no sibling-sgRNA leakage). Inner
CV: 3-fold guide split within the outer training set, TSVD rank r ∈
{1, 2, 3, 5, 8, 12, 16, 20, 25, 30}. Pick r that minimizes inner-fold
pooled MSE of A·S_val + U_val. Refit on the full outer training set
with the picked r and score the outer test.

Datasets:
  - K562 essential — one guide per target after Replogle aggregation.
    Target-grouped and guide-grouped folds are the same here; label
    the folds "target-grouped (equivalent to guide-grouped)".
  - RPE1 essential — same.
  - Jost 2020 — target-grouped folds mean all sgRNAs of a target held
    out together.

Matched-SNR linear truth:
  - K562: σ = 0.240, α_S = 369 (from F1/F2)
  - RPE1: σ = 0.352, α_S = 199
  - Jost:  σ = 0.066, α_S = 29.6

Runs off the existing pickles. Minutes.
Output: results/recheck/F_nested_cv_rho.json.
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
# rank 0 = A ≡ 0, giving ρ = 1 exactly. Including it lets nested-CV pick
# "shrink to zero" when that's genuinely the best model, and turns the
# > 1 excess into a diagnostic rather than a fit artifact.
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]

DATASETS = {
    "K562_essential": {
        "pkl": "k562_essential_measurement.pkl", "sigma": 0.240, "alpha_S": 369.0,
    },
    "RPE1_essential": {
        "pkl": "rpe1_essential_measurement.pkl", "sigma": 0.352, "alpha_S": 199.0,
    },
    "Jost_2020": {
        "pkl": "jost_measurement.pkl",          "sigma": 0.066, "alpha_S": 29.6,
    },
}


def _fit_A_at_rank(S, U, r):
    """TSVD fit at fixed rank r. r = 0 -> A ≡ 0 (predict-zero baseline)."""
    r = int(r)
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]), dtype=float)
    Us, sv, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(sv))
    sv_inv = np.zeros_like(sv); sv_inv[:r] = 1.0 / sv[:r]
    S_pinv = (Vt.T * sv_inv) @ Us.T
    return -U @ S_pinv


def _rho_pooled(A, S_test, U_test):
    resid = A @ S_test + U_test
    return float(np.sqrt(np.sum(resid ** 2) / max(np.sum(U_test ** 2), 1e-30)))


def _mse_pooled(A, S_test, U_test):
    resid = A @ S_test + U_test
    return float(np.sum(resid ** 2))


def _kfold_indices(idx_array, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx_array)
    return list(np.array_split(idx, k))


def _target_grouped_folds(target_of_g, k, seed):
    """All sgRNAs of a target in the same outer fold."""
    rng = np.random.default_rng(seed)
    targets = sorted(set(target_of_g.tolist()))
    rng.shuffle(targets)
    tgroups = np.array_split(np.array(targets), k)
    folds = []
    for i in range(k):
        test_targets = set(tgroups[i].tolist())
        test_mask = np.array([t in test_targets for t in target_of_g])
        folds.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return folds


def _inner_pick_rank(S_tr, U_tr, target_of_tr, seed):
    """Inner CV: target-grouped folds within the outer training set, pick
    TSVD rank minimizing pooled MSE. Grouped-inner is required to avoid
    sibling-sgRNA leakage on Jost."""
    # Try n_inner = 3 target-grouped folds; if only < 3 targets remain, use
    # as many as are available (still target-grouped).
    unique_targets = sorted(set(target_of_tr.tolist()))
    k = min(N_INNER, len(unique_targets))
    if k < 2:  # can't do CV with a single target; return rank 0 (predict-zero)
        return 0
    rng = np.random.default_rng(seed)
    tgt_perm = np.array(unique_targets, dtype=object); rng.shuffle(tgt_perm)
    tgroups = np.array_split(tgt_perm, k)
    inner = []
    for i in range(k):
        te_t = set(tgroups[i].tolist())
        te_mask = np.array([t in te_t for t in target_of_tr])
        inner.append((np.where(~te_mask)[0], np.where(te_mask)[0]))

    best_r, best_mse = None, np.inf
    for r in RANK_GRID:
        pooled_num = 0.0; ok = True
        for tr, val in inner:
            if r > 0 and len(tr) <= r:
                ok = False; break
            A = _fit_A_at_rank(S_tr[:, tr], U_tr[:, tr], r)
            pooled_num += _mse_pooled(A, S_tr[:, val], U_tr[:, val])
        if ok and pooled_num < best_mse:
            best_mse, best_r = pooled_num, r
    return best_r if best_r is not None else 0


def _nested_cv_rho(S, U, target_of_g, seed):
    """Return {rho_pooled, per_fold, picked_ranks}."""
    outer = _target_grouped_folds(target_of_g, N_OUTER, seed)
    num_sq = 0.0; den_sq = 0.0
    per_fold = []; picked = []
    for i, (tr, te) in enumerate(outer):
        if len(tr) < 3:
            continue
        r = _inner_pick_rank(S[:, tr], U[:, tr], target_of_g[tr], seed + i * 17)
        if r > 0 and len(tr) <= r:
            r = 0  # fall back to predict-zero rather than an ill-defined fit
        A = _fit_A_at_rank(S[:, tr], U[:, tr], r)
        resid = A @ S[:, te] + U[:, te]
        num_sq += float(np.sum(resid ** 2))
        den_sq += float(np.sum(U[:, te] ** 2))
        per_fold.append({"fold": i, "picked_rank": int(r),
                         "n_train": int(len(tr)), "n_test": int(len(te)),
                         "rho_fold": float(np.sqrt(np.sum(resid ** 2)
                                                    / max(np.sum(U[:, te] ** 2), 1e-30)))})
        picked.append(int(r))
    return {"rho_pooled": float(np.sqrt(num_sq / max(den_sq, 1e-30))),
            "per_fold": per_fold, "picked_ranks_summary": {
                "min": int(min(picked)) if picked else None,
                "median": int(np.median(picked)) if picked else None,
                "max": int(max(picked)) if picked else None,
            }}


def _derive_targets(guide_names, stored_map):
    if stored_map:
        # Some retained_guides keys are numpy str; force plain str.
        clean = {str(k): str(v) for k, v in dict(stored_map).items()}
        return np.array([clean.get(str(g), str(g)) for g in guide_names])
    out = []
    for g in guide_names:
        s = str(g)
        if s.startswith("guide_"):
            out.append(s[len("guide_"):])
        else:
            out.append(s.split("_")[0])
    return np.array(out)


def _linear_sim_nested(U, target_of_g, sigma, alpha_S, seed_base, n_reps):
    d, m = U.shape
    rhos = []
    per_rep_picks = []
    for r in range(n_reps):
        rng = np.random.default_rng(seed_base + r + 5000)
        J_ref = rng.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
        J = J_ref / max(alpha_S, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_obs = S_true + sigma * rng.normal(size=S_true.shape)
        out = _nested_cv_rho(S_obs, U, target_of_g, seed_base + r + 5000)
        rhos.append(out["rho_pooled"])
        per_rep_picks.append(out["picked_ranks_summary"]["median"])
    return {"n_reps": n_reps, "sigma": sigma, "alpha_S": alpha_S,
            "rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos)),
            "picked_rank_median_summary": [int(x) if x is not None else None
                                            for x in per_rep_picks]}


def main():
    payload = {"seed": SEED, "n_outer": N_OUTER, "n_inner": N_INNER,
               "rank_grid": RANK_GRID, "datasets": {}}
    for tag, cfg in DATASETS.items():
        with (RESULTS / cfg["pkl"]).open("rb") as f:
            b = pickle.load(f)
        meas = b["measurement"]
        S = meas.S; U = meas.U
        guides = list(meas.report.retained_guides)
        stored_map = getattr(meas.report, "guide_targets", None)
        target_of_g = _derive_targets(guides, stored_map)
        n_targets = len(set(target_of_g.tolist()))
        note = ""
        if len(guides) == n_targets:
            note = "one guide per target (target-grouped folds ≡ guide-grouped folds)"
        else:
            note = f"{n_targets} targets across {len(guides)} sgRNAs; siblings held out together"
        real = _nested_cv_rho(S, U, target_of_g, SEED)
        lin = _linear_sim_nested(U, target_of_g, cfg["sigma"], cfg["alpha_S"],
                                 seed_base=SEED, n_reps=N_REPS_LIN_SIM)
        payload["datasets"][tag] = {
            "note": note,
            "n_sgRNA": int(len(guides)),
            "n_targets": int(n_targets),
            "sigma": cfg["sigma"], "alpha_S": cfg["alpha_S"],
            "real_nested_cv": real,
            "linear_truth_matched_alpha_nested_cv": lin,
        }
        print(f"\n### {tag}: {note}")
        print(f"  real ρ (nested-CV, target folds) = {real['rho_pooled']:.4f}   "
              f"picked ranks {real['picked_ranks_summary']}")
        print(f"  linear-truth matched-α ρ (nested-CV) = {lin['rho_mean']:.4f} ± {lin['rho_std']:.4f}")
    (OUT / "F_nested_cv_rho.json").write_text(json.dumps(payload, indent=2))
    print("\nsaved:", OUT / "F_nested_cv_rho.json")


if __name__ == "__main__":
    main()
