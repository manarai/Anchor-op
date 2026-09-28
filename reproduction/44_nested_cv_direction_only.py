"""Nested-CV direction-only ρ on all three screens (real and matched linear truth).

Column-normalize S and U (each column divided by its own Frobenius norm)
before the fit, then run the same nested cross-validation recipe as in
reproduction/40_nested_cv_rho.py: 5-fold target-grouped outer folds,
3-fold target-grouped inner folds selecting a TSVD rank from
{0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}. Report pooled outer-fold ρ with
across-outer-fold SD on the real data, and mean ± SD across 15 matched-
SNR linear-truth simulations under the same recipe (also
column-normalized).

Runs off the stored measurement pkls. Minutes.
Output: results/recheck/F_nested_cv_direction_only.json.
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
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]

DATASETS = {
    "K562_essential": {"pkl": "k562_essential_measurement.pkl", "sigma": 0.240, "alpha_S": 369.0},
    "RPE1_essential": {"pkl": "rpe1_essential_measurement.pkl", "sigma": 0.352, "alpha_S": 199.0},
    "Jost_2020":      {"pkl": "jost_measurement.pkl",          "sigma": 0.066, "alpha_S": 29.6},
}


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


def _target_folds(target_of_g, k, seed):
    rng = np.random.default_rng(seed)
    targets = sorted(set(target_of_g.tolist()))
    rng.shuffle(targets)
    tg = np.array_split(np.array(targets, dtype=object), k)
    out = []
    for i in range(k):
        te = set(tg[i].tolist())
        m = np.array([t in te for t in target_of_g])
        out.append((np.where(~m)[0], np.where(m)[0]))
    return out


def _inner_pick(S_tr, U_tr, target_of_tr, seed):
    ts = sorted(set(target_of_tr.tolist()))
    k = min(N_INNER, len(ts))
    if k < 2:
        return 0
    rng = np.random.default_rng(seed)
    tp = np.array(ts, dtype=object); rng.shuffle(tp)
    tg = np.array_split(tp, k)
    inner = []
    for i in range(k):
        te = set(tg[i].tolist())
        m = np.array([t in te for t in target_of_tr])
        inner.append((np.where(~m)[0], np.where(m)[0]))
    best_r, best = None, np.inf
    for r in RANK_GRID:
        pooled, ok = 0.0, True
        for tr, val in inner:
            if r > 0 and len(tr) <= r:
                ok = False; break
            A = _fit_A_at_rank(S_tr[:, tr], U_tr[:, tr], r)
            pooled += _mse(A, S_tr[:, val], U_tr[:, val])
        if ok and pooled < best:
            best, best_r = pooled, r
    return best_r if best_r is not None else 0


def _nested_rho(S, U, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, picks, per_fold = 0.0, 0.0, [], []
    for i, (tr, te) in enumerate(outer):
        r = _inner_pick(S[:, tr], U[:, tr], target_of_g[tr], seed + i * 17)
        if r > 0 and len(tr) <= r:
            r = 0
        A = _fit_A_at_rank(S[:, tr], U[:, tr], r)
        resid = A @ S[:, te] + U[:, te]
        num += float(np.sum(resid ** 2))
        den += float(np.sum(U[:, te] ** 2))
        picks.append(int(r))
        per_fold.append(float(np.sqrt(np.sum(resid ** 2) / max(np.sum(U[:, te] ** 2), 1e-30))))
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "picked_rank_median": int(np.median(picks)),
            "per_fold_rho": per_fold,
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0}


def _column_normalize(S, U):
    sn = np.linalg.norm(S, axis=0)
    un = np.linalg.norm(U, axis=0)
    keep = (sn > 0) & (un > 0)
    S = S[:, keep] / sn[keep]
    U = U[:, keep] / un[keep]
    return S, U, keep


def _derive_targets(guides, stored):
    if stored:
        clean = {str(k): str(v) for k, v in dict(stored).items()}
        return np.array([clean.get(str(g), str(g)) for g in guides])
    out = []
    for g in guides:
        s = str(g)
        if s.startswith("guide_"):
            out.append(s[len("guide_"):])
        else:
            out.append(s.split("_")[0])
    return np.array(out)


def main():
    payload = {"seed": SEED, "n_outer": N_OUTER, "n_inner": N_INNER,
               "rank_grid": RANK_GRID, "n_reps_lin_sim": N_REPS_LIN_SIM,
               "note": ("Direction-only nested-CV: column-normalize S and U per guide, "
                        "then run the same target-grouped nested-CV recipe as script 40."),
               "datasets": {}}
    for tag, cfg in DATASETS.items():
        with (RESULTS / cfg["pkl"]).open("rb") as f:
            b = pickle.load(f)
        meas = b["measurement"]
        S_raw = meas.S; U_raw = meas.U
        guides_all = list(meas.report.retained_guides)
        stored_map = getattr(meas.report, "guide_targets", None)

        # Column-normalize and drop any zero-column guides
        S, U, keep = _column_normalize(S_raw, U_raw)
        guides = [guides_all[i] for i in range(len(guides_all)) if keep[i]]
        target_of_g = _derive_targets(guides, stored_map)
        n_targets = len(set(target_of_g.tolist()))
        d, m = U.shape

        real = _nested_rho(S, U, target_of_g, SEED)

        # Matched-SNR linear-truth control, column-normalized after adding noise.
        # For like-for-like: build S_obs on the ORIGINAL U (not column-normalized),
        # then column-normalize both S_obs and U before running nested CV. This
        # matches how the real column-normalized ρ is computed.
        rhos, picks = [], []
        for r in range(N_REPS_LIN_SIM):
            rng = np.random.default_rng(SEED + r + 6000)
            J_ref = rng.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
            J = J_ref / max(cfg["alpha_S"], 1e-30)
            S_true = -np.linalg.solve(J, U_raw[:, keep])  # use raw U on the retained guides
            S_obs = S_true + cfg["sigma"] * rng.normal(size=S_true.shape)
            # column-normalize
            sn = np.linalg.norm(S_obs, axis=0)
            un = np.linalg.norm(U_raw[:, keep], axis=0)
            k2 = (sn > 0) & (un > 0)
            S_norm = (S_obs[:, k2] / sn[k2])
            U_norm = (U_raw[:, keep][:, k2] / un[k2])
            tot = target_of_g[k2]
            out = _nested_rho(S_norm, U_norm, tot, SEED + r + 6000)
            rhos.append(out["rho_pooled"])
            picks.append(out["picked_rank_median"])

        payload["datasets"][tag] = {
            "n_sgRNA_used": int(m),
            "n_targets": int(n_targets),
            "sigma": cfg["sigma"], "alpha_S": cfg["alpha_S"],
            "real_nested_cv_direction_only": real,
            "linear_truth_matched_alpha_nested_cv_direction_only": {
                "n_reps": N_REPS_LIN_SIM,
                "rho_mean": float(np.mean(rhos)),
                "rho_std": float(np.std(rhos, ddof=1)),
                "picked_rank_median_across_reps": int(np.median(picks)),
            },
        }
        print(f"\n### {tag}")
        print(f"  real ρ (nested-CV, direction-only) = {real['rho_pooled']:.4f} (SD {real['per_fold_sd']:.4f}); "
              f"picked ranks {real['picked_rank_median']}")
        print(f"  linear-truth matched-α ρ = {np.mean(rhos):.4f} ± {np.std(rhos, ddof=1):.4f} (N={N_REPS_LIN_SIM})")

    (OUT / "F_nested_cv_direction_only.json").write_text(json.dumps(payload, indent=2))
    print("\nsaved:", OUT / "F_nested_cv_direction_only.json")


if __name__ == "__main__":
    main()
