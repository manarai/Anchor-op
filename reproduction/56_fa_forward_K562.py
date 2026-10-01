"""K562 forward comparator (fixed encoding) — PCA vs FA(QR) through the 55_ pipeline.

Companion to 55_fa_table1_recheck.py. For both arms produces:
  - rho_fwd(forward_fixed)                     — real nested forward rho (+ fold SD)
  - rho_fwd(training_mean_baseline)            — predict-training-mean baseline
  - rho_fwd(forward_fixed_matched_linear)      — matched-SNR linear-truth control
Everything else (K_outer=5, K_inner=3, lambda grid, training-mean intercept,
seeds) matches 51_comparator_panel.py.

Reads:
  - results/k562_essential_pca55_measurement.pkl
  - results/k562_essential_fa_measurement.pkl
  - results/recheck/F_pca_fa_nested_cv_rho.json  (for sigma/alpha_S per arm)
Writes:
  - results/recheck/F_pca_fa_K562_forward_fixed.json

Runtime <2 min.
"""
from __future__ import annotations
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260928
N_OUTER = 5
N_INNER = 3
N_LIN_SIM = 15
LAMBDA_GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
               1e3, 1e4, 1e6, 1e30]


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


def _ridge_B(U_tr, S_tr, lam):
    d_in = U_tr.shape[0]
    if not np.isfinite(lam) or lam > 1e30:
        return np.zeros((S_tr.shape[0], d_in))
    return S_tr @ U_tr.T @ np.linalg.inv(U_tr @ U_tr.T + lam * np.eye(d_in))


def _fwd_mse(S_pred, S_test):
    return float(np.sum((S_pred - S_test) ** 2))


def _fwd_rho(S_pred, S_test):
    r = S_pred - S_test
    return float(np.sqrt(np.sum(r ** 2) / max(np.sum(S_test ** 2), 1e-30)))


def _nested_forward(S, U, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, picks, per_fold = 0.0, 0.0, [], []
    for i, (tr, te) in enumerate(outer):
        inner_targets = target_of_g[tr]
        inner = _target_folds(inner_targets,
                              min(N_INNER, len(set(inner_targets))),
                              seed + i * 17)
        best_lam, best_mse = None, np.inf
        for lam in LAMBDA_GRID:
            pooled_mse = 0.0
            for tr_in, val_in in inner:
                gtr = tr[tr_in]; gval = tr[val_in]
                b_in = np.mean(S[:, gtr], axis=1, keepdims=True)
                B = _ridge_B(U[:, gtr], S[:, gtr] - b_in, lam)
                Shat_val = B @ U[:, gval] + b_in
                pooled_mse += _fwd_mse(Shat_val, S[:, gval])
            if pooled_mse < best_mse:
                best_mse, best_lam = pooled_mse, lam
        b_out = np.mean(S[:, tr], axis=1, keepdims=True)
        B = _ridge_B(U[:, tr], S[:, tr] - b_out, best_lam)
        Shat = B @ U[:, te] + b_out
        num += _fwd_mse(Shat, S[:, te])
        den += float(np.sum(S[:, te] ** 2))
        per_fold.append(_fwd_rho(Shat, S[:, te]))
        picks.append(best_lam)
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_rho": per_fold,
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
            "picked_lambda": picks}


def _training_mean_baseline_forward(S, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, per_fold = 0.0, 0.0, []
    for tr, te in outer:
        mean_col = np.mean(S[:, tr], axis=1, keepdims=True)
        Shat = np.tile(mean_col, (1, len(te)))
        num += _fwd_mse(Shat, S[:, te])
        den += float(np.sum(S[:, te] ** 2))
        per_fold.append(_fwd_rho(Shat, S[:, te]))
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_rho": per_fold,
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0}


def _matched_linear_truth_forward(U, target_of_g, sigma, alpha_S, seed, n_reps=N_LIN_SIM):
    rhos = []
    for r in range(n_reps):
        rng = np.random.default_rng(seed + r + 9000)
        J_ref = rng.normal(size=(U.shape[0], U.shape[0])) / np.sqrt(U.shape[0]) \
                 - 1.5 * np.eye(U.shape[0])
        J = J_ref / max(alpha_S, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_sim = S_true + sigma * rng.normal(size=S_true.shape)
        out = _nested_forward(S_sim, U, target_of_g, seed=seed + r + 9000)
        rhos.append(out["rho_pooled"])
    return {"n_reps": n_reps, "sigma": sigma, "alpha_S": alpha_S,
            "rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos, ddof=1))}


def _guide_targets(meas):
    stored = getattr(meas.report, "guide_targets", None)
    if stored:
        return {str(k): str(v) for k, v in dict(stored).items()}
    out = {}
    for g in meas.report.retained_guides:
        s = str(g)
        out[s] = s[len("guide_"):] if s.startswith("guide_") else s.split("_")[0]
    return out


def _one_arm(arm_name, bundle_path):
    print(f"\n--- K562 / {arm_name} ---", flush=True)
    with Path(bundle_path).open("rb") as f:
        b = pickle.load(f)
    meas = b["measurement"]
    S = meas.S; U = meas.U
    guides = list(meas.report.retained_guides)
    gt_map = _guide_targets(meas)
    target_of_g = np.array([gt_map[str(g)] for g in guides])
    print(f"  d={S.shape[0]}, m={S.shape[1]}, n_targets={len(set(target_of_g.tolist()))}",
          flush=True)

    table = json.loads((OUT / "F_pca_fa_nested_cv_rho.json").read_text())
    k562 = table["datasets"]["K562_essential"][arm_name.upper()]
    sigma = float(k562["sigma"])
    alpha_S = float(k562["alpha_S"])
    print(f"  sigma={sigma:.4f}, alpha_S={alpha_S:.2f}", flush=True)

    print(f"  forward_fixed …", flush=True)
    forward_fixed = _nested_forward(S, U, target_of_g, SEED)
    print(f"    rho_fwd = {forward_fixed['rho_pooled']:.4f} "
          f"(fold SD {forward_fixed['per_fold_sd']:.4f})", flush=True)

    print(f"  training_mean_baseline …", flush=True)
    train_mean = _training_mean_baseline_forward(S, target_of_g, SEED)
    print(f"    rho_fwd = {train_mean['rho_pooled']:.4f} "
          f"(fold SD {train_mean['per_fold_sd']:.4f})", flush=True)

    print(f"  matched-linear-truth forward …", flush=True)
    matched = _matched_linear_truth_forward(U, target_of_g, sigma, alpha_S, SEED)
    print(f"    rho_fwd = {matched['rho_mean']:.4f} +/- {matched['rho_std']:.4f}",
          flush=True)

    return {
        "sigma": sigma, "alpha_S": alpha_S,
        "n_sgRNA": int(len(guides)),
        "n_targets": int(len(set(target_of_g.tolist()))),
        "forward_fixed": forward_fixed,
        "training_mean_baseline": train_mean,
        "forward_fixed_matched_linear": matched,
    }


def main():
    pca_bundle = RESULTS / "k562_essential_pca55_measurement.pkl"
    fa_bundle = RESULTS / "k562_essential_fa_measurement.pkl"
    for p in (pca_bundle, fa_bundle):
        if not p.exists():
            raise SystemExit(f"missing {p} — run reproduction/55_fa_table1_recheck.py first")

    payload = {
        "seed": SEED,
        "n_outer": N_OUTER, "n_inner": N_INNER,
        "lambda_grid": LAMBDA_GRID,
        "arms": {},
    }
    payload["arms"]["PCA"] = _one_arm("pca", pca_bundle)
    payload["arms"]["FA"] = _one_arm("fa", fa_bundle)

    out_path = OUT / "F_pca_fa_K562_forward_fixed.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
