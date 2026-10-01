"""A1 reproduction gate (preregistered 2026-09-30).

Runs the PCA-30 arm of experiment 1 through this branch's code with
the same target-grouped 5-outer × 3-inner nested-CV recipe as
anchor-op paper 1 (MANUSCRIPT.md §4.5), with the TSVD rank grid
including 0, on the Replogle K562 essential measurement bundle.

Pass gate: real K562 ρ ≈ 0.96 and matched-linear-truth ρ ≈ 0.18,
reproducing Table 1 of the preprint. If the branch's StateSpace +
anchor-op loop cannot reproduce those values within rounding, STOP
and report before any encoder arm runs.

Output: experiments/exp1_statespace/A1_reproduction_gate.json
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PKL = REPO / "results" / "k562_essential_measurement.pkl"
OUT = Path(__file__).parent / "A1_reproduction_gate.json"

SEED = 20260930
N_OUTER = 5
N_INNER = 3
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
N_LIN_SIM = 15
SIGMA = 0.240   # per-screen anchor from §4.4
ALPHA_S = 369.0  # per-screen matched-α from F1_F4_summary.json


# ─── nested-CV plumbing (same as reproduction/40_nested_cv_rho.py) ────
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


def _tsvd_A(S, U, r):
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]))
    Us, s, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(s))
    S_pinv = Vt[:r].T @ np.diag(1.0 / s[:r]) @ Us[:, :r].T
    return -U @ S_pinv


def _inv_rho(A, S_te, U_te):
    num = np.sum((A @ S_te + U_te) ** 2)
    den = np.sum(U_te ** 2)
    return float(np.sqrt(num / max(den, 1e-30)))


def _nested_rho(S, U, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    picks, rhos, num, den = [], [], 0.0, 0.0
    for i, (tr, te) in enumerate(outer):
        inner_targets = target_of_g[tr]
        inner = _target_folds(
            inner_targets,
            min(N_INNER, len(set(inner_targets))),
            seed + i * 17)
        best_r, best_mse = None, np.inf
        for r in RANK_GRID:
            pooled = 0.0
            for tr_in, val_in in inner:
                gtr = tr[tr_in]; gval = tr[val_in]
                A = _tsvd_A(S[:, gtr], U[:, gtr], r)
                pooled += float(np.sum((A @ S[:, gval] + U[:, gval]) ** 2))
            if pooled < best_mse:
                best_mse, best_r = pooled, r
        picks.append(best_r)
        A = _tsvd_A(S[:, tr], U[:, tr], best_r)
        num += float(np.sum((A @ S[:, te] + U[:, te]) ** 2))
        den += float(np.sum(U[:, te] ** 2))
        rhos.append(_inv_rho(A, S[:, te], U[:, te]))
    return {
        "rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
        "per_fold_rho": rhos,
        "per_fold_sd": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
        "picked_ranks": picks,
    }


def _matched_linear_truth(U, target_of_g, sigma, alpha_S, seed, n_reps=N_LIN_SIM):
    rhos = []
    d = U.shape[0]
    for r in range(n_reps):
        rng = np.random.default_rng(seed + r + 8000)
        G = rng.normal(size=(d, d)) / np.sqrt(d)
        J_ref = G - 1.5 * np.eye(d)
        J = J_ref / max(alpha_S, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_sim = S_true + sigma * rng.normal(size=S_true.shape)
        out = _nested_rho(S_sim, U, target_of_g, seed=seed + r + 8000)
        rhos.append(out["rho_pooled"])
    return {
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos, ddof=1)),
        "n_reps": n_reps,
    }


def main():
    if not PKL.exists():
        print(f"ERROR: {PKL} missing. Run reproduction/03_fig3_k562_essential.py "
              "or an equivalent that writes k562_essential_measurement.pkl.")
        sys.exit(1)
    bundle = pickle.load(open(PKL, "rb"))
    meas = bundle["measurement"]
    S = meas.S.astype(np.float64)
    U = meas.U.astype(np.float64)
    guides = list(meas.report.retained_guides)
    stored_map = getattr(meas.report, "guide_targets", None)
    if stored_map:
        clean = {str(k): str(v) for k, v in dict(stored_map).items()}
        target_of_g = np.array([clean.get(str(g), str(g)) for g in guides])
    else:
        # Fall back to the convention used elsewhere in the repo.
        target_of_g = np.array([
            s[len("guide_"):] if str(s).startswith("guide_") else str(s).split("_")[0]
            for s in guides
        ])

    print(f"[A1] K562 essential: {U.shape[0]}-dim program space, "
          f"{S.shape[1]} guides, {len(set(target_of_g.tolist()))} targets")

    print(f"[A1] real target-held-out nested-CV ρ …")
    real = _nested_rho(S, U, target_of_g, SEED)
    print(f"      ρ = {real['rho_pooled']:.4f} ± {real['per_fold_sd']:.4f}  "
          f"picked ranks = {real['picked_ranks']}")

    print(f"[A1] matched linear-truth ρ at α_S = {ALPHA_S}, σ = {SIGMA} "
          f"({N_LIN_SIM} replicates) …")
    lin = _matched_linear_truth(U, target_of_g, SIGMA, ALPHA_S, SEED)
    print(f"      ρ = {lin['rho_mean']:.4f} ± {lin['rho_std']:.4f} "
          f"(N = {lin['n_reps']})")

    # Gate: K562 real ≈ 0.96 (tolerance ± 0.03) and matched-linear ≈ 0.18
    # (tolerance ± 0.03). Values are the preprint's Table 1.
    tol = 0.03
    gate_real = abs(real["rho_pooled"] - 0.96) <= tol
    gate_lin = abs(lin["rho_mean"] - 0.18) <= tol
    passed = bool(gate_real and gate_lin)

    report = {
        "seed": SEED,
        "pkl": str(PKL.relative_to(REPO)),
        "sigma": SIGMA, "alpha_S": ALPHA_S,
        "n_guides": int(S.shape[1]),
        "n_targets": int(len(set(target_of_g.tolist()))),
        "n_outer": N_OUTER, "n_inner": N_INNER,
        "rank_grid": RANK_GRID,
        "real": real,
        "matched_linear_truth": lin,
        "reference_table1_K562_real": 0.96,
        "reference_table1_K562_matched_linear": 0.18,
        "tolerance": tol,
        "gate_real_within_tolerance": bool(gate_real),
        "gate_matched_linear_within_tolerance": bool(gate_lin),
        "passed": passed,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[A1] saved: {OUT}")
    print(f"[A1] VERDICT: {'PASS' if passed else 'FAIL'}  "
          f"(real {real['rho_pooled']:.4f} vs 0.96, "
          f"matched-linear {lin['rho_mean']:.4f} vs 0.18, tol ±{tol})")
    if not passed:
        print("[A1] HALT: branch's recipe does not reproduce Table 1. "
              "Stop before any encoder arm runs.")
        sys.exit(2)


if __name__ == "__main__":
    main()
