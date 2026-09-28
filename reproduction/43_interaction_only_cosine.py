"""Interaction-only cosine at matched α for K562 and RPE1.

Reviewer note (2026-09-27): the full-matrix cos ≈ 0.99 / 0.91 at
matched α is dominated by the shared −cI diagonal, so its cross-
replicate null is ~0.70. The manuscript should report the
interaction-only cosine — cos(A_int, J_int) where the diagonal
contribution is removed — against a null near zero.

Compute A_int = A − diag(A), J_int = J − diag(J), and their cosine
under the α_S matched-recovery regime (K562 α=369, RPE1 α=199).
Report mean and cross-replicate null (paired shift=1) at N=200.

Runs off results/{k562,rpe1}_essential_measurement.pkl.
Output: results/recheck/F_interaction_only_cosine.json.
"""
from __future__ import annotations
import json, pickle
from pathlib import Path

import numpy as np

from anchorop.identifiability import regularized_pseudoinverse

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260927
RANK_TOL = 1e-2
N_REPS = 200
DATASETS = {
    "K562_essential": {"pkl": "k562_essential_measurement.pkl", "sigma": 0.240, "alpha_S": 369.0},
    "RPE1_essential": {"pkl": "rpe1_essential_measurement.pkl", "sigma": 0.352, "alpha_S": 199.0},
}


def _fit_A(S, U):
    S_pinv, *_ = regularized_pseudoinverse(S, method="tsvd", parameter="path", rank_tol=RANK_TOL)
    return -U @ S_pinv


def _draw_J(d, seed, structure="dense", c=1.5):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    return G - c * np.eye(d)


def _off_diag(M):
    """Return M with its diagonal zeroed (interaction-only component)."""
    M2 = M.copy()
    np.fill_diagonal(M2, 0.0)
    return M2


def _cosine(A, B):
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    return float(np.sum(A * B) / max(denom, 1e-30))


def main():
    payload = {"seed": SEED, "rank_tol": RANK_TOL, "n_reps": N_REPS,
               "definition": ("interaction-only cosine subtracts the diagonal "
                              "of both A and J before computing the Frobenius cosine; "
                              "this removes the shared −cI baseline that dominates "
                              "the full-matrix cosine at matched α."),
               "datasets": {}}
    for tag, cfg in DATASETS.items():
        with (RESULTS / cfg["pkl"]).open("rb") as f:
            b = pickle.load(f)
        meas = b["measurement"]
        U = meas.U
        d, m = U.shape
        alpha_S = cfg["alpha_S"]
        sigma = cfg["sigma"]

        Js, S_trues, A_fits = [], [], []
        rng = np.random.default_rng(SEED + 42)
        for r in range(N_REPS):
            J_ref = _draw_J(d, SEED + r)
            J = J_ref / alpha_S
            S_true = -np.linalg.solve(J, U)
            S_obs = S_true + sigma * rng.normal(size=S_true.shape)
            A = _fit_A(S_obs, U)
            Js.append(J); S_trues.append(S_true); A_fits.append(A)

        cos_full = np.array([_cosine(A_fits[r], Js[r]) for r in range(N_REPS)])
        cos_int  = np.array([_cosine(_off_diag(A_fits[r]), _off_diag(Js[r])) for r in range(N_REPS)])

        null_full = np.array([_cosine(A_fits[r], Js[(r + 1) % N_REPS]) for r in range(N_REPS)])
        null_int  = np.array([_cosine(_off_diag(A_fits[r]), _off_diag(Js[(r + 1) % N_REPS]))
                              for r in range(N_REPS)])

        payload["datasets"][tag] = {
            "alpha_S": alpha_S, "sigma": sigma, "n_reps": N_REPS, "d": d, "n_guides": m,
            "cos_full_mean": float(np.mean(cos_full)),
            "cos_full_null_mean_shift1": float(np.mean(null_full)),
            "cos_interaction_only_mean": float(np.mean(cos_int)),
            "cos_interaction_only_std": float(np.std(cos_int, ddof=1)),
            "cos_interaction_only_null_mean_shift1": float(np.mean(null_int)),
            "cos_interaction_only_null_std_shift1": float(np.std(null_int, ddof=1)),
            "cos_interaction_only_paired_diff_mean": float(np.mean(cos_int - null_int)),
        }
        print(f"\n### {tag} @ matched α={alpha_S}, σ={sigma}, N={N_REPS}")
        print(f"  full cos: mean {np.mean(cos_full):.4f}  null(shift=1) {np.mean(null_full):.4f}")
        print(f"  interaction-only cos: mean {np.mean(cos_int):.4f} ± {np.std(cos_int, ddof=1):.4f}"
              f"   null(shift=1) {np.mean(null_int):.4f}"
              f"   paired-diff {np.mean(cos_int - null_int):.4f}")
    (OUT / "F_interaction_only_cosine.json").write_text(json.dumps(payload, indent=2))
    print("\nsaved:", OUT / "F_interaction_only_cosine.json")


if __name__ == "__main__":
    main()
