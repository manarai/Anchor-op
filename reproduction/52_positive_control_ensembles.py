"""Step 2: matched-SNR linear-truth positive control across five ground-truth
ensembles, on K562 / RPE1 / Jost.

Preregistered 2026-09-28 (commit 08560c3). α_S re-derived per (screen,
ensemble). Unstable draws (spectral abscissa > 0) rejected and re-drawn;
report the rejection fraction. N ≥ 15 accepted draws per cell.

Ensembles
---------
- dense:          `J_int = G` with `G_ij ~ N(0, 1/d)` (Ginibre-scaled).
- sparse-10 %:    Bernoulli mask, entries `~ N(0, 1/d)` where present, off-
                  diagonal only.
- sparse-2 %:     same at 2 % density.
- rank-5:         `J_int = B · V` with `B ∈ ℝ^(d×5), V ∈ ℝ^(5×d)` both scaled
                  by 1/√d.
- block-modular:  5 diagonal blocks of size ~ d/5. Within-block entries
                  ~ N(0, 1/√d); between-block entries ~ N(0, 1/(5√d)).

Metrics
-------
- Nested-CV real ρ (using the *current* K562/RPE1/Jost measurement pkls;
  matched linear-truth ρ per ensemble; reject if spectral abscissa
  of `J_true = J_int − 1.5 I` is > 0).
- Interaction-only Frobenius cosine of the fit A vs J_true (diagonal
  subtracted from both), N = 200 replicates, vs cross-replicate null
  (shift-1 pairing).

Output: results/recheck/F_step2_ensembles.json.
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore")

from anchorop.identifiability import regularized_pseudoinverse

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260928
RANK_TOL = 1e-2
N_OUTER = 5
N_INNER = 3
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
N_LIN_ACCEPT = 15
N_LIN_MAX_TRIES = 200
N_INT_REPS = 200

SCREENS = {
    "K562_essential": {"pkl": "k562_essential_measurement.pkl", "sigma": 0.240},
    "RPE1_essential": {"pkl": "rpe1_essential_measurement.pkl", "sigma": 0.352},
    "Jost_2020":      {"pkl": "jost_measurement.pkl",          "sigma": 0.066},
}


def _draw_J_interaction(structure, d, rng):
    if structure == "dense":
        return rng.normal(size=(d, d)) / np.sqrt(d)
    if structure == "sparse_10":
        G = rng.normal(size=(d, d)) / np.sqrt(d)
        mask = rng.random(size=(d, d)) < 0.10
        np.fill_diagonal(mask, False)
        return np.where(mask, G, 0.0)
    if structure == "sparse_2":
        G = rng.normal(size=(d, d)) / np.sqrt(d)
        mask = rng.random(size=(d, d)) < 0.02
        np.fill_diagonal(mask, False)
        return np.where(mask, G, 0.0)
    if structure == "rank_5":
        B = rng.normal(size=(d, 5)) / np.sqrt(d)
        V = rng.normal(size=(5, d)) / np.sqrt(d)
        return B @ V
    if structure == "block_modular":
        blocks = 5
        sizes = [d // blocks + (1 if i < d % blocks else 0) for i in range(blocks)]
        starts = np.cumsum([0] + sizes[:-1])
        # Between-block small, within-block larger
        J = rng.normal(size=(d, d)) / (5 * np.sqrt(d))
        for i in range(blocks):
            r0, r1 = starts[i], starts[i] + sizes[i]
            J[r0:r1, r0:r1] = rng.normal(size=(sizes[i], sizes[i])) / np.sqrt(d)
        return J
    raise ValueError(structure)


def _stable(J):
    return float(np.max(np.linalg.eigvals(J).real)) < 0.0


def _target_folds(target_of_g, k, seed):
    rng = np.random.default_rng(seed)
    ts = sorted(set(target_of_g.tolist()))
    rng.shuffle(ts)
    tg = np.array_split(np.array(ts, dtype=object), k)
    out = []
    for i in range(k):
        te = set(tg[i].tolist())
        m = np.array([t in te for t in target_of_g])
        out.append((np.where(~m)[0], np.where(m)[0]))
    return out


def _fit_A_at_rank(S, U, r):
    r = int(r)
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]))
    Us, sv, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(sv))
    inv = np.zeros_like(sv); inv[:r] = 1.0 / sv[:r]
    return -U @ ((Vt.T * inv) @ Us.T)


def _inner_pick(S_tr, U_tr, target_of_tr, seed):
    ts = sorted(set(target_of_tr.tolist()))
    k = min(N_INNER, len(ts))
    if k < 2: return 0
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
        for tr_in, val in inner:
            if r > 0 and len(tr_in) <= r: ok = False; break
            A = _fit_A_at_rank(S_tr[:, tr_in], U_tr[:, tr_in], r)
            pooled += float(np.sum((A @ S_tr[:, val] + U_tr[:, val]) ** 2))
        if ok and pooled < best: best, best_r = pooled, r
    return best_r if best_r is not None else 0


def _nested_rho(S, U, target_of_g, seed):
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, picks, per_fold = 0.0, 0.0, [], []
    for i, (tr, te) in enumerate(outer):
        r = _inner_pick(S[:, tr], U[:, tr], target_of_g[tr], seed + i * 17)
        if r > 0 and len(tr) <= r: r = 0
        A = _fit_A_at_rank(S[:, tr], U[:, tr], r)
        resid = A @ S[:, te] + U[:, te]
        num += float(np.sum(resid ** 2))
        den += float(np.sum(U[:, te] ** 2))
        picks.append(int(r))
        per_fold.append(float(np.sqrt(np.sum(resid ** 2) / max(np.sum(U[:, te] ** 2), 1e-30))))
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
            "picked_rank_median": int(np.median(picks))}


def _cos_full(A, J):
    return float(np.sum(A * J)) / max(np.linalg.norm(A) * np.linalg.norm(J), 1e-30)


def _off_diag(M):
    m = M.copy(); np.fill_diagonal(m, 0.0); return m


def _fit_A_default(S_obs, U):
    S_pinv, *_ = regularized_pseudoinverse(S_obs, method="tsvd", parameter="path", rank_tol=RANK_TOL)
    return -U @ S_pinv


def run_screen(name, cfg):
    print(f"\n### {name}")
    b = pickle.load((RESULTS / cfg["pkl"]).open("rb"))
    meas = b["measurement"]
    S_real = meas.S; U = meas.U
    d, m = U.shape
    stored = getattr(meas.report, "guide_targets", None)
    if stored:
        stored = {str(k): str(v) for k, v in dict(stored).items()}
        target_of_g = np.array([stored.get(str(g), str(g)) for g in meas.report.retained_guides])
    else:
        target_of_g = np.array([str(g)[len("guide_"):] if str(g).startswith("guide_") else str(g).split("_")[0]
                                 for g in meas.report.retained_guides])
    S_med_real = float(np.median(np.linalg.norm(S_real, axis=0)))

    result = {"screen": name, "d": d, "n_guides": m, "sigma": cfg["sigma"],
              "S_median_real": S_med_real, "ensembles": {}}

    ENSEMBLES = ["dense", "sparse_10", "sparse_2", "rank_5", "block_modular"]
    for ens in ENSEMBLES:
        print(f"  [{ens}]")
        # α_S calibration: use the SAME construction as Table 1 (script 40),
        # where the shift is baked into J_ref and α_S rescales the whole J.
        #   J_ref = J_int - 1.5·I          (α = 1 reference)
        #   J     = J_ref / α_S            (S_true scales linearly with α_S)
        rng = np.random.default_rng(SEED + hash(ens) % 100000)
        tried, accepted = 0, 0
        ref_J = None
        while tried < N_LIN_MAX_TRIES:
            J_int = _draw_J_interaction(ens, d, rng)
            J = J_int - 1.5 * np.eye(d)
            tried += 1
            if _stable(J):
                ref_J = J; accepted += 1; break
        if ref_J is None:
            print(f"    ✗ no stable J for {ens} in {N_LIN_MAX_TRIES} tries")
            result["ensembles"][ens] = {"error": "no stable J drawn"}
            continue
        S_ref = -np.linalg.solve(ref_J, U)
        ref_med = float(np.median(np.linalg.norm(S_ref, axis=0)))
        alpha_S = S_med_real / max(ref_med, 1e-30)

        # ── nested-CV matched linear-truth ρ (15 accepted draws) ──
        rho_lin, tries_needed, rejected = [], 0, 0
        rng = np.random.default_rng(SEED + hash(ens) % 100000 + 7)
        while len(rho_lin) < N_LIN_ACCEPT and tries_needed < N_LIN_MAX_TRIES:
            J_int = _draw_J_interaction(ens, d, rng)
            J_ref = J_int - 1.5 * np.eye(d)
            J = J_ref / alpha_S
            tries_needed += 1
            if not _stable(J):
                rejected += 1; continue
            S_true = -np.linalg.solve(J, U)
            S_sim = S_true + cfg["sigma"] * rng.normal(size=S_true.shape)
            rho_lin.append(_nested_rho(S_sim, U, target_of_g, seed=SEED + tries_needed + hash(ens) % 100000)["rho_pooled"])

        # ── interaction-only cosine at matched α (N_INT_REPS accepted) ──
        Js, S_trues, A_fits = [], [], []
        int_reps_seen, int_rejected = 0, 0
        rng2 = np.random.default_rng(SEED + hash(ens) % 100000 + 42)
        while len(Js) < N_INT_REPS and int_reps_seen < N_INT_REPS * 4:
            J_int = _draw_J_interaction(ens, d, rng2)
            J_ref = J_int - 1.5 * np.eye(d)
            J = J_ref / alpha_S
            int_reps_seen += 1
            if not _stable(J):
                int_rejected += 1; continue
            S_true = -np.linalg.solve(J, U)
            S_obs = S_true + cfg["sigma"] * rng2.normal(size=S_true.shape)
            A = _fit_A_default(S_obs, U)
            Js.append(J); S_trues.append(S_true); A_fits.append(A)
        cos_int = [_cos_full(_off_diag(A_fits[r]), _off_diag(Js[r])) for r in range(len(Js))]
        null_int = [_cos_full(_off_diag(A_fits[r]), _off_diag(Js[(r + 1) % len(Js)])) for r in range(len(Js))]

        result["ensembles"][ens] = {
            "alpha_S": float(alpha_S),
            "reject_rate_lin": float(rejected / max(tries_needed, 1)),
            "reject_rate_int": float(int_rejected / max(int_reps_seen, 1)),
            "linear_truth_nested_cv_rho_mean": float(np.mean(rho_lin)),
            "linear_truth_nested_cv_rho_std": float(np.std(rho_lin, ddof=1)) if len(rho_lin) > 1 else 0.0,
            "linear_truth_nested_cv_n_accepted": len(rho_lin),
            "interaction_only_cos_mean": float(np.mean(cos_int)),
            "interaction_only_cos_std": float(np.std(cos_int, ddof=1)),
            "interaction_only_null_mean": float(np.mean(null_int)),
            "interaction_only_null_std": float(np.std(null_int, ddof=1)),
            "n_int_accepted": len(Js),
            "beats_predict_zero": bool(np.mean(rho_lin) < 1.0 - 2 * (np.std(rho_lin, ddof=1) / np.sqrt(len(rho_lin)))),
        }
        print(f"    α_S={alpha_S:.2f}   reject_lin={rejected/max(tries_needed,1):.2%}   "
              f"lin-truth ρ={np.mean(rho_lin):.3f}±{np.std(rho_lin, ddof=1):.3f}  "
              f"cos_int={np.mean(cos_int):.3f} vs null {np.mean(null_int):.3f}")
    return result


def main():
    payload = {"seed": SEED, "n_lin_accept": N_LIN_ACCEPT, "n_int_reps": N_INT_REPS,
               "screens": {}}
    for name, cfg in SCREENS.items():
        payload["screens"][name] = run_screen(name, cfg)
    out_path = OUT / "F_step2_ensembles.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {out_path}")
    # summary
    print("\n=== SUMMARY (matched-SNR linear-truth ρ beats predict-zero?) ===")
    for name, s in payload["screens"].items():
        print(f"\n[{name}]")
        for ens, r in s["ensembles"].items():
            if "error" in r:
                print(f"  {ens:<15s} ERROR: {r['error']}")
                continue
            print(f"  {ens:<15s} ρ={r['linear_truth_nested_cv_rho_mean']:.3f}±{r['linear_truth_nested_cv_rho_std']:.3f}  "
                  f"cos_int={r['interaction_only_cos_mean']:.3f}  beats_zero={r['beats_predict_zero']}")


if __name__ == "__main__":
    main()
