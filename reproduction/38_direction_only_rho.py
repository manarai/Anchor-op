"""Direction-only held-out ρ on Replogle (settles F4 reading (c)).

Reviewer test: κ only rescales columns of S and U. If we remove
per-column scale (column-normalize S and U, or fit with a free
per-guide scale) and refit, and ρ still fails, then (c) κ-proxy
mis-scaling cannot be the explanation for the linearity gap.

Primary variant: column-normalize each guide's S and U by its own
Frobenius column norm. This removes ALL per-column scale — the fit
now sees only directions, so any per-guide κ error is absorbed by
construction. Then run held_out_prediction_check.

Secondary variant: free per-guide scale c_g fitted jointly with A to
minimize sum_g ||A s_g + c_g u_g||^2. Equivalent per column up to
sign; not reported here — the column-normalize path is simpler and
strictly stronger (it discards all per-column magnitude, not just
each guide's specific scale).

Runtime: seconds. Runs on results/*_essential_measurement.pkl.
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import anchorop as ao

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

BUNDLE = {"K562_essential": "k562_essential_measurement.pkl",
          "RPE1_essential": "rpe1_essential_measurement.pkl"}
SEED = 20260927
RANK_TOL = 1e-2


def _derive_targets(guide_names):
    out = {}
    for g in guide_names:
        s = str(g)
        out[s] = s[len("guide_"):] if s.startswith("guide_") else s
    return out


def _direction_only(meas):
    S = meas.S.copy()
    U = meas.U.copy()
    s_norm = np.linalg.norm(S, axis=0)
    u_norm = np.linalg.norm(U, axis=0)
    keep = (s_norm > 0) & (u_norm > 0)
    if not keep.all():
        S, U = S[:, keep], U[:, keep]
        s_norm, u_norm = s_norm[keep], u_norm[keep]
    S_n = S / s_norm
    U_n = U / u_norm
    guide_names = [g for i, g in enumerate(meas.report.retained_guides) if keep[i]]
    kappa = {g: meas.report.guide_efficiencies[g] for g in guide_names}
    return S_n, U_n, guide_names, kappa


def main():
    payload = {"seed": SEED, "rank_tol": RANK_TOL, "datasets": {}}
    for tag, path in BUNDLE.items():
        with (RESULTS / path).open("rb") as f:
            b = pickle.load(f)
        meas = b["measurement"]
        rho_real = ao.held_out_prediction_check(meas, n_folds=5, seed=SEED)
        rel_real = ao.linearity_check(meas)

        S_n, U_n, gnames, kappa = _direction_only(meas)
        tgt = getattr(meas.report, "guide_targets", None) or _derive_targets(gnames)
        # κ is not used by the fit after column-normalization, but pass the
        # real κ through so downstream metadata-consumers (e.g. linearity_check
        # splitting at the median κ) still see a meaningful distribution.
        meas_dir = ao.measure_from_sensitivity(
            S=S_n, U=U_n,
            guide_names=gnames,
            guide_efficiencies=kappa,
            guide_targets=tgt,
            reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
        )
        rho_dir = ao.held_out_prediction_check(meas_dir, n_folds=5, seed=SEED)
        rel_dir = ao.linearity_check(meas_dir)

        # Sanity: what happens on a synthetic linear ground truth at the same
        # (d, n) after column-normalization? Should give ρ ≪ 1 at published
        # α; this is a control that the direction-only estimator itself is
        # not the source of a ρ ≈ 1 outcome.
        d, m = S_n.shape
        rng = np.random.default_rng(SEED)
        J = rng.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
        # Use the real (unnormalized) U to have a matched design, then normalize.
        U_full = meas.U
        S_true = -np.linalg.solve(J, U_full)
        S_true_n = S_true / np.linalg.norm(S_true, axis=0)
        U_full_n = U_full / np.linalg.norm(U_full, axis=0)
        meas_lin_dir = ao.measure_from_sensitivity(
            S=S_true_n, U=U_full_n,
            guide_names=list(meas.report.retained_guides),
            guide_efficiencies=dict(meas.report.guide_efficiencies),
            guide_targets=getattr(meas.report, "guide_targets", None)
                          or _derive_targets(list(meas.report.retained_guides)),
            reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
        )
        rho_lin_dir = ao.held_out_prediction_check(meas_lin_dir, n_folds=5, seed=SEED)

        row = {
            "cell_line": tag,
            "d": int(S_n.shape[0]), "n_guides_used": int(S_n.shape[1]),
            "baseline": {
                "rho_real": float(rho_real.rho_pooled),
                "rel_diff_real": float(rel_real.relative_difference),
            },
            "direction_only": {
                "rho_real": float(rho_dir.rho_pooled),
                "rel_diff_real": float(rel_dir.relative_difference),
                "delta_rho": float(rho_dir.rho_pooled - rho_real.rho_pooled),
            },
            "control_linear_sim_direction_only": {
                "rho": float(rho_lin_dir.rho_pooled),
                "note": ("Noise-free linear J on real U, then column-normalized. "
                         "Should be near 0 if the direction-only estimator can "
                         "recover a linear operator when one is present. "
                         "A value >> baseline direction-only ρ_real would mean "
                         "the estimator itself has a floor.")
            },
            "reading_c_verdict": None,
        }
        # Verdict: (c) is live if direction-only ρ drops substantially (say
        # >0.5 below baseline). If ρ stays near 1, (c) can't be the story.
        delta = rho_dir.rho_pooled - rho_real.rho_pooled
        if delta < -0.5:
            row["reading_c_verdict"] = "reading (c) is LIVE: direction-only ρ drops substantially"
        elif abs(delta) < 0.1:
            row["reading_c_verdict"] = "reading (c) is RULED OUT: ρ essentially unchanged under column-normalization"
        else:
            row["reading_c_verdict"] = f"reading (c) is PARTIAL: ρ delta = {delta:+.3f}"
        payload["datasets"][tag] = row
        print(json.dumps(row, indent=2))

    (OUT / "F_direction_only_rho.json").write_text(json.dumps(payload, indent=2))
    print("saved:", OUT / "F_direction_only_rho.json")


if __name__ == "__main__":
    main()
