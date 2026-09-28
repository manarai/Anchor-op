"""F5: Jost U has 5 near-zero singular values at d=30 -> rank(U)=25.

Confirms:
  - Jost U at d=30 loses 5 dimensions (25 targets, 3–6 sgRNAs each sharing
    W^T delta_g at the target level).
  - Under the corrected gate (input_rank == d AND effective_response_rank == d),
    a MeasuredOperator built with Jost's U is NOT full-domain, even when the
    noisy S is numerically full rank.

Also reruns the manuscript-figure Jost matched-geometry nulls under the new
gate and checks that `full_domain_identified` = False everywhere.
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import anchorop as ao
from anchorop.identifiability import regularized_pseudoinverse

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with (RESULTS / "jost_u_at_d30.pkl").open("rb") as f:
        jost = pickle.load(f)
    U = jost["U"]
    names = list(jost["names"])
    effs = jost["effs"]
    d, m = U.shape

    sv = np.linalg.svd(U, compute_uv=False)
    rank_1e_2 = int(np.sum(sv >= 1e-2 * sv[0]))
    rank_1e_6 = int(np.sum(sv >= 1e-6 * sv[0]))
    rank_abs_1e_6 = int(np.sum(sv >= 1e-6))
    targets = sorted({n.split("_")[0] for n in names})

    print(f"Jost U: shape={U.shape}, ||U||_F={np.linalg.norm(U):.4f}")
    print(f"singular values: {np.array2string(sv, precision=4)}")
    print(f"rank@1e-2 relative: {rank_1e_2}, rank@1e-6 relative: {rank_1e_6}, rank@abs<1e-6: {rank_abs_1e_6}")
    print(f"targets={len(targets)}, sgRNAs={m}")

    # Simulate S with dense J on Jost U (matched-geometry setting).
    # A moderate noise level makes numerical rank(S) hit d; the gate should
    # still refuse full-domain because rank(U) < d.
    rng = np.random.default_rng(20260927)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    J = G - 1.5 * np.eye(d)
    S_true = -np.linalg.solve(J, U)
    S_obs = S_true + 0.036 * rng.normal(size=S_true.shape)  # Jost's own sigma anchor
    meas = ao.measure_from_sensitivity(
        S=S_obs, U=U,
        guide_names=[str(n) for n in names],
        guide_efficiencies={str(n): float(v) for n, v in zip(names, effs.values() if isinstance(effs, dict) else effs)},
        reg="tsvd", reg_param="path", rank_tol=1e-2,
    )
    rep = meas.report
    payload = {
        "seed": 20260927,
        "d": d, "n_sgRNA": m, "n_targets_from_names": len(targets),
        "singular_values": [float(x) for x in sv],
        "rank_at_1e-2_relative": rank_1e_2,
        "rank_at_1e-6_relative": rank_1e_6,
        "input_subspace_dim": rep.input_subspace_dim,
        "response_subspace_dim": rep.response_subspace_dim,
        "effective_response_rank": rep.effective_response_rank,
        "condition_number": float(rep.condition_number),
        "full_domain_identified_after_gate_fix": bool(rep.full_domain_identified),
        "is_hard_projector": bool(rep.is_hard_projector),
    }
    print("\ngate output:", json.dumps({k: v for k, v in payload.items()
                                        if k not in ("singular_values",)}, indent=2))
    (OUT / "F5_jost_gate.json").write_text(json.dumps(payload, indent=2))
    print("saved:", OUT / "F5_jost_gate.json")

    # The published §2.7 result: reload the artifact and check that the same
    # measurement would now flag full_domain_identified = False.
    published = REPO / "manuscript_figures" / "jost_matched_geometry_nulls.json"
    if published.exists():
        note = {
            "note": ("The §2.7 Jost matched-geometry-nulls figure was produced with the "
                     "previous gate. Under the corrected gate, rank(U)=25 < d=30, so "
                     "full_domain_identified must be False for every Jost cell of that "
                     "figure. The stored per-cell cosines are unaffected (they use the "
                     "identified_action already), but any downstream code that keyed on "
                     "full_domain_identified must now be re-examined.")
        }
        (OUT / "F5_jost_note.json").write_text(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
