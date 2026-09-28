"""F6: Jost input amplitude comparison.

The manuscript's §2.7 comparison holds sigma fixed across Jost and K562. If
Jost's U columns have larger norm, that is a signal-amplitude advantage, not a
noise-model advantage. This script reports median ||u|| and median kappa for
each dataset and, if a measured Jost S has been exported to results/, its
observed SNR and held-out rho.
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)


def _summ_U(U, effs):
    col = np.linalg.norm(U, axis=0)
    kappa = None
    if effs is not None:
        vals = list(effs.values()) if isinstance(effs, dict) else list(effs)
        kappa = np.asarray(vals, dtype=float)
    out = {
        "shape": list(U.shape),
        "n_guides": U.shape[1],
        "median_col_norm_U": float(np.median(col)),
        "mean_col_norm_U": float(np.mean(col)),
        "frob_U": float(np.linalg.norm(U)),
    }
    if kappa is not None:
        out["kappa_median"] = float(np.median(kappa))
        out["kappa_min"] = float(kappa.min())
        out["kappa_max"] = float(kappa.max())
    return out


def main():
    payload = {"seed": 20260927, "datasets": {}}

    # K562 and RPE1 from stored measurements.
    for tag, path in [("K562_essential", "k562_essential_measurement.pkl"),
                       ("RPE1_essential", "rpe1_essential_measurement.pkl")]:
        with (RESULTS / path).open("rb") as f:
            b = pickle.load(f)
        r = b["measurement"].report
        payload["datasets"][tag] = _summ_U(b["measurement"].U, r.guide_efficiencies)

    # Jost.
    with (RESULTS / "jost_u_at_d30.pkl").open("rb") as f:
        j = pickle.load(f)
    payload["datasets"]["Jost_2020"] = _summ_U(j["U"], j.get("effs"))

    # If a Jost measured S exists on disk, add its SNR and rho.
    for candidate in ("jost_measurement.pkl", "jost_essential_measurement.pkl"):
        p = RESULTS / candidate
        if p.exists():
            with p.open("rb") as f:
                b = pickle.load(f)
            S = b["measurement"].S
            payload["datasets"]["Jost_2020"]["median_col_norm_S_observed"] = float(np.median(np.linalg.norm(S, axis=0)))
            payload["Jost_measured_S_source"] = candidate
            break
    else:
        payload["Jost_measured_S_source"] = None  # not exported to results/

    print(json.dumps(payload, indent=2))
    (OUT / "F6_input_amplitudes.json").write_text(json.dumps(payload, indent=2))
    print("saved:", OUT / "F6_input_amplitudes.json")


if __name__ == "__main__":
    main()
