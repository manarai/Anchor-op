"""Task 2b — log1p input check (preregistered in PREREG amendment 2).

Steps:

1. Load the Replogle 2022 K562 essential **raw** single-cell counts
   from ``examples/data/K562_essential_raw_singlecell_01.h5ad``.
2. Match barcodes with the measurement bundle used in A1 (same
   cells, guides, targets as the residual h5ad).
3. Normalise: ``sc.pp.normalize_total(target_sum=1e4)``, then
   ``sc.pp.log1p``. State the method in the output JSON.
4. Report counts: n cells by category (NT control / perturbed /
   retained after ≥60 cells per target), n targets matched to the
   measurement bundle, median counts per cell before and after
   normalisation.
5. Fit a `PCARep(dim=30)` on the log1p-normalised NT controls.
6. For every one of the 188 target genes, compute
   `knockdown_scale_difference(X_ctrl, g, kappa=0.7, input_space='log1p')`
   and record its L2 norm.
7. Assertion: ≥ 95 % (179 / 188) have norm > 1e-6. Halt otherwise.

Output: ``TASK2b_lognorm_input_check.json``.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = Path(__file__).parent / "TASK2b_lognorm_input_check.json"
RAW_H5AD = REPO / "examples" / "data" / "K562_essential_raw_singlecell_01.h5ad"
PKL = REPO / "results" / "k562_essential_measurement.pkl"
TARGET_SUM = 1e4
KAPPA_PRIMARY = 0.7
NORM_THRESHOLD = 1e-6
PASS_FRAC = 0.95


def main():
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp

    from anchorop.state_space import PCARep

    if not RAW_H5AD.exists():
        print(f"ERROR: raw-counts h5ad missing at {RAW_H5AD}. "
              "Finish the Figshare download first.")
        sys.exit(1)

    # 1. Load raw counts.
    print(f"[2b] loading {RAW_H5AD.name} …")
    a = ad.read_h5ad(RAW_H5AD, backed="r")
    print(f"[2b] raw: {a.n_obs} cells × {a.n_vars} genes")

    # 2. Match target set with the measurement bundle used in A1.
    bundle = pickle.load(open(PKL, "rb"))
    retained_guides = list(bundle["measurement"].report.retained_guides)
    stored = getattr(bundle["measurement"].report, "guide_targets", None)
    if stored:
        clean = {str(k): str(v) for k, v in dict(stored).items()}
        targets_measurement = sorted({clean[str(g)] for g in retained_guides})
    else:
        targets_measurement = sorted({
            str(g)[6:] if str(g).startswith("guide_") else str(g).split("_")[0]
            for g in retained_guides
        })
    print(f"[2b] measurement-bundle targets: {len(targets_measurement)}")

    var_names = list(a.var_names)
    gene_index = {g: i for i, g in enumerate(var_names)}
    matched_targets = [t for t in targets_measurement if t in gene_index]
    print(f"[2b] matched to raw-h5ad var_names: {len(matched_targets)} "
          f"/ {len(targets_measurement)}")

    # 3. Cell-level categories.
    obs = a.obs
    ctrl_mask = (obs["gene_id"] == "non-targeting").to_numpy()
    perturbed_mask = ~ctrl_mask
    n_ctrl = int(ctrl_mask.sum())
    n_perturbed = int(perturbed_mask.sum())
    print(f"[2b] NT control cells: {n_ctrl}, perturbed cells: {n_perturbed}")

    # Count per target.
    counts_per_target = obs["gene_id"].value_counts()
    qualifying = counts_per_target[counts_per_target >= 60].index.tolist()
    qualifying = [t for t in qualifying if t != "non-targeting"
                  and t in gene_index]
    print(f"[2b] qualifying essential targets (≥60 cells, in var_names): "
          f"{len(qualifying)}")

    # 4. Pull raw counts for NT controls into memory (dense).
    print(f"[2b] materialising raw counts for {n_ctrl} NT control cells …")
    ctrl_idx = np.where(ctrl_mask)[0]
    X_raw_ctrl = a.X[ctrl_idx]
    if sp.issparse(X_raw_ctrl):
        X_raw_ctrl = X_raw_ctrl.toarray()
    X_raw_ctrl = np.asarray(X_raw_ctrl, dtype=np.float64)
    median_counts_per_cell_raw = float(np.median(X_raw_ctrl.sum(axis=1)))
    print(f"[2b] median counts per NT control cell: "
          f"{median_counts_per_cell_raw:.1f}")

    # Normalise + log1p.
    print(f"[2b] normalise_total(target_sum={TARGET_SUM:.0e}) + log1p …")
    sums = X_raw_ctrl.sum(axis=1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    X_norm_ctrl = X_raw_ctrl * (TARGET_SUM / sums)
    X_log_ctrl = np.log1p(X_norm_ctrl)
    median_counts_per_cell_norm = float(np.median(X_log_ctrl.sum(axis=1)))

    # 5. Fit a PCARep(30) on the log1p-normalised NT controls.
    print(f"[2b] fitting PCARep(30) on log1p NT controls …")
    rep = PCARep(dim=30).fit(X_log_ctrl)

    # Report the per-gene control-cell mean distribution for the
    # matched target genes.
    cols = [gene_index[t] for t in matched_targets]
    mean_abs = np.abs(X_log_ctrl[:, cols].mean(axis=0))
    q = np.quantile(mean_abs, [0, 0.05, 0.5, 0.95, 1.0])
    print(f"[2b] |mean(X_log_ctrl[:, g])| over {len(cols)} target genes:")
    print(f"      min={q[0]:.4e}  p05={q[1]:.4e}  "
          f"median={q[2]:.4e}  p95={q[3]:.4e}  max={q[4]:.4e}")

    # 6. Knockdown-scale FD per target.
    print(f"[2b] knockdown-scale FD at κ = {KAPPA_PRIMARY} on all "
          f"{len(matched_targets)} target genes …")
    per_target = []
    for t in matched_targets:
        g = gene_index[t]
        u_z = rep.knockdown_scale_difference(
            X_log_ctrl, g, kappa=KAPPA_PRIMARY, input_space="log1p")
        per_target.append({"target": t, "gene_idx": int(g),
                           "u_z_norm": float(np.linalg.norm(u_z))})

    # 7. Assertion.
    norms = np.array([r["u_z_norm"] for r in per_target])
    n_nontrivial = int((norms > NORM_THRESHOLD).sum())
    frac = n_nontrivial / len(per_target)
    passed = bool(frac >= PASS_FRAC)
    print(f"[2b] non-trivial (‖u_z‖ > {NORM_THRESHOLD:.0e}): "
          f"{n_nontrivial} / {len(per_target)} ({100*frac:.1f} %)")
    print(f"[2b] VERDICT: {'PASS' if passed else 'FAIL'} "
          f"(threshold {100*PASS_FRAC:.0f} %)")

    report = {
        "raw_h5ad": str(RAW_H5AD.relative_to(REPO)),
        "n_cells_raw": int(a.n_obs),
        "n_genes_raw": int(a.n_vars),
        "n_nt_control": n_ctrl,
        "n_perturbed": n_perturbed,
        "n_measurement_bundle_targets": len(targets_measurement),
        "n_targets_matched_to_raw_var_names": len(matched_targets),
        "n_qualifying_targets_min_60_cells": len(qualifying),
        "normalisation": {
            "method": "scanpy.pp.normalize_total(target_sum=1e4) + scanpy.pp.log1p",
            "target_sum": TARGET_SUM,
            "median_raw_counts_per_control_cell": median_counts_per_cell_raw,
            "median_log1p_sum_per_control_cell": median_counts_per_cell_norm,
        },
        "log1p_control_mean_abs": {
            "min": float(q[0]), "p05": float(q[1]), "median": float(q[2]),
            "p95": float(q[3]), "max": float(q[4]),
        },
        "knockdown_scale": {
            "kappa_primary": KAPPA_PRIMARY,
            "input_space": "log1p",
            "n_targets": len(per_target),
            "n_norm_gt_threshold": n_nontrivial,
            "norm_threshold": NORM_THRESHOLD,
            "fraction_nontrivial": frac,
            "pass_fraction_threshold": PASS_FRAC,
            "passed": passed,
            "norm_summary": {
                "min": float(norms.min()), "p05": float(np.percentile(norms, 5)),
                "median": float(np.median(norms)),
                "p95": float(np.percentile(norms, 95)),
                "max": float(norms.max()),
            },
        },
        "per_target": per_target,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[2b] saved: {OUT}")

    if not passed:
        print("[2b] HALT: fewer than 95% of targets have a non-trivial "
              "knockdown-scale FD under log1p. Report before any fitting.")
        sys.exit(2)


if __name__ == "__main__":
    main()
