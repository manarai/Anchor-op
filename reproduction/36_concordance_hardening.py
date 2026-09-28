"""Descriptive-concordance hardening (reviewer request).

The current backed_exploratory_response_analysis.py reports **within-target**
mean pairwise cosine for target-response directions. Per the reviewer, that
is not evidence on its own because:

  1. There is no between-target null. In a dataset where responses are
     dominated by a shared mode (see F4), unrelated targets look concordant
     for reasons that have nothing to do with target-specific biology.
  2. There is no noise ceiling. A within-guide split-half cosine — split
     each guide's cells randomly in half, compute two responses, take
     their cosine — gives the maximum concordance any real measurement
     could achieve.
  3. Replogle essential is the wrong primary dataset for this diagnostic:
     it aggregates to one guide per target, so target-level pairwise
     cosines are undefined for most targets. Jost 2020 is the primary
     dataset for within-target replicate concordance (3–6 sgRNAs per
     target).

This script provides the between-target null, using guide-response tables
that backed_exploratory produces (or that a similar per-guide fit
produces). Split-half is not computed here because it needs cell-level
access to the h5ad; the required extension to backed_exploratory is
described at the bottom of this file.

Usage:
    python reproduction/36_concordance_hardening.py \
        --guide-responses-csv path/to/<stem>_guide_responses.csv \
        --output results/recheck/F_concordance_null.json

If no CSV is supplied, the script prints the guide-count table for the
in-repo Jost pickle (per-sgRNA U columns are available, but S is not, so
target-level "responses" from Jost are not computable from what is
tracked in results/).
"""
from __future__ import annotations
import argparse, json, pickle, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]


def _cosine(a, b):
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an == 0 or bn == 0:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def between_target_null_from_csv(csv_path: Path, seed: int = 20260927) -> dict:
    import pandas as pd

    tbl = pd.read_csv(csv_path)
    pc_cols = [c for c in tbl.columns if c.startswith("pc_")]
    if not pc_cols:
        raise SystemExit("no pc_* columns in the guide-responses CSV.")
    vectors = tbl[pc_cols].to_numpy(dtype=float)
    targets = tbl["target"].astype(str).to_numpy()
    guides = tbl["guide"].astype(str).to_numpy() if "guide" in tbl.columns else np.arange(len(tbl)).astype(str)

    # Within-target pairwise cosines (guide, guide) where target_left == target_right.
    within, between = [], []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            c = _cosine(vectors[i], vectors[j])
            (within if targets[i] == targets[j] else between).append(c)
    within = np.asarray(within, dtype=float)
    between = np.asarray(between, dtype=float)

    # Shared-mode removed.
    v_mean = vectors.mean(axis=0)
    v_mean /= max(np.linalg.norm(v_mean), 1e-30)
    P = np.eye(vectors.shape[1]) - np.outer(v_mean, v_mean)
    vec_dm = vectors @ P.T
    within_dm, between_dm = [], []
    for i in range(len(vec_dm)):
        for j in range(i + 1, len(vec_dm)):
            c = _cosine(vec_dm[i], vec_dm[j])
            (within_dm if targets[i] == targets[j] else between_dm).append(c)
    within_dm = np.asarray(within_dm, dtype=float)
    between_dm = np.asarray(between_dm, dtype=float)

    return {
        "source": str(csv_path),
        "n_guides": int(len(vectors)),
        "n_targets": int(len(np.unique(targets))),
        "n_within_pairs": int(within.size),
        "n_between_pairs": int(between.size),
        "raw": {
            "within_mean": float(np.nanmean(within)) if within.size else None,
            "within_median": float(np.nanmedian(within)) if within.size else None,
            "between_mean": float(np.nanmean(between)) if between.size else None,
            "between_median": float(np.nanmedian(between)) if between.size else None,
            "difference_mean": float(np.nanmean(within) - np.nanmean(between)) if within.size and between.size else None,
        },
        "shared_mode_removed": {
            "within_mean": float(np.nanmean(within_dm)) if within_dm.size else None,
            "within_median": float(np.nanmedian(within_dm)) if within_dm.size else None,
            "between_mean": float(np.nanmean(between_dm)) if between_dm.size else None,
            "between_median": float(np.nanmedian(between_dm)) if between_dm.size else None,
            "difference_mean": float(np.nanmean(within_dm) - np.nanmean(between_dm)) if within_dm.size and between_dm.size else None,
        },
    }


def replogle_guide_count_table():
    out = {}
    for tag, pkl in [("K562_essential", "k562_essential_measurement.pkl"),
                      ("RPE1_essential", "rpe1_essential_measurement.pkl")]:
        with (REPO / "results" / pkl).open("rb") as f:
            b = pickle.load(f)
        rep = b["measurement"].report
        gt = getattr(rep, "guide_targets", None) or {n: n.replace("guide_", "") for n in rep.retained_guides}
        from collections import Counter
        c = Counter(gt.values())
        counts = np.asarray(list(c.values()))
        out[tag] = {
            "n_targets": int(counts.size),
            "n_targets_with_at_least_2_guides": int((counts >= 2).sum()),
            "n_targets_with_at_least_3_guides": int((counts >= 3).sum()),
            "n_guides_per_target_median": int(np.median(counts)),
            "note": ("Replogle essential is aggregated to target-level (one 'guide' per "
                     "target) in this pipeline. Within-target replicate concordance is "
                     "therefore not measurable on this dataset without disabling the "
                     "target-aggregation step in examples/01b_measure_k562_replogle.ipynb.")
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide-responses-csv", type=Path,
                    help="path to a backed_exploratory *_guide_responses.csv")
    ap.add_argument("--output", type=Path,
                    default=REPO / "results" / "recheck" / "F_concordance_null.json")
    args = ap.parse_args()

    result = {"seed": 20260927, "replogle_at_least_2_guides": replogle_guide_count_table()}
    if args.guide_responses_csv:
        result["between_target_null"] = between_target_null_from_csv(args.guide_responses_csv)
    else:
        result["between_target_null_note"] = (
            "No guide-response CSV supplied. Run "
            "backed_exploratory_response_analysis.py on the Jost 2020 h5ad first, "
            "then re-run this script with --guide-responses-csv.")

    result["split_half_ceiling_status"] = (
        "Not computed — blocker for the restructure sign-off. Requires "
        "extending backed_exploratory_response_analysis.py to (i) randomly "
        "partition each retained guide's cells in half, (ii) fit each half "
        "separately in the same basis, (iii) emit <stem>_split_half.csv with "
        "per-guide (guide, cos_split, n_left, n_right). Change is ~30 LoC "
        "around control_sample_and_mean and stream_group_statistics. What "
        "this diagnostic separates: within-target direction concordance is "
        "an additive-model check — under any additive-linear J·P_X model, "
        "all guides for one target move the system along the same direction "
        "(κ rescales, W^T delta_g is fixed), so it separates (b) "
        "dose-dependent non-linearity from {a, c}, not (a) from (c). "
        "See MANUSCRIPT_RESTRUCTURE_PLAN.md §1 for the full triage and "
        "the run order that pulls apart (a), (b), and (c).")
    result["primary_dataset_note"] = (
        "Jost 2020 (GSE132080) is the primary dataset for within-target guide-"
        "replicate concordance: 25 targets, 3–6 sgRNAs each. Replogle essential "
        "in this pipeline is aggregated to one guide per target, so it is not "
        "usable for this diagnostic without turning aggregation off.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("\nsaved:", args.output)


if __name__ == "__main__":
    main()
