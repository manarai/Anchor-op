# Reproduction of manuscript figures and analyses

Every figure, table, and numeric claim in `../MANUSCRIPT.md` (v0.3.2) is reproduced by scripts in this directory. Every script has a top-of-file docstring describing what it computes; the tables below are the entry-point map.

## Quick start

```bash
cd /path/to/anchor-op-source
mamba env create -f environment.yml && mamba activate anchor-op
pip install -e .
PYTHONPATH=src python3 reproduction/run_all.sh    # runs everything, skips scripts whose data deps are missing
```

Output PNGs land in `../manuscript_figures/`. Recheck JSONs land in `../results/recheck/`. Every script prints its output paths and any numeric summary it computes. All scripts fix `SEED` at the top and are deterministic.

## Manuscript figures and tables → script map (v0.3.2)

The current preprint has 6 main-text figures + 2 supplementary figures, all rendered by `50_build_figures.py` from JSONs written by the recheck / comparator scripts. This map is authoritative for what the paper actually shows.

| Manuscript object | Rendered by | Data JSON(s) sourced | Upstream script (writes JSON) |
|---|---|---|---|
| Figure 1 (pipeline schematic + matched-SNR control cartoon) | `50_build_figures.py::fig1` | none — self-contained matplotlib | — |
| Figure 2 (target-held-out nested-CV ρ on all three screens) | `50_build_figures.py::fig2` | `results/recheck/F_nested_cv_rho.json` | `40_nested_cv_rho.py` |
| Figure 3 ((a) α_S sweep + (b) interaction-only cosine) | `50_build_figures.py::fig3` | `F1_F4_summary.json`, `F_interaction_only_cosine.json` | `30_recheck_F1_F4.py`, `43_interaction_only_cosine.py` |
| Figure 4 ((a) K562 random-panel distribution, (b) Jost d-sweep, (c) direction-only) | `50_build_figures.py::fig4` | `F_step3_random_panels.json`, `F_jost_d_sweep_grouped.json`, `F_nested_cv_direction_only.json` | `53_random_panel_distribution.py`, `41_jost_d_sweep_grouped.py`, `44_nested_cv_direction_only.py` |
| Figure 5 (footprint encoding on the four earlier fits, with two nulls) | `50_build_figures.py::fig5` | `F_footprint.json` | `45_footprint_encoding.py` |
| Figure 6 (comparator panel — fixed / footprint / learned, forward + inverse) | `50_build_figures.py::fig6` | `F_step1_comparator_panel.json` | `51_comparator_panel.py` |
| Supp Fig S1 ((left) Jost dose interpolation, (right) rel_diff calibration) | `50_build_figures.py::fig_supp` | `F3_Jost_grouped_folds.json` + inline values | `39_recheck_jost_grouped_folds.py` (dose panel); rel_diff values from `30_recheck_F1_F4.py` |
| Supp Fig S2 (positive-control ensembles at matched SNR) | `50_build_figures.py::fig_supp_ensembles` | `F_step2_ensembles.json` | `52_positive_control_ensembles.py` |
| Table 1 (target-held-out ρ) | text-rendered | `F_nested_cv_rho.json` | `40_nested_cv_rho.py` |
| Table 2 (Jost d-sweep) | text-rendered | `F_jost_d_sweep_grouped.json` | `41_jost_d_sweep_grouped.py` |
| Table 3 (forward comparator panel) | text-rendered | `F_step1_comparator_panel.json` | `51_comparator_panel.py` |
| Table 4 (claims vs scope) | text-rendered | — (narrative) | — |

To rebuild every figure from the JSONs already checked into `results/recheck/`:

```bash
PYTHONPATH=src python3 reproduction/50_build_figures.py
```

## Recheck / audit-response scripts (v0.2.x → v0.3.2)

These are the scripts that wrote the JSONs feeding the manuscript. They are numbered in the order they were added to the audit trail.

| Script | Writes | Reads | Approx. runtime |
|---|---|---|---:|
| `30_recheck_F1_F4.py` | `F1_{K562,RPE1}_essential.json`, `F1_F4_summary.json`, `F2_*.json`, `F3_*.json`, `F4_*.json` | Replogle pkls in `results/` | ~10 min |
| `33_recheck_F5_jost_rank.py` | `F5_jost_gate.json`, `F5_jost_note.json` | `results/jost_measurement.pkl` | <1 min |
| `34_recheck_F6_jost_amplitude.py` | `F6_input_amplitudes.json` | Replogle + Jost pkls | ~1 min |
| `35_recheck_F7_selection_sensitivity.py` | (superseded by `42`) | K562 h5ad | ~30 min |
| `36_concordance_hardening.py` | `F_concordance_null.json` | Replogle pkls | <1 min |
| `37_recheck_F3_jost_end_to_end.py` | `results/jost_measurement.pkl` | Jost 2020 GSE132080 raw counts | ~15 min |
| `38_direction_only_rho.py` | `F_direction_only_rho.json` | Replogle + Jost pkls | ~2 min |
| `39_recheck_jost_grouped_folds.py` | `F3_Jost_grouped_folds.json`, `F3_Jost_pca_controls.json` | `results/jost_measurement.pkl` | ~3 min |
| `40_nested_cv_rho.py` | `F_nested_cv_rho.json` | Replogle + Jost pkls | ~3 min |
| `41_jost_d_sweep_grouped.py` | `F_jost_d_sweep_grouped.json` | `results/jost_measurement.pkl` | ~2 min |
| `42_f7_k562_random200.py` | `F7_K562_random200.json`, `results/k562_random200_*.pkl` | K562 h5ad (10 GB) | ~25 min |
| `43_interaction_only_cosine.py` | `F_interaction_only_cosine.json` | Replogle pkls | ~5 min |
| `44_nested_cv_direction_only.py` | `F_nested_cv_direction_only.json` | Replogle + Jost pkls | ~3 min |
| `45_footprint_encoding.py` | `F_footprint.json`, `results/*_sigma_ctrl.npz` | Replogle + Jost pkls + h5ads | ~10 min |
| `51_comparator_panel.py` | `F_step1_comparator_panel.json` | Replogle + Jost pkls + h5ads (10 GB + 8 GB) | ~35 min |
| `52_positive_control_ensembles.py` | `F_step2_ensembles.json` | Replogle + Jost pkls | ~10 min |
| `53_random_panel_distribution.py` | `F_step3_random_panels.json`, `results/k562_all_targets_measurement.pkl` | K562 h5ad (10 GB) | ~30 min |
| `54_learned_encoding_fidelity.py` | `F_step5_learned_fidelity.json` | K562 h5ad (10 GB) | ~20 min |

## Pre-v0.3.0 exploratory scripts (00–26)

These scripts (`00_*.py` through `26_*.py`) are the earlier figure/analysis pipeline that predates the audit response and the comparator panel. They still work and are kept for continuity with prior versions of the manuscript; the current v0.3.2 preprint does NOT depend on them (all current figures are rebuilt from JSONs written by the 30+ scripts above via `50_build_figures.py`). Each has a top-of-file docstring describing what it produced.

| Range | Purpose |
|---|---|
| `00`–`04` | Original per-figure scripts (v0.1 layout: fig0 pipeline schematic, fig1 synth composite, fig2 K562 aggregate, fig3 K562 essential, fig4 RPE1 essential). |
| `05`–`13` | Supplementary sweeps and diagnostics from v0.1–v0.2 (rank_tol, estimators, additive-to-clamp, RPE1-vs-Jost, realscale positive control, κ range, rejection power, operator recovery). |
| `14`–`26` | Utility scripts (lasso recovery, per-direction recovery, empirical null, noise-model sensitivity, stability shift, σ bootstrap, composites, all-ensemble anchor nulls, Jost matched-geometry nulls). |

## Data dependencies

| Category | Files | Size | Where from |
|---|---|---:|---|
| Replogle 2022 K562 essential h5ad | `examples/data/K562_essential_normalized_singlecell_01.h5ad` | 10 GB | <https://gwps.wi.mit.edu>, Figshare Plus deposit 20029387 |
| Replogle 2022 RPE1 essential h5ad | `examples/data/rpe1_normalized_singlecell_01.h5ad` | 8 GB | Same source |
| Jost 2020 raw counts | `examples/data/jost2020/GSE132080_10X_{matrix,barcodes,genes}.tsv.gz` + `GSE132080_{cell_identities,sgRNA_barcode_sequences_and_phenotypes}.csv.gz` | ~500 MB | GEO GSE132080 |
| Local raw (only for `02_fig2_k562_aggregate.py`) | K562 84K noncoding-element aggregate 10x mtx + CRISPR analysis | | Set `DATA_ROOT` at top of script; not required for current figures |
| Measurement bundles (regenerated on first run) | `../results/{k562_essential,rpe1_essential,jost,k562_all_targets,k562_random200}_measurement.pkl`, `../results/*_sigma_ctrl.npz` | ~2–3 MB each | Written by scripts 03, 04, 37, 42, 45, 53 |

`run_all.sh` skips a script cleanly when its data dependency is missing and reports a `SKIPPED` summary at the end.

## Preregistration

Every analysis is labelled preregistered vs post hoc in `../MANUSCRIPT.md` §4.8 (Preregistered vs post hoc) and `../PREREGISTRATION_AMENDMENT.md`:

- `PREREGISTRATION.md` — the original preregistered decisions (`rank_tol`, TSVD, d = 30, bin-split `rel_diff ≤ 0.25`).
- `PREREGISTRATION_AMENDMENT.md` — the 2026-09-27 footprint amendment, the 2026-09-28 Steps 1–3 preregistration (comparator panel, positive-control ensembles, random-panel distribution), and the 2026-09-28 learned-encoding fidelity-check preregistration.

Post-hoc bug fixes and rerun logs (with SUPERSEDED numbers kept for the record) live in `../RECHECK_LOG.md`.

## Number provenance

Every numeric claim in `../MANUSCRIPT.md` that is not derived in-text has a row in `../results/recheck/number_provenance.csv` giving:

- the claim (short label),
- the reported value (round-form),
- the manuscript location (§ / table / figure),
- the source JSON in `../results/recheck/`,
- the reproduction script that produced the JSON,
- notes on rounding or provenance.

Round-form numbers cited in the abstract and Results are drawn from the un-rounded values in that table.

## Provenance and sanity checks

- All scripts are deterministic. Each fixes a `SEED` at the top and uses it consistently.
- Every script has a top-of-file docstring describing exactly what it computes.
- `50_build_figures.py` reads only from `results/recheck/*.json` and writes only to `manuscript_figures/*.png`, so a clean rebuild is `python3 reproduction/50_build_figures.py`.
- The fidelity check (`54_learned_encoding_fidelity.py`) is a Python port of `solve_y_axb` from Ahlmann-Eltze et al. 2025's linear-baseline code (`github.com/const-ae/linear_perturbation_prediction-Paper`); it verifies our learned-encoding implementation before the analyses that depend on it. See `../RECHECK_LOG.md` §"v0.3.1 — Learned-encoding fidelity check + one-table fix" for the pass/fail record.
- The test suite (`pytest -q` from repo root, 59 tests) covers the core anchor-op API; the reproduction scripts are not part of the pytest suite but are individually deterministic and re-runnable.
