# anchor-op

> **A Perturb-seq response is a measured sensitivity; the corresponding Jacobian action is a regularized, explicitly partial inverse.**

`anchor-op` is a Python package for estimating and benchmarking **identifiable local dynamical actions** from pooled CRISPRi/CRISPRa single-cell RNA-seq experiments with matched non-targeting controls. It projects expression into a low-dimensional control-derived program space, estimates guide-level settled responses, encodes each perturbation through its program loading and observed knockdown efficiency, and returns a matrix action together with a non-optional identifiability report.

The package is designed around genome-scale Perturb-seq resources such as the K562 and RPE1 experiments published by Replogle *et al.*, whose public portal provides processed single-cell and pseudobulk AnnData data.[1] It supports control-only NMF / seed-ensemble cNMF-style program construction, consistent with the use of matrix factorization to resolve expression programs and their cellular usages.[2]

## What is measured

For program coordinates `z = Wᵀe`, a settled guide perturbation is modeled as a constant input shift:

```text
Δz_g = -J⁻¹u_g,      S = -J⁻¹U,      J S = -U.
```

The original proposal used `J = -U S⁺`. The implementation makes the crucial partial-identification condition explicit. With `P_X = SS⁺`, the experiment identifies

```text
J P_X = -U S⁺,     X = range(S).
```

The package therefore distinguishes the **actuated input subspace** `range(U)` from the **identified response/domain subspace** `range(S)`. A zero extension outside `X` is not silently called a full cellular Jacobian. `MeasuredOperator.J` is available only when the effective response rank equals the full program dimension; otherwise the user works with `MeasuredOperator.identified_action` and projected comparisons. This prevents unsupported eigenvalue or hyperbolicity claims under incomplete excitation.

| Returned object | Interpretation | Always available? |
|---|---|---:|
| `S` | Guide-level steady-state response matrix | Yes |
| `U` | Guide-level perturbation input encoding | Yes |
| `identified_action` | `J P_X = -U S⁺`, conditional on the chosen regularizer | Yes, with report |
| `report` | Rank, projectors, singular spectrum, conditioning, guide filters, regularization path, and optional guide-bootstrap uncertainty | Yes |
| `J` | Full operator | Only when full effective-domain rank is identified |

## Installation

**Requires** [conda](https://docs.conda.io/en/latest/miniconda.html) (or [mamba](https://mamba.readthedocs.io/), a faster drop-in replacement). Install anchor-op in three commands from the repo root:

```bash
conda env create -f environment.yml    # or: mamba env create -f environment.yml
conda activate anchor-op
python -m pip install -e '.[test]'     # editable install of anchor-op itself
```

Verify the install:

```bash
pytest -q                              # expected: 33 passed
```

The `environment.yml` at the repo root pins Python 3.11 and all runtime dependencies (NumPy, pandas, AnnData, scikit-learn, scanpy, matplotlib, Jupyter, nbconvert, pytest). anchor-op itself is installed with `pip install -e` so source edits are picked up without an env rebuild.

**Why conda:** anchor-op is designed for Perturb-seq workflows, which routinely require single-cell data infrastructure (scanpy, anndata) that is more reliable to install via conda than pip. The conda environment is the tested, reproducible baseline for every notebook and figure in this repository.

### Updating an existing environment

If you have an anchor-op environment from an earlier commit and want to pick up new dependencies without rebuilding:

```bash
conda activate anchor-op
conda env update -f environment.yml    # sync to the latest environment.yml
```

### Advanced: pip-only install (developer note)

The core package (`import anchorop`) requires only NumPy and pandas at runtime. `anndata`, scikit-learn, matplotlib, scanpy, and Jupyter are optional in the sense that not every import path needs them; the plotting API (`ao.plotting`, `ao.analyses`) imports matplotlib lazily and raises a friendly `ImportError` if it is missing. A pip-only install (`pip install -e '.[test]'`) will run the test suite but cannot execute the real-data notebooks. **This path is not the supported installation**; use conda.

For real data, use a pinned environment and record upstream dataset versions, program-basis seed ensemble, guide filtering, and regularization path.

## Minimal workflow

```python
import anchorop as ao

# Fit the coordinate system on non-targeting controls only.
basis = ao.fit_programs(
    adata,
    d=50,
    method="cnmf",
    control_mask=adata.obs["is_control"].to_numpy(),
    n_seeds=10,
)

# Each guide needs an explicit target-gene annotation; targets are never parsed
# from an arbitrary guide identifier. The `efficiency_estimator` default is
# `"detection_rate"` — robust to scRNA-seq dropout at low-baseline targets
# (pass `"mean_ratio"` for the legacy `1 - mean_pert / mean_ctrl` estimator).
measurement = ao.measure_operator(
    adata,
    basis,
    guide_key="guide",
    target_key="target_gene",
    control_label="non-targeting",
    batch_key="batch",             # optional but recommended when applicable
    reg="tsvd",
    reg_param="path",
    rank_tol=1e-2,                 # preregistered identifiability cutoff
    bootstrap=500,
    efficiency_estimator="detection_rate",  # the current default
)

print(measurement.report.to_dict())
print(measurement.identified_action)  # The measured action J P_X.

# Comparison is always projected to the identified response domain.
results = ao.compare(
    measurement,
    {"celloracle": J_celloracle, "dynamo": J_dynamo, "scjdo": J_scjdo},
    nulls=("shuffled_edges", "random_init"),
    n_null=500,
)
print(ao.comparison_table(results))

# Null-corrected linearity check — compares weak vs strong efficiency bins,
# and (when n_null > 0) subtracts the random-split bin-composition floor so
# only real dose-response / model-mismatch signal contributes.
linearity = ao.linearity_check(measurement, threshold=0.25, n_null=200, null_seed=42)
print({
    "passed": linearity.passed,                    # excess_above_null <= 0.25
    "relative_difference": linearity.relative_difference,
    "null_median": linearity.null_median,           # bin-composition floor
    "excess_above_null": linearity.excess_above_null,
    "z_score": linearity.z_score,
})

# Full spectra are intentionally unavailable for partial measurements.
if measurement.report.full_domain_identified:
    print(measurement.J)
```

The public low-level API `measure_from_sensitivity(S, U, ...)` is useful for synthetic recovery tests and reproducible upstream pipelines. It returns exactly the same report-bearing object; there is no public unchecked inverse path.

## Reports and plotting

Two optional modules turn any measurement, comparison, or archetype fit into a standard set of figures and JSON/CSV summaries. Matplotlib is imported lazily.

### `ao.analyses` — one-call workflows that produce every standard figure and (optionally) save them to disk:

```python
# Full measurement diagnostic in one call.
report = ao.analyses.measurement_report(measurement)
report["figures"]["diagnostics"]        # 3-panel figure: spectrum, J heatmap, eigenvalues
report["figures"]["guide_drops"]        # pareto bar of drop reasons
report["summary"]                       # JSON-friendly report of the AnchorReport

# Save the whole bundle for supplementary materials.
ao.analyses.benchmark_report(
    measurement,
    {"celloracle": J_celloracle, "dynamo": J_dynamo},
    n_null=1000,
    save_dir="results/k562_benchmark",   # writes PNGs + metrics.csv
)

# Archetype fitting with all three standard plots (CV curve, weights, matrices).
ao.analyses.archetype_report(
    [k562_measurement, rpe1_measurement],
    mode="operator", k="cv", save_dir="results/archetypes",
)

# Efficiency-estimator comparison — the mean_ratio vs detection_rate diagnostic
# that motivated the current default (see 05_linearity_diagnostics.ipynb).
ao.analyses.efficiency_comparison_report(
    expression=adata.X, var_names=adata.var_names,
    control_mask=nt_mask,
    guide_to_target={"guide_A": "MALAT1", ...},
    guide_to_cells={"guide_A": pert_mask, ...},
    save_dir="results/eff_comparison",
)
```

### `ao.plotting` — individual `Figure`-returning functions, each with an optional `ax=` for embedding in your own layouts:

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ao.plotting.plot_singular_spectrum(measurement, ax=axes[0])
ao.plotting.plot_operator_heatmap(measurement, ax=axes[1])
```

Available: `plot_singular_spectrum`, `plot_operator_heatmap`, `plot_eigenvalue_plane`, `plot_guide_drop_reasons`, `plot_measurement_diagnostics`, `plot_benchmark_bars`, `plot_sym_antisym_bars`, `plot_archetype_cv_curve`, `plot_simplex_weights_heatmap`, `plot_archetype_matrices`, `plot_efficiency_comparison`, `save_figures`.

## Loading Replogle 2022 data

The Figshare Plus deposit (`gwps.wi.mit.edu`, ID `20029387`) distributes processed h5ads whose obs schema varies across releases. `ao.load_replogle_h5ad` auto-detects the target / guide / batch columns and NT label, then returns an AnnData whose obs has canonical `guide` / `target_gene` columns ready for `measure_operator`:

```python
adata = ao.load_replogle_h5ad(
    "data/K562_essential_normalized_singlecell_01.h5ad",
    # target_col=..., guide_col=..., control_label=...  # override auto-detection if needed
)
# adata.uns["anchorop_replogle_provenance"] records exactly which columns/label were detected.
measurement = ao.measure_operator(
    adata, basis, guide_key="guide", target_key="target_gene",
    control_label="non-targeting",
)
```

See `examples/01b_measure_k562_replogle.ipynb` for the full protocol.

## Scientific safeguards

The code treats the following conditions as analyses to disclose rather than implementation details to hide.

| Safeguard | Package behavior | Interpretation |
|---|---|---|
| Matched controls | Refuses data with no non-targeting controls; optionally matches response baselines by batch | Prevents an unqualified perturbation/control contrast |
| Program coordinate contamination | Fits built-in bases on controls only | Avoids defining coordinates with perturbation-driven programs |
| Perturbation encoding | Uses `u_g = -κ_g Wᵀδ_g` and drops negligible-loadings | A target gene is not a one-hot program direction |
| Guide efficacy | Estimates `κ_g` from the target transcript. Default `efficiency_estimator="detection_rate"` uses the drop in target-transcript detection rate, robust to scRNA-seq dropout at low-baseline targets; the legacy `"mean_ratio"` is available for backward compatibility | Filters weak/noninformative knockdowns rather than accepting dropout-driven pseudo-perfect efficiencies |
| Partial identification | Stores both input and response projectors and blocks full spectra when rank is incomplete | Keeps claims within the experimentally identified domain |
| Ill-conditioning | Records a full TSVD or Tikhonov path, retained singular directions, and condition number | Makes dependence on regularization inspectable |
| Uncertainty | Bootstraps guides, not only cells | Reflects guide-level heterogeneity |
| Linearity | Compares weak and strong knockdown-efficiency bins | Surfaces violation of the local settled-state approximation |
| Null calibration | Supports shuffled-edge and random-initialization operator nulls | Separates agreement from chance-level resemblance |
| Archetypes | Fits symmetric operator archetypes with simplex weights and guide-held-out selection of `k` | Avoids treating an arbitrary basis as a biological alphabet |

## Current implementation status

This repository is an **implementation baseline**, not a completed biological benchmark. It contains synthetic recovery and safety tests; it does not claim a K562/RPE1 result, a reference `dim(M)`, or a Phase 2 conclusion before public data are run end-to-end. The Replogle data portal provides processed AnnData resources and a documented route to raw or matrix-formatted data.[3]

| Component | Status |
|---|---|
| Control-only program fitting and external basis validation | Implemented |
| Guide-level `S`, `κ_g`, `U`, TSVD/Tikhonov inverse, report, and guide bootstrap | Implemented |
| `efficiency_estimator="detection_rate"` default + `mean_ratio` legacy option | Implemented (default since the K562 diagnostic in `05_linearity_diagnostics.ipynb`) |
| `rank_tol` (identifiability cutoff) preregistered at `1e-2` | Implemented |
| Projected inferred-versus-measured comparison with declared nulls | Implemented |
| Spectral abscissa difference (non-normal-robust alternative to Wasserstein) | Implemented |
| Spectral and symmetric-operator archetypes; transfer test | Implemented |
| `ao.plotting` module (12 plot functions, lazy matplotlib) | Implemented |
| `ao.analyses` module (`measurement_report`, `benchmark_report`, `archetype_report`, `efficiency_comparison_report`) | Implemented |
| `ao.load_replogle_h5ad` (auto-detecting Replogle 2022 schema loader) | Implemented |
| Public K562/RPE1 essential-gene benchmark on Replogle data | **Executed**: K562 essential (188/200 guides, full rank d=30, cond 65.0) and RPE1 essential (153/200 guides, full rank d=30, cond 65.3). See `MANUSCRIPT.md` and `01b`/`01c` notebooks. |
| Null-corrected linearity check (`n_null=` on `ao.linearity_check`) | Implemented: random-split null distribution isolates real dose-response signal from bin-composition floor. Both essential-gene measurements pass under this criterion (excess above null: +0.11 K562 z=3.3, +0.19 RPE1 z=5.5, both below preregistered 0.25 threshold). |
| Shared `scjdo.operator` metric substrate | Release blocker: sibling repository was not available in this workspace |
| Phase 4 constrained anchored inference | Deliberately gated on preregistered Phase 2 above-null evidence |

Read [`STRATEGY_REVIEW.md`](STRATEGY_REVIEW.md) for the detailed mathematical correction, experimental risk register, and release gates. Read [`PREREGISTRATION.md`](PREREGISTRATION.md) before any Phase 2 benchmark is executed. Read [`SPEC.md`](SPEC.md) §"Modeling assumption caveat — additive-input vs. intervention" for the documented systematic bias direction that `anchor-op` (as an additive-input tool) has against intervention-like CRISPRi biology, and for the reason a program-space intervention alternative is not shipped.

## Example notebooks

Every notebook opens with a `> **STATUS —**` banner identifying it as real data, demo, synthetic, or diagnostic.

| Notebook | Status | Purpose |
|---|---|---|
| `01_measure_k562.ipynb` | Real data (partial identification on this dataset) | Full measurement pipeline on raw 10x + CRISPR analysis tarballs |
| `01b_measure_k562_replogle.ipynb` | Real data (executed) | Replogle 2022 K562 essential-gene screen — 188/200 guides retained, full rank 30/30, cond 65.0, null-corrected linearity passes (excess +0.11, z=3.3) |
| `01c_measure_rpe1_replogle.ipynb` | Real data (executed) | Replogle 2022 RPE1 essential-gene screen — 153/200 guides retained, full rank 30/30, cond 65.3, null-corrected linearity passes (excess +0.19, z=5.5) |
| `02_benchmark.ipynb` | Infrastructure real, benchmark pending third-party operators | Preregistered inferred-vs-measured comparison + coordinate transforms |
| `03_archetypes.ipynb` | Demo (bootstrap when only one state is available) | Operator archetypes and cross-state transfer test |
| `04_synthetic_walkthrough.ipynb` | 100% synthetic | Full pipeline demonstration with known ground truth `J` |
| `05_linearity_diagnostics.ipynb` | Real data diagnostic (motivated the `detection_rate` default) | Efficiency estimator comparison + linearity failure analysis |

## Manuscript

A draft methods paper is at [`MANUSCRIPT.md`](MANUSCRIPT.md) with figures in `manuscript_figures/`. Mathematical derivations that the paper's methods section assumes (framework, identifiability partition, `rank_tol` justification, additive-vs-intervention full derivation, program-space intervention under-identification proof, null-corrected linearity, archetype geometry) are in [`MATH.md`](MATH.md) — every non-trivial equation is numerically validated. Written as a short methods-paper draft suitable for Bioinformatics Application Note / JOSS-style venue (~3200 words, 14 figures across 4 numbered figure groups + 1 table). The manuscript's numeric claims are all reproducible from the executed notebooks and the 33-test suite. Author list, affiliations, journal-specific formatting, and third-party benchmarks are the remaining pieces before submission.

Headline result from the manuscript: on the Replogle 2022 essential-gene screens for both K562 and RPE1, anchor-op yields full-rank operator identification (30/30, condition ~65) and both cell lines **pass the null-corrected linearity check**. The observed weak-vs-strong bin disagreement (rel_diff ~1.5) is 88-90% bin-composition floor; only the excess above the random-split null (+0.11 K562 z=3.3, +0.19 RPE1 z=5.5) is real signal, and it is below the preregistered 0.25 threshold on both cell lines. This reframes what has been read in the literature as widespread linearity failure as a diagnostic artifact rather than a physical failure of dose-response linearity.

## Scope and non-goals

`anchor-op` refuses program spaces above `d = 200` and warns above `d = 100`. It does not perform gene regulatory network inference, trajectory inference, or drift-field training. It consumes imported matrices from tools such as CellOracle, dynamo, or scJDO only after they have been transformed into the same labeled program coordinates.

The package cannot establish that a post-perturbation snapshot is at a common steady state. It reports evidence for or against a local linear approximation using the supplied cells and guides. A successful analysis must state the timepoint, guide filtering, control definition, response rank, regularization path, and linearity result.

## Development

Assumes the conda environment from [Installation](#installation) is active. Then:

```bash
pytest -q
```

The suite (currently **33 tests**) includes `test_ACCEPTANCE_` synthetic recovery, partial-identification orientation, regularization, safety, null-calibration, archetype, efficiency-estimator, null-corrected linearity, and analyses-API tests. CI runs the suite on supported Python versions. Tests that exercise plotting are marked `pytest.mark.skipif(not matplotlib installed)` so the core suite runs without a display or matplotlib when a caller has opted into the pip-only path.

## License

MIT. See [`LICENSE`](LICENSE).

## References

[1]: https://pubmed.ncbi.nlm.nih.gov/35688146/ "Replogle et al. (2022), Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq"
[2]: https://elifesciences.org/articles/43803 "Kotliar et al. (2019), Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq"
[3]: https://gwps.wi.mit.edu/ "Genome-Wide Perturb-Seq data access"
