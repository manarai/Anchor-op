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
# `"auto"` — routes by data format: count-like data → `mean_ratio` (the
# sample-moment MLE, unbiased under both Poisson and Poisson-with-dropout);
# pre-scaled residual data (contains meaningful negatives, e.g. z-scored
# Replogle h5ads) → `detection_rate` (a signed distributional-shift statistic,
# analytically `0.5 − Φ(Δ/σ)`, valid on that data class).
# `min_control_detection_rate` (default 0.05) drops info-limited targets on
# count data where any estimator would be dominated by discretization noise;
# on pre-scaled data the filter is effectively inert.
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
    # efficiency_estimator="auto" (default) — pin explicitly to override
    # min_control_detection_rate=0.05 (default)
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

# Linearity diagnostic — compares weak vs strong efficiency bin operators.
# IMPORTANT: at published Perturb-seq scale (d≈30, n≈200, per-guide noise
# σ≈0.27) this diagnostic is noise-limited and the preregistered 0.25
# threshold is unreachable for any dataset. See MANUSCRIPT.md §3.5–§3.6 and
# `reproduction/10_figS10_realscale_positive_control.py` for the
# matched-scale positive-control methodology required to interpret observed
# rel_diff / ρ values against a linear-noise baseline before drawing any
# linearity conclusion.
linearity = ao.linearity_check(measurement, threshold=0.25, n_null=200, null_seed=42)
rho = ao.held_out_prediction_check(measurement, n_folds=5, seed=0,
                                    n_permutation_null=30, null_seed=100)
print({
    "rel_diff": linearity.relative_difference,
    "null_median": linearity.null_median,
    "held_out_rho": rho.rho_pooled,
    # Interpret these against a matched-scale linear positive control
    # (see tutorial/04_linearity_diagnostics_power_analysis.ipynb).
})

# Full spectra are intentionally unavailable for partial measurements.
if measurement.report.full_domain_identified:
    print(measurement.J)
```

The public low-level API `measure_from_sensitivity(S, U, ...)` is useful for synthetic recovery tests and reproducible upstream pipelines. It returns exactly the same report-bearing object; there is no public unchecked inverse path.

## API

Every function listed here is importable directly from the top-level `anchorop` package (`import anchorop as ao; ao.<name>`), except for the two submodules `ao.plotting` and `ao.analyses` which group figure-producing calls. All signatures below are the current public surface; docstrings in the source carry the full parameter documentation.

### Program basis — `ao.fit_programs`, `ao.make_program_basis`, `ao.project_expression`

| Function | Purpose |
|---|---|
| `fit_programs(adata, *, d, method="cnmf", control_mask, n_seeds=10, seed=0, ...)` | Fit a non-negative program basis (NMF / cNMF-approx) on control cells only. Returns a `ProgramBasis`. |
| `make_program_basis(loadings, gene_names, *, method="external", ...)` | Wrap externally-fit loadings (e.g. from SCENIC, dedicated cNMF, PCA) as a `ProgramBasis`. |
| `project_expression(expression, basis, *, gene_names=None)` | Project a cell-by-gene matrix into program coordinates `z = W^T e`. |

### Measurement — `ao.measure_operator`, `ao.measure_from_sensitivity`, `ao.build_guide_responses`, `ao.linearity_check`, `ao.estimate_knockdown_efficiency*`

| Function | Purpose |
|---|---|
| `measure_operator(adata, basis, *, guide_key, control_label, target_key=..., batch_key=..., rank_tol=1e-2, efficiency_estimator="auto", min_control_detection_rate=0.05, bootstrap=0, ...)` | Full pipeline: build guide responses, invert, return a `MeasuredOperator` with its inseparable `AnchorReport`. |
| `measure_from_sensitivity(S, U, *, guide_names=..., guide_efficiencies=..., reg="tsvd", reg_param="path", rank_tol=None, bootstrap=0, ...)` | Low-level inversion from pre-computed `S` and `U`. Same return contract; no unchecked inverse path exists. |
| `build_guide_responses(adata, basis, *, guide_key, control_label, efficiency_estimator="auto", min_control_detection_rate=0.05, ...)` | Guide-level Δz + input-encoding computation, without the inversion step. |
| `linearity_check(measurement, *, threshold=0.25, n_null=0, null_seed=0)` | Weak/strong-bin diagnostic on the common identified subspace. Optional `n_null > 0` draws a random-split null distribution. At published Perturb-seq scale this diagnostic is noise-limited — interpret against a matched-scale positive control per MANUSCRIPT.md §3.5–§3.6, not against the raw 0.25 threshold. |
| `held_out_prediction_check(measurement, *, n_folds=5, seed=0, n_permutation_null=0)` | Out-of-sample linearity diagnostic: fit `A = J·P_X` on train guides, evaluate `ρ = ‖A·S_test + U_test‖_F / ‖U_test‖_F` on held-out. Invariant to global U rescaling. Also noise-limited at published Perturb-seq scale — see MANUSCRIPT.md §3.5–§3.6 and Fig. S10 for the matched-scale positive-control methodology. |
| `estimate_knockdown_efficiency(expression, *, target_index, perturbed_mask, control_mask, ...)` | Current default: `1 − mean_pert / mean_ctrl`. Unbiased under both Poisson and Poisson-with-dropout; at very low baseline it becomes bimodal (0 or 1) — pair with `min_control_detection_rate` to filter those targets. |
| `estimate_knockdown_efficiency_poisson_mle(expression, *, target_index, perturbed_mask, control_mask, ...)` | Poisson MLE: `1 − λ̂_pert/λ̂_ctrl` with `λ̂ = −log(1 − detection_rate)`. Equivalent to `mean_ratio` under pure Poisson; biased low under independent zero-inflation. |
| `estimate_knockdown_efficiency_detection_rate(expression, *, target_index, perturbed_mask, control_mask, ...)` | Raw detection-shift `Pr[X_ctrl>0] − Pr[X_pert>0]`. NOT an unbiased estimator of `κ` — a bounded shift diagnostic that scales with baseline. Retained for backward compatibility; see `examples/06_estimator_simulation.ipynb`. |

### Regularization + identifiability — `ao.regularized_pseudoinverse`, `ao.regularization_path`

| Function | Purpose |
|---|---|
| `regularized_pseudoinverse(sensitivity, *, method="tsvd", parameter="path", rank_tol=None)` | Return `S⁺`, selected path entry, full path, and the response projector `P_X`. |
| `regularization_path(sensitivity, *, method, parameters=None, rank_tol=None)` | Full TSVD or Tikhonov path with per-entry rank, filter factors, and condition numbers. |

### Comparison / benchmarking — `ao.compare`, `ao.comparison_table`, `ao.spectral_*`, `ao.hyperbolicity_sign`

| Function | Purpose |
|---|---|
| `compare(measured, inferred_operators, *, nulls=("shuffled_edges", "random_init"), n_null=100, ...)` | Benchmark a dict of inferred `d × d` operators against the measured action on its identified subspace, with declared operator-level nulls. |
| `comparison_table(results)` | Flatten `compare()`'s dict of `ComparisonResult` into a `pandas.DataFrame`. |
| `spectral_wasserstein(A, B)` | Exact 2-Wasserstein distance between complex eigenvalue clouds. Unstable for non-normal `J`; use as supplementary. |
| `spectral_abscissa_difference(A, B)` | `|max Re(λ(A)) − max Re(λ(B))|`. Preferred hyperbolicity metric — Lipschitz-stable on diagonalizable operators. |
| `hyperbolicity_sign(operator, tolerance=1e-8)` | `+1 / 0 / −1` for max real eigenvalue's sign. |

### Archetypes — `ao.fit_archetypes`, `ao.transfer_test`, `ao.spectral_summary`

| Function | Purpose |
|---|---|
| `fit_archetypes(measurements, *, mode="operator", k="cv", max_k=8, n_splits=5, holdout_fraction=0.2, ...)` | Fit spectral or symmetric-operator archetypes with simplex-constrained weights. `k="cv"` uses guide-held-out equation residual to pick `k`. |
| `transfer_test(source, target, *, mode="operator", k="cv", ...)` | Fit archetypes on source states, project target states, compare with a target refit. |
| `spectral_summary(operator)` | Fixed-length real spectrum summary (sorted real + imaginary parts) for archetype fitting. |

### IO and coordinate handling — `ao.load_operator`, `ao.validate_operator`, `ao.load_replogle_h5ad`

| Function | Purpose |
|---|---|
| `load_operator(path, *, d=None, delimiter=",")` | Read an externally-inferred `d × d` operator from `.npy`, `.csv`, `.tsv`, or `.txt`. Validates finiteness and shape. |
| `validate_operator(operator, *, d=None, name="operator")` | Programmatic shape + finiteness check on any array intended for `compare()`. |
| `load_replogle_h5ad(path, *, target_col=None, guide_col=None, batch_col=None, control_label=None, ...)` | Load a Replogle 2022 processed h5ad with schema auto-detection (target/guide/batch column names, NT label, ENSG-vs-symbol var_names). Populates canonical `guide` / `target_gene` obs columns. |

### Report bundles — `ao.analyses` (all figure-producing workflow functions)

Each returns a dict with `figures` + result/summary objects; passing `save_dir=` writes PNGs + `summary.json` (or `metrics.csv`) to disk.

| Function | Purpose |
|---|---|
| `ao.analyses.measurement_report(measurement, save_dir=None)` | Full diagnostic bundle: 3-panel figure (spectrum + operator heatmap + eigenvalues) + guide-drop pareto + `AnchorReport` summary dict. |
| `ao.analyses.benchmark_report(measurement, inferred_operators, *, nulls=..., n_null=100, save_dir=None)` | Runs `compare()` and produces benchmark-bars + sym/antisym-bars figures + metrics table. |
| `ao.analyses.archetype_report(measurements, *, mode="operator", k="cv", save_dir=None, **fit_kwargs)` | Runs `fit_archetypes()` and produces CV curve + simplex weights + archetype matrices. |
| `ao.analyses.efficiency_comparison_report(expression, var_names, control_mask, guide_to_target, guide_to_cells, save_dir=None)` | Computes both `mean_ratio` and `detection_rate` efficiencies over the same target set as a diagnostic side-by-side comparison (independent of the `auto` router's per-dataset choice). |

### Individual plot functions — `ao.plotting` (each returns a `Figure`, accepts optional `ax=`)

| Function | Purpose |
|---|---|
| `plot_singular_spectrum(measurement, ax=None)` | Log-scale singular values of `S` with `rank_tol` cutoff overlay. |
| `plot_operator_heatmap(measurement, ax=None)` | Heatmap of `J` (or `J·P_X` at partial rank). |
| `plot_eigenvalue_plane(measurement, ax=None)` | Complex-plane scatter of `J` eigenvalues (blocked at partial rank). |
| `plot_guide_drop_reasons(measurement, ax=None)` | Pareto bar of drop reasons. |
| `plot_measurement_diagnostics(measurement)` | 3-panel combined: spectrum + operator + eigenvalues. |
| `plot_benchmark_bars(results, metrics=None)` | Per-method bars with null overlays for each requested metric. |
| `plot_sym_antisym_bars(results, ax=None)` | Preregistered sym-vs-antisym paired bars per method. |
| `plot_archetype_cv_curve(archetype_result, ax=None)` | CV error vs candidate `k` with selection marked. |
| `plot_simplex_weights_heatmap(archetype_result, ax=None)` | Per-state simplex weights over archetypes (rows sum to 1). |
| `plot_archetype_matrices(archetype_result)` | Side-by-side heatmaps of each archetype matrix. |
| `plot_efficiency_comparison(mean_ratio_values, detection_rate_values, ax=None)` | Two-panel `mean_ratio` vs `detection_rate` efficiency histograms. |
| `save_figures(figures, directory, dpi=140)` | Bulk save a dict of `Figure` objects to `directory/<name>.png`. |

### Types / result objects

| Type | Purpose |
|---|---|
| `MeasuredOperator` | Result of `measure_operator` / `measure_from_sensitivity`. Fields: `S`, `U`, `guide_names`, `identified_action`, `J` (property, raises unless full-rank), `report`, `response_projector`, `input_projector`. |
| `AnchorReport` | The mandatory identifiability report attached to every `MeasuredOperator`. Full JSON serialization via `.to_dict()`. |
| `ComparisonResult` | Per-method output from `compare()`: `metrics`, `null_metrics`, `metadata`. |
| `ArchetypeResult` | Result of `fit_archetypes()`: `mode`, `archetypes`, `weights`, `selected_k`, `candidate_errors`, `reconstruction_error`, `metadata`. |
| `LinearityResult` | Result of `linearity_check()`: `relative_difference`, `overlap_rank`, `passed`, `weak_guides`, `strong_guides`, plus (when `n_null > 0`) `null_median`, `null_std`, `null_p95`, `excess_above_null`, `z_score`, `n_null`. |
| `TransferResult` | Result of `transfer_test()`: `weights`, `transfer_error`, `refit_error`, `error_ratio`. |
| `ProgramBasis` | Frozen dataclass returned by `fit_programs` / `make_program_basis`. |
| `AnchorOpError`, `IdentifiabilityError`, `DimensionGuardError` | Exception hierarchy. `IdentifiabilityError` is what `MeasuredOperator.J` raises at partial rank. |

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
# that motivated the current `auto` router (see tutorial/02_efficiency_estimators.ipynb).
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
| Guide efficacy | Estimates `κ_g` from the target transcript. Default `efficiency_estimator="auto"` inspects the matrix: count-like data → `mean_ratio` (the sample-moment MLE, unbiased under Poisson and Poisson-with-dropout); pre-scaled residual data → `detection_rate` (a signed distributional-shift statistic, valid on that class). `min_control_detection_rate` (default 0.05) drops information-limited targets on count data before estimation, rather than choosing between two artifactual responses. | Selects a principled estimator for the data class and filters targets where any estimator would be dominated by discretization noise |
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
| `efficiency_estimator="auto"` (default) + `mean_ratio`, `poisson_mle`, `detection_rate` as explicit options; `min_control_detection_rate` filter | Implemented (§2.3 of MANUSCRIPT.md; simulation in `examples/06_estimator_simulation.ipynb`; Fig S2/S3 in `manuscript_figures/`) |
| `rank_tol` (identifiability cutoff) preregistered at `1e-2` | Implemented |
| Projected inferred-versus-measured comparison with declared nulls | Implemented |
| Spectral abscissa difference (non-normal-robust alternative to Wasserstein) | Implemented |
| Spectral and symmetric-operator archetypes; transfer test | Implemented |
| `ao.plotting` module (12 plot functions, lazy matplotlib) | Implemented |
| `ao.analyses` module (`measurement_report`, `benchmark_report`, `archetype_report`, `efficiency_comparison_report`) | Implemented |
| `ao.load_replogle_h5ad` (auto-detecting Replogle 2022 schema loader) | Implemented |
| Public K562/RPE1 essential-gene benchmark on Replogle data | **Executed**: K562 essential (188/200 guides, full rank d=30, cond 65.0) and RPE1 essential (153/200 guides, full rank d=30, cond 65.27). See `MANUSCRIPT.md` and `01b`/`01c` notebooks. |
| Random-split null diagnostic on `linearity_check` (`n_null=` argument) + `held_out_prediction_check` | Both diagnostics implemented and shipped. **Central paper finding**: at published Perturb-seq scale (d≈30, n≈200, per-guide Δz noise σ≈0.266), both diagnostics are noise-limited — observed values on Replogle K562 and RPE1 essential-gene screens (rel_diff = 1.47/1.57; held-out ρ = 1.12/1.22) fall within 0.05 of the noise-floor prediction for a synthetic linear ground truth at matched (d, n, U, κ, σ). The preregistered 0.25 threshold is unreachable at this scale for any dataset (linear or nonlinear). Detection of moderate saturating nonlinearity requires ~48k cells/guide with wide κ (~300× current published Perturb-seq). See MANUSCRIPT.md §3.5–§3.6 for the power analysis; `reproduction/10_figS10_realscale_positive_control.py` for the methodology; `tutorial/04_linearity_diagnostics_power_analysis.ipynb` for how to apply it to your own data. |
| Shared `scjdo.operator` metric substrate | Release blocker: sibling repository was not available in this workspace |
| Phase 4 constrained anchored inference | Deliberately gated on preregistered Phase 2 above-null evidence |

Read [`STRATEGY_REVIEW.md`](STRATEGY_REVIEW.md) for the detailed mathematical correction, experimental risk register, and release gates. Read [`PREREGISTRATION.md`](PREREGISTRATION.md) before any Phase 2 benchmark is executed. Read [`SPEC.md`](SPEC.md) §"Modeling assumption caveat — additive-input vs. intervention" for the documented systematic bias direction that `anchor-op` (as an additive-input tool) has against intervention-like CRISPRi biology, and for the reason a program-space intervention alternative is not shipped.

## Repository layout

- **`tutorial/`** — task-oriented notebooks walking through every public API entry point on synthetic data (self-contained; no external downloads). Start here to learn the tool. See `tutorial/README.md` for the reading order.
- **`reproduction/`** — one script per manuscript figure. Regenerates every figure in `manuscript_figures/`. See `reproduction/README.md` for data dependencies and runtimes.
- **`examples/`** — original API walkthrough notebooks on real data (Replogle K562, Replogle RPE1, K562 aggregate, synthetic). Overlaps partly with `tutorial/` but retained as executed real-data references.
- **`src/anchorop/`** — the package.
- **`tests/`** — pytest suite (56 tests).
- **`manuscript_figures/`** — every figure referenced in `MANUSCRIPT.md`.
- **`results/`** — pickled measurement bundles produced by `reproduction/03` and `reproduction/04`; loaded by `reproduction/05` and `reproduction/10`.

## Example notebooks in `examples/`

Every notebook opens with a `> **STATUS —**` banner identifying it as real data, demo, or synthetic. **Note**: numeric claims in older notebook markdown may pre-date the paper's power-analysis reframing of the linearity diagnostics (§3.5–§3.6). The `tutorial/` notebooks reflect the current framing.

| Notebook | Status | Purpose |
|---|---|---|
| `01_measure_k562.ipynb` | Real data (partial identification on this dataset) | Full measurement pipeline on raw 10x + CRISPR analysis tarballs (K562 84K noncoding-element aggregate) |
| `01b_measure_k562_replogle.ipynb` | Real data (executed) | Replogle 2022 K562 essential-gene screen — 188/200 guides retained, full rank 30/30, cond 65.0. Linearity diagnostics run but are noise-limited at this scale (see MANUSCRIPT.md §3.5–§3.6). |
| `01c_measure_rpe1_replogle.ipynb` | Real data (executed) | Replogle 2022 RPE1 essential-gene screen — 153/200 guides retained, full rank 30/30, cond 65.27. Linearity diagnostics also noise-limited at this scale. |
| `02_benchmark.ipynb` | Infrastructure real + illustrative baseline methods | Preregistered inferred-vs-measured comparison + coordinate transforms. Runs against the K562 essential-gene measurement with four constructed baselines until real third-party operators are supplied. |
| `03_archetypes.ipynb` | Demo (bootstrap when only one state is available) | Operator archetypes and cross-state transfer test |
| `04_synthetic_walkthrough.ipynb` | 100% synthetic | Full pipeline demonstration with known ground truth `J` |

## Manuscript

A full methods paper is at [`MANUSCRIPT.md`](MANUSCRIPT.md) (~6,300 words main text, ~370-word abstract) with figures in `manuscript_figures/`. Mathematical derivations are in [`MATH.md`](MATH.md) — every non-trivial equation is numerically validated. The paper's central contribution is a matched-scale positive control for perturbation-response operator recovery at published Perturb-seq scale. Target venue is a full methods journal (PLOS Computational Biology / Bioinformatics research paper / Genome Biology methods track). All numeric claims reproduce from the 56-test suite plus the per-figure scripts in `reproduction/`.

Headline result: **at Replogle-scale geometry and each dataset's own noise level, full-operator recovery is practically absent; leading-direction alignment is stronger but does not meet the predefined recovery threshold. The result is strongly incompatible with interpreting fitted spectra or edges as quantitatively estimated full operators under the tested model and noise conditions.** Under a synthetic linear ground truth `J_true` at Replogle-matched (d=30, real U, real κ, per-dataset σ measured from within-guide bootstrap: K562 σ=0.240, RPE1 σ=0.352, Jost σ=0.036 per target-aggregate), a 200-replicate pipeline-matched empirical null shows the anchor-op fit's Frobenius cosine with truth (+0.033 K562, +0.025 RPE1) is *statistically indistinguishable* from a cross-replicate null pairing each fit with an independently drawn ground truth (K562 null +0.035, z = −0.05). The fit is shrunk ~150-fold in norm. Result holds under dense, 10%-sparse, 2%-sparse, and rank-5 ground-truth structures (Fig S13), is not rescued by a sparsity-aware row-wise LASSO fit under oracle penalty selection (cos ≈ +0.06; Fig S14), and is robust to the noise-model choice (residual-resampled vs i.i.d. Gaussian differences within 1 SD; Fig S17) and to the ground-truth stability shift (full-operator cos ∈ [0.02, 0.06] across c ∈ [0.5, 3.0]; Fig S18). Leading-direction alignment (cos_1 ≈ 0.29 mean over 200 replicates on K562) is statistically detectable on the population mean (mean-difference z ≈ 12 vs shuffled-U null) but not on any individual replicate (per-rep SD ≈ 0.33; z_per-rep ≈ 0.9), and remains below the prespecified up-to-scale threshold of 0.5. Full-operator direction recovery would require per-guide σ ≲ 0.01, corresponding to ~68k cells/guide (~550× current). Downstream: both linearity diagnostics (`linearity_check`, `held_out_prediction_check`) are also noise-limited at published scale (Figs S10–S12). Tool-level positive contributions independent of the recovery gap: (1) type-level identifiability discipline (preregistered `rank_tol` guard, hard block on full Jacobian at partial identification); (2) data-format-aware efficiency-estimation regime (`efficiency_estimator="auto"` router: `mean_ratio` on count data + `min_control_detection_rate` filter, `detection_rate` documented as the analytic signed-shift statistic on pre-scaled residual data). Actionable output: run the matched-scale operator-recovery positive control on evaluation datasets before drawing inference-tool conclusions.

## Scope and non-goals

`anchor-op` refuses program spaces above `d = 200` and warns above `d = 100`. It does not perform gene regulatory network inference, trajectory inference, or drift-field training. It consumes imported matrices from tools such as CellOracle, dynamo, or scJDO only after they have been transformed into the same labeled program coordinates.

The package cannot establish that a post-perturbation snapshot is at a common steady state. It reports evidence for or against a local linear approximation using the supplied cells and guides. A successful analysis must state the timepoint, guide filtering, control definition, response rank, regularization path, and linearity result.

## Development

Assumes the conda environment from [Installation](#installation) is active. Then:

```bash
pytest -q
```

The suite (currently **56 tests**) includes `test_ACCEPTANCE_` synthetic recovery, partial-identification orientation, regularization, safety, null-calibration, archetype, efficiency-estimator, linearity-check, held-out-prediction, and analyses-API tests. CI runs the suite on supported Python versions. Tests that exercise plotting are marked `pytest.mark.skipif(not matplotlib installed)` so the core suite runs without a display or matplotlib when a caller has opted into the pip-only path.

## License

MIT. See [`LICENSE`](LICENSE).

## References

[1]: https://pubmed.ncbi.nlm.nih.gov/35688146/ "Replogle et al. (2022), Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq"
[2]: https://elifesciences.org/articles/43803 "Kotliar et al. (2019), Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq"
[3]: https://gwps.wi.mit.edu/ "Genome-Wide Perturb-Seq data access"
