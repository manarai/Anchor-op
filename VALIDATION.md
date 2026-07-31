# Validation Record

**Repository:** `anchor-op`  
**Commit:** `9a283cd` — `Initial validated anchor-op implementation`  
**Validation date:** 2026-07-29

## Scope of validation

This record distinguishes **software verification** from biological validation. The repository has been checked for mathematical guardrails, package integrity, and reproducible synthetic acceptance behavior. It does **not** claim that any real Perturb-seq dataset supports a biological conclusion. The provided notebooks are deliberately unexecuted protocols that require a versioned external dataset and recorded preprocessing provenance.

| Check | Result | Evidence |
|---|---:|---|
| Acceptance and regression tests | Passed | `13 passed` with `pytest -q` |
| Static analysis | Passed | `ruff check src tests` returned “All checks passed!” |
| Format conformance | Passed | `ruff format --check src tests` reported 14 formatted files |
| Python compilation | Passed | `python3 -m compileall -q src` completed successfully |
| Wheel build | Passed | `anchor_op-0.1.0-py3-none-any.whl` built from `pyproject.toml` |
| Isolated installation | Passed | Clean virtual environment imported `anchorop 0.1.0` successfully |
| Notebook structure | Passed | Every example notebook was parsed successfully as JSON |
| Continuous integration | Included | GitHub Actions tests Python 3.10, 3.11, and 3.12, including a wheel build |

## Strategic corrections incorporated

The main revision is mathematical rather than cosmetic. The defining equation `J S = -U` identifies the right-projected action `J P_X = -U S⁺`, where `X = range(S)`. The response domain is therefore the domain of identified Jacobian action. `range(U)` is reported separately as the actuated input space and is never substituted for that response domain.

| Risk identified during review | Repository safeguard |
|---|---|
| A partial measurement could be mislabeled as a full Jacobian | `MeasuredOperator.J` raises unless full response-domain rank is certified. |
| Spectra of a zero extension could be misreported as biological spectra | Spectral and hyperbolicity metrics are blocked on partial measurements. |
| Controls could leak into basis construction or fail to match perturbations | The basis accepts an explicit control mask, and guide responses require matched non-targeting controls. |
| Target genes could be guessed from guide labels | Target mapping requires a target column or explicit guide-to-target mapping. |
| Inverse instability could be hidden | TSVD/Tikhonov paths, effective rank, singular values, condition number, and retained directions are returned in every report. |
| Efficiency-bin comparison could conflate non-overlapping domains | The linearity diagnostic compares only the common identified response subspace and records `overlap_rank`. |
| Operator archetypes could be fit to partial zero extensions | Operator archetypes require full rank, use symmetric actions, simplex weights, observed extremes, and held-out-guide `k` selection. |
| Premature constrained inference could be asserted | Anchored inference remains a hard Phase 2 gate requiring explicit above-null evidence. |

## Reproducibility and remaining boundaries

The test suite uses synthetic linear systems solely to verify algebraic recovery and error handling; it is not biological evidence. The public-data notebooks expect users to obtain and document a specific processed AnnData source before use. They do not silently fetch, curate, normalize, or analyze a public dataset. Consequently, a future real-data run should commit the data source/version, selection mask, feature universe, seed values, regularization path, and final `AnchorReport` alongside results.

The package baseline is ready for repository publication or further scientific review. It is intentionally conservative: external inferred operators must already be in the same ordered program coordinates, and the Phase 4 constrained inference API refuses to operate until the preregistered Phase 2 evidence record is complete.
