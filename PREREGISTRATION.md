# Phase 2 Preregistration — anchor-op

**Status:** recorded before any repository Phase 2 benchmark output  
**Recorded at:** 2026-07-29 MDT  
**Repository commit:** populated after the initial baseline commit; do not amend after benchmark data are inspected.

## Question

When an inferred local operator is transformed into the same control-derived program coordinates as a Perturb-seq measured action, does it predict the experimentally constrained action better than declared null operators?

The experiment measures `J P_X = -U S⁺`, where `P_X` projects onto the retained response/domain subspace `X = range(S)`. All primary operator comparisons will therefore evaluate `((J_inf - J_meas) P_X)`. A full spectral result is reported only when `rank_eff(S) = d`; otherwise it is explicitly unavailable rather than inferred from an arbitrary zero extension.

## Primary prediction

> **Pre-registered directional prediction:** On the identified domain, inferred methods will show stronger agreement with the symmetric component of the measured action than with the antisymmetric component.

The prediction is evaluated using the paired method-level difference

```text
antisymmetric_relative_error − symmetric_relative_error.
```

A positive difference supports the prediction. The benchmark will report the raw errors, their guide-bootstrap intervals where available, the null distributions, and the sign of the paired difference for each imported method. No method-specific choice of regularizer, program dimension, edge filtering, or coordinate transform may be adjusted after observing these metrics to improve the result.

## Confirmatory analysis plan

| Element | Locked choice |
|---|---|
| Primary dataset | Replogle K562 genome-scale Perturb-seq, with its published experimental timepoint and documented preprocessing provenance |
| Secondary state | Replogle RPE1, reserved for post-benchmark archetype transfer testing |
| Program basis | Control-only program basis; `d` chosen and documented before importing inferred matrices; cNMF seed ensemble concordance reported |
| Perturbation unit | Guide, with target gene supplied explicitly and guide-level `κ_g` estimated from the target transcript |
| Required controls | Matched non-targeting controls; batch matching when batch metadata are present |
| Measurement regularization | Full TSVD and/or Tikhonov path reported; primary setting selected by the documented pre-data rule; sensitivity analysis retained |
| Primary matrix endpoint | Projected operator relative error and held-out guide equation residual |
| Secondary endpoints | Symmetric/antisymmetric relative errors, their bounded agreement transforms, and, only under full effective rank, spectral Wasserstein, hyperbolicity sign, and invariant-subspace distance |
| Nulls | Shuffled-edge operator and random-initialization operator, with the number of draws and seed reported |
| Uncertainty | Bootstrap guide columns; no cell-only bootstrap as the primary uncertainty estimate |
| Linearity diagnostic | Weak versus strong `κ_g` bins, comparing implied measured actions and held-out residuals |

## Predeclared numeric thresholds

These thresholds are locked before any Phase 2 output. Amendments require a dated entry in §Amendments **and** must state whether benchmark outputs were already observed. No threshold may be relaxed after seeing a result it would otherwise fail.

| Threshold | Value | Meaning | Rationale |
|---|---:|---|---|
| Linearity — weak/strong bin action agreement | `0.25` | `frobenius_relative_error(J_weak · P_common, J_strong · P_common) ≤ 0.25` on the common identified subspace, where bins are split by median guide efficiency | 25% is the largest disagreement that leaves the linear settled-state model useful as a first-order description; above this, the operator is reported as a limitation, not a result. |
| Identifiability — rank tolerance | `1e-2` | A singular direction of `S` counts toward `effective_response_rank` only if its singular value exceeds `1e-2 · σ_max(S)` | Real Perturb-seq guides are frequently collinear (paralogs, complex subunits, shared pathway members); the machine-precision default silently declares full rank on rank-deficient inputs. |
| Full-rank gate for spectral metrics | `effective_response_rank == d` | Spectral Wasserstein, hyperbolicity sign, invariant-subspace angle, and operator archetype fits are reported **only** at full effective rank | A partial zero-extended action has no generally interpretable spectrum. |
| Above-null decision for Phase 4 gate | Primary endpoint on **held-out** guides strictly better than the 95th percentile of the null distribution | Held-out generalization is the only Phase 2 outcome that unlocks Phase 4 | Self-reported in-sample fit does not open the gate. |

## Decision rules

The release must state the Phase 2 result regardless of direction. If inferred operators are indistinguishable from nulls on the declared primary endpoints, that null result is the benchmark conclusion. The analysis will not add post hoc transformations, datasets, or metric variants to improve agreement.

If weak/strong guide bins imply materially different identified actions above the linearity threshold above, the local linear settled-state model is reported as a primary limitation. If effective response rank is materially lower than `d`, claims are restricted to `J P_X`; full Jacobian spectra and hyperbolicity claims are not made.

Phase 4 anchored inference remains disabled unless a separately frozen Phase 2 evidence record shows above-null performance on **held-out perturbation guides** at the threshold above. A self-reported in-sample fit does not open the gate.

## Amendments

Any amendment must add a dated entry below, retain the original text, and state whether benchmark outputs were already observed.

| Date | Benchmark outputs observed? | Amendment | Rationale |
|---|---:|---|---|
| 2026-07-29 | No | Initial preregistration | Repository baseline created before Phase 2 execution |
| 2026-08-09 | Yes | **Threshold reframing, not relaxation.** The linearity threshold above (`rel_diff ≤ 0.25` on the weak/strong bin comparison) is retained as a preregistered numerical value but its interpretation is amended: on the observed Replogle K562 and RPE1 essential-gene measurements (188/200 and 153/200 guides at d=30), the threshold is unreachable — not because linearity fails, but because a real-scale positive control on synthetic linear ground truth matched to Replogle (d=30, n=200 guides, narrow κ ∈ [0.05, 0.50], per-guide Δz noise σ = 0.266 measured from within-guide cell-level bootstrap on the K562 essential h5ad) reproduces the observed rel_diff = 1.47 within 0.05 (§3.5, MANUSCRIPT.md). The diagnostic has essentially no rejection power against saturating nonlinearity at this scale (§3.6, Fig S12): detection of sat=0.5 tanh saturation requires ~48,000 cells/guide with wide κ (~300× current Perturb-seq); the narrow-κ single-sgRNA-per-target design does not reach detection at any tested noise level. The threshold is therefore **not relaxed** — no dataset that previously "failed" is now claimed to "pass." Instead, the pass/fail decision on all currently-analyzed datasets is amended to **not defensible either way**: neither the identifiability discipline nor the observed rel_diff can distinguish linear from moderately-nonlinear at published Perturb-seq scale. The threshold remains at 0.25 for datasets whose (d, n_guides, κ range, σ) point would give the diagnostic detection power; per Fig S12, no such dataset presently exists. The Phase 4 anchored-inference gate remains closed. | Real-scale positive control (Fig S10 in MANUSCRIPT.md) and rejection-power surface (Fig S12) show the preregistered threshold was set without a power analysis; adding it retrospectively per §3.5–3.6 does not relax the decision but reframes what a "pass" or "fail" would require. |
