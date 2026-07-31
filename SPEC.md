# anchor-op — Revised Build Specification

**Repository:** `manarai/anchor-op`  
**Package / import:** `anchor-op` / `anchorop`  
**License:** MIT  
**Python:** 3.10 or later

## Purpose

`anchor-op` treats a pooled Perturb-seq experiment as a measurement of an **identified local dynamical action** in a low-dimensional program space. Given single-cell CRISPRi/CRISPRa data and matched non-targeting controls, it estimates guide-level response and perturbation matrices, reports the regularized action constrained by those responses, benchmarks externally inferred operators, and learns interpretable archetypes of sufficiently identified operator geometry.

The package does not report “the cellular operator” without qualification. Its scientific product is a measured sensitivity/action with a declared coordinate system, timepoint, linearity assumption, regularization path, and identified subspace.

## Mathematical contract

Let `E ∈ R^(n×G)` denote normalized expression and let `W ∈ R^(G×d)` be a program basis fitted only on control cells. Program coordinates are `z = Wᵀe`; supported `d` lies in `1…200`, with a warning above 100 and an error above 200.

For a guide targeting gene `g`, define `κ_g ∈ (0, 1]` as its target-transcript knockdown efficiency and let

```text
u_g = -κ_g Wᵀδ_g.
```

At the chosen settled timepoint, the model assumes a local constant-input response:

```text
Δz_g = z̄_pert,g − z̄_ctrl = -J⁻¹u_g.
```

Stacking retained guides gives `S = -J⁻¹U`, equivalently `J S = -U`. Therefore, if `P_X = S S⁺` projects onto `X = range(S)`,

```text
J P_X = -U S⁺.
```

> **Non-negotiable correction:** `range(U)` is the actuated input space, whereas `range(S)` is the domain on which the Jacobian action is identified by the equation above. The code reports both. It compares inferred and measured actions using right projection by `P_X`.

The pseudo-inverse is computed by TSVD or Tikhonov regularization. The complete regularization path, effective rank, retained singular directions, and condition number are returned. Small singular values of `S` directly govern inverse instability; the implementation does not equate this unconditionally with small eigenvalues of a biological `J`.

## Public object contract

```python
basis = ao.fit_programs(adata, d=50, method="cnmf", control_mask=control_mask)
measurement = ao.measure_operator(
    adata,
    basis,
    guide_key="guide",
    target_key="target_gene",
    control_label="non-targeting",
    reg="tsvd",
    reg_param="path",
)
```

`measurement` is a `MeasuredOperator` holding `S`, `U`, guide identifiers, an identified action, and an `AnchorReport`. The report cannot be omitted. `measurement.identified_action` returns the reported `J P_X`. `measurement.J` raises an `IdentifiabilityError` unless the effective response-domain rank equals `d`.

`AnchorReport` contains the input and response projectors, their dimensions, guide-retention and drop reasons, guide efficiencies, singular spectrum, truncation/regularization selections, condition number, full path, and optional guide-bootstrap covariance.

## Required behavior

| Requirement | Required implementation behavior |
|---|---|
| Matched controls | Reject analyses without matched non-targeting control cells. When batch metadata are supplied, compute each guide response relative to the matching control batch composition. |
| Target mapping | Require `target_key` or explicit `guide_to_target`; never parse a target from an arbitrary guide ID. |
| Input encoding | Construct each input from the target’s program-loading row and measured `κ_g`; drop negligible-loading targets rather than zero-imputing them. |
| Guide uncertainty | Bootstrap guides, preserving guide-level columns of `S` and `U`. |
| Program basis | Fit only on controls. Use external cNMF/SCENIC-style loadings through validated `ProgramBasis` input rather than reimplementing upstream network tools. |
| Partial rank | Expose action-only comparisons; block full spectra, hyperbolicity, and operator archetypes until full effective rank is established. |
| Comparison coordinate system | Reject inferred matrices whose shape differs from `d`; require users to transform them into the exact same program coordinates. |
| Regularization | Make regularization explicit and reportable; `reg_param='path'` records the full default path. |

## Benchmark metrics

Every inferred matrix is evaluated against the action on its identified domain:

```text
operator_relative_error = ||(J_inf − J_meas) P_X||_F / ||J_meas P_X||_F.
```

The package also reports `||J_inf S + U||_F / ||U||_F`, symmetric/antisymmetric projected relative errors and bounded agreement transforms, and declared operator-level null distributions. When and only when the effective response rank equals `d`, it adds complex-spectrum 2-Wasserstein distance, agreement of the largest-real-part eigenvalue sign, and a leading invariant-subspace Grassmann distance.

## Archetypes

Spectral archetypes summarize full measured/inferred spectra into fixed-length real vectors. Operator archetypes use only the symmetric component of fully identified operators. Archetype weights are projected to the nonnegative simplex, and archetypes are observed extreme profiles selected by farthest-point traversal. For operator mode, `k='cv'` selects the number of archetypes through guide-held-out equation residuals, not an information criterion. `transfer_test()` compares source-fitted simplex reconstruction with an independent target refit.

## Phase boundary

The repository contains a structural gate for anchored inference. Before a Phase 4 constrained solver is introduced, `verify_phase2_gate()` requires an immutable evidence record with preregistration and benchmark commit references, held-out metric, null metric, and an above-null decision. The current baseline does not claim that gate has been passed.

## Acceptance requirements

| Acceptance test | Required result |
|---|---|
| Full-rank synthetic recovery | Recover the known `J` under noiseless linear response to numerical precision. |
| Partial-domain recovery | Recover `J P_X`, distinguish it from an unrelated left projection, and reject full `J` access. |
| Regularization path | Record TSVD ranks / Tikhonov filters and selected settings reproducibly. |
| Identifiability enforcement | No public code path exposes an operator action without a report. |
| Dimension guard | Error above `d=200`; warn above `d=100`. |
| Comparison calibration | Exact inferred operator scores better than declared shuffled/random null draws. |
| Spectral guard | Partial measurements report spectral metrics as unavailable rather than silently returning values. |
| Archetype simplex | Weights are nonnegative and sum to one; operator `k` selection uses held-out guides. |
| Reproducibility | Tests use pinned synthetic seeds; real-data notebooks must record dataset source/version and seeds. |

## Modeling assumption caveat — additive-input vs. intervention

`anchor-op` treats a CRISPRi perturbation as a **constant additive input** in program coordinates: `u_g = -κ Wᵀ δ_g`, giving `Δz = -J⁻¹ u_g`. This enables the closed-form inversion `J = -U S⁺`.

The biology of CRISPRi is closer to a **hard-clamp intervention**: dCas9-KRAB blocks transcription at the target locus, target expression drops toward zero, and other genes then relax through the network. The additive-input model and the intervention model coincide only when the target gene has no feedback interaction with the rest of the system — a degenerate case for a regulatory network.

**Empirical bias direction.** On synthetic gene-space data where the ground truth follows the intervention model, fitting under the additive model gives ~45% relative Frobenius error in `J`, and the largest real eigenvalue systematically shifts toward zero (in one d=8 experiment: true `Re λ_max = −1.17`, additive fit `−0.69`, intervention fit `−1.20`). The practical consequence: **the additive fit may report spurious near-hyperbolicity where the true system is more strongly damped**. Report `Re λ_max` close to zero as a candidate finding, not a conclusion.

**Why we do not ship an intervention-fit alternative.** A program-space intervention fit under the natural rank-`d` assumption is exactly under-identified — the projected observation `W ᵀ Δe` is identically zero (independent of `J_prog`), so program-space observations contain no information about the reduced Jacobian under this model. A gene-space intervention fit is well-posed but scales as `O(n_genes³)` per response evaluation, which is not tractable at Perturb-seq feature counts. Both are documented as out of scope for this release.

## Out of scope

The package refuses gene-level high-dimensional Jacobians, GRN inference, trajectory inference, and drift-field training. It does not silently fetch, preprocess, or claim results from public data. It imports user-provided inferred operators only after coordinate alignment and evaluates them independently. Program-space intervention (do-calculus) fitting is intentionally not shipped — see the modeling-assumption caveat above.
