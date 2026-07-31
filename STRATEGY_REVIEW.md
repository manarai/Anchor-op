# anchor-op Strategy Review

**Status:** implementation baseline, reviewed before coding  
**Scope:** the supplied design brief, corrected where the linear algebra or experimental assumptions require a stronger guardrail.

## Executive assessment

`anchor-op` is a valuable and testable proposal: genome-scale Perturb-seq supplies an unusually rich collection of controlled transcriptional response measurements, and the public Replogle resources include processed single-cell and pseudobulk AnnData files for the K562 and RPE1 experiments.[1] The proposed control-only program basis is also defensible: consensus NMF was developed to infer shared expression programs and their per-cell usages from single-cell data.[2]

The original plan is strongest when it treats the **sensitivity matrix** `S` as the directly estimated experimental object and treats every operator estimate as conditional on regularization, input encoding, and the linear settled-state model. The build therefore preserves the mandatory report, guide-level bootstrap, basis-stability check, linearity check, preregistration, and phase gate. It does **not** promise a universal cellular Jacobian.

> **Central refinement:** From `J S = -U`, the experimentally identified domain of the Jacobian is `range(S)`, not generally `range(U)`. `range(U)` is the *actuated input* subspace. The package must distinguish these two spaces and avoid calling a zero-extended partial map a fully measured operator.

| Design point | Original formulation | Implemented refinement |
|---|---|---|
| Identified subspace | `M = range(U)` used for operator projection | Store both `X = range(S)` (identified response/domain subspace) and `Y = range(U)` (actuated/codomain subspace). The directly identified map is `J P_X = -U S^+`. |
| Operator comparison | Left projection `P_M(J_inf - J_meas)` | Compare `((J_inf - J_meas) P_X)`; this evaluates exactly the action constrained by observed responses. |
| Spectral claims | Spectra always compared on `M` | Spectral and hyperbolicity metrics are **blocked** for partial, non-closed maps. They are available only after full effective-rank identification or an explicit reduced-model closure diagnostic. |
| Regularization rationale | Ill-conditioning attributed specifically to small eigenvalues of `J` | Report singular spectra and effective rank of `S` and `U`. Instability in `S^+` is governed directly by small singular directions of `S`, not by an unconditional one-to-one statement about eigenvalues of `J`. |
| Phase 4 | Listed as a later feature | Kept deliberately unavailable unless a recorded phase-2 evidence file proves above-null signal. |

## Mathematical correction: what the experiment identifies

For guide-level response columns, the linear steady-state model is

```text
S = -J^{-1} U,      equivalently      J S = -U.
```

Let `P_X = S S^+` be the orthogonal projector onto `X = range(S)`. Right-multiplying the equation by `S^+` yields

```text
J P_X = -U S^+.
```

Thus the experiment identifies the **action of `J` on the response subspace**. It does not, absent further conditions, identify `P_Y J` or the unrestricted `J`, where `Y = range(U)`. In the full-rank case, where the regularization retains `rank(S) = d`, `P_X = I` and the distinction disappears. In the partial case, spectral decomposition of `-U S^+` depends on the arbitrary zero extension outside `X`; it is not a measured spectrum of the biological Jacobian.

This distinction changes the default result object. `MeasuredOperator.identified_action` is always available after the mandatory report. `MeasuredOperator.J` is available only when the report certifies full effective-domain identification. Code must not silently pass a partial zero extension to eigenvalue-based analysis.

## Experimental assumptions that must be tested

The K562 and RPE1 public resources make the planned primary and transfer analyses feasible in principle, but the public study provides measurements at specified post-transduction days rather than a direct verification that every guide-positive cell is at the same local steady state.[1] The repository therefore treats settled-state behavior as an assumption to be stress-tested, not as a given.

| Risk | Required diagnostic | Failure consequence |
|---|---|---|
| Nonlinear or non-settled guide response | Fit weak and strong knockdown-efficiency bins separately; compare their identified actions and held-out equation residuals | Label the operator as a local linear approximation failure; do not headline spectral biology. |
| Control / perturbation compositional imbalance | Require batch-aware centering when a batch key is supplied; report guide and control counts by batch | Refuse strata with no matched controls and report dropped guide-batch cells. |
| Guide heterogeneity | Estimate `κ` at guide level; bootstrap guides rather than treating cells as independent replicates | Report covariance and guide-resampling intervals; never use cell-only bootstrap as the primary uncertainty estimate. |
| Basis instability | Run a seed ensemble for cNMF and report component concordance | Flag unstable programs and halt downstream operator claims if basis concordance falls below the configured threshold. |
| Incomplete excitation | Report `rank(U)`, `rank(S)`, the retained regularized rank, and projectors | Restrict claims to `J P_X`; block full-operator spectra and anchoring. |
| Regularization sensitivity | Produce a complete TSVD or Tikhonov path and selection record | Report a range of actions, not a single unqualified matrix. |

## Implementation decisions

The codebase is intentionally organized around experimental objects rather than a generic matrix inverse. A guide-level design table supplies the guide identifier, target gene, guide-positive mask, matched-control label, optional batch, and optional weight. No target is inferred from an arbitrary guide sequence; a `target_key` or explicit mapping is required. The target transcript must be present in the program-basis gene index to construct `u_g = -κ_g W^T δ_g`.

Program fitting uses control cells only. The baseline supports deterministic NMF and a seed-ensemble cNMF approximation, while externally derived regulon/loadings can be supplied as a validated precomputed basis. The baseline deliberately does not claim to reimplement SCENIC or CellOracle. Those packages remain upstream producers of program scores or inferred operators.

A measured response is computed per guide, not first collapsed per target. This preserves guide-level heterogeneity and makes bootstrap-over-guides meaningful. The package estimates a robust guide knockdown efficiency from its target transcript relative to matched controls, removes nonpositive / uninformative perturbations, constructs `S` and `U`, and then estimates the identified action using a TSVD or Tikhonov pseudo-inverse of `S`.

The Phase 2 benchmark begins with equation-level and projected-operator tests. For a full effective-rank measurement it may additionally compare spectral Wasserstein distance, hyperbolicity sign, and invariant subspaces. For a partial measurement, the valid primary endpoint is projected action error plus held-out guide equation residual. The symmetric/antisymmetric split is reported as a projected matrix comparison; it is not automatically interpreted as a direct measurement of the unrestricted antisymmetric biological Jacobian.

## Additional acceptance tests

The supplied acceptance criteria are retained and strengthened with the following checks.

| Test | Purpose | Pass condition |
|---|---|---|
| Exact full-rank recovery | Verify `J = -U S^+` under the model | Relative error is near machine precision for noiseless full-rank synthetic data. |
| Partial-identification orientation | Catch the `range(U)` versus `range(S)` error | `J P_X` recovers exactly while an unrelated left-projected map is demonstrably not asserted. |
| TSVD-path monotonicity | Ensure truncation metadata is honest | Effective rank and retained singular-direction flags match the requested truncation. |
| Tikhonov-path construction | Ensure the alternate regularizer is inspectable | Every requested alpha yields a finite action and recorded spectral filter factors. |
| Spectral guard | Prevent invalid biological claims | Partial non-closed measurements raise a clear error when spectral metrics are requested. |
| Held-out guide equation check | Test generalization beyond fitted guides | Report `||J_train S_test + U_test||` alongside the null distribution. |
| Basis seed ensemble | Detect coordinate instability | Component concordance is persisted in `ProgramBasis` and can gate downstream analysis. |

## Scope boundary and release discipline

The source specification requires a shared `scjdo.operator` substrate. That sibling package was not provided in this workspace, so this baseline isolates a small local, documented metric implementation behind `anchorop.compare` for testability. Before a public scientific release, it must be replaced by or extracted into the shared metric package and checked against the sibling repository's tests. This is a release blocker, not a silent deviation.

The repository does not download or analyze the multi-million-cell public dataset during package construction. Its data-facing notebook provides a reproducible, pinned workflow and explicit data source; the automated suite uses only synthetic data. Consequently, the README must not claim a K562 operator, a reference `dim(M)`, or a Phase 2 result until those computations have actually been run. The preregistered prediction is recorded before any benchmark execution, and anchored inference remains gated.

## References

[1]: https://pubmed.ncbi.nlm.nih.gov/35688146/ "Replogle et al. (2022), Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq"
[2]: https://elifesciences.org/articles/43803 "Kotliar et al. (2019), Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq"
[3]: https://gwps.wi.mit.edu/ "Genome-Wide Perturb-Seq data access"
