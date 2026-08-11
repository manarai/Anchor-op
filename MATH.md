# anchor-op — Mathematical derivations

This document derives every non-trivial mathematical claim `anchor-op` relies on from first principles. Every non-trivial equation has been numerically validated against the code; the validation script lives in the repository history and every claim below reproduces from `pytest -q` (50 tests).

**Complement to other docs:**
- [`SPEC.md`](SPEC.md) — the *contract* (what the tool guarantees to users).
- [`MANUSCRIPT.md`](MANUSCRIPT.md) — the *methods paper*, constrained by word count.
- [`STRATEGY_REVIEW.md`](STRATEGY_REVIEW.md) — early mathematical corrections + risk register.
- **This document** — the *derivations*, without word limits, that the above three assume.

**Audience:** reviewers of the paper; users extending the code; anyone trying to determine whether the tool's assumptions apply to their data.

**Notation:**
- Lower case bold-ish letters are vectors; upper case are matrices.
- `z ∈ R^d` — cell state in program coordinates.
- `e ∈ R^G` — cell state in gene coordinates (unnormalized expression).
- `W ∈ R^(G × d)` — program basis (gene × program loadings), fit on non-targeting control cells only.
- `κ_g ∈ (0, 1]` — knockdown efficiency of guide targeting gene *g*.
- `u_g ∈ R^d` — perturbation input in program coordinates.
- `S ∈ R^(d × m)` — sensitivity matrix (columns are per-guide response `Δz`).
- `U ∈ R^(d × m)` — perturbation input matrix (columns are per-guide `u_g`).
- `J ∈ R^(d × d)` — local Jacobian of the dynamics in program space.
- `S⁺` — Moore-Penrose pseudo-inverse of `S`.
- `P_X = S S⁺` — orthogonal projector onto `range(S)`.
- `‖·‖_F` — Frobenius norm.

**Cross-references to code:** each section names the module and function it derives.

---

## 1. Linear response framework

*Code: `src/anchorop/measure.py::measure_from_sensitivity`, `src/anchorop/identifiability.py::regularized_pseudoinverse`.*

### 1.1 Setup

Assume cell dynamics in program coordinates are governed by

$$\frac{dz}{dt} = h(z),$$

where `h: R^d → R^d` encodes the aggregate regulatory dynamics of the cell in the low-dimensional program space. Assume there exists a locally-stable fixed point `z* ∈ R^d` at which control cells sit (`h(z*) = 0`, and the linearization around it is stable).

Linearizing around `z*`,

$$\frac{dz}{dt} \approx J \cdot (z - z^*), \qquad J = \frac{\partial h}{\partial z}\bigg|_{z^*}.$$

`J` is the local *Jacobian* — the operator we ultimately want to measure.

### 1.2 Additive-input perturbation

Under CRISPRi, the *modeling assumption* is that a knockdown of gene *g* at efficiency `κ_g` adds a constant input `u_g` in program coordinates:

$$\frac{dz}{dt} = J (z - z^*) + u_g, \qquad u_g = -\kappa_g \, W^T \delta_g \, z_g^*,$$

where `δ_g ∈ R^G` is the standard basis vector for gene *g* in gene space, so `W^T δ_g` is the *g*-th row of `W` — the gene's loadings across programs. The scalar `z_g*` is the baseline expression of gene *g*. The overall sign is chosen so that `u_g` represents a "reduction" in gene *g*'s pro-abundance drive.

(Section 4 discusses whether this additive-input model faithfully represents CRISPRi biology. Short answer: it is a linearization, and it is systematically biased for finite knockdown strength.)

At the new fixed point,

$$0 = J \Delta z + u_g \quad \Rightarrow \quad \Delta z = -J^{-1} u_g.$$

Stacking `m` perturbation experiments column-wise into `S` (columns are observed responses) and `U` (columns are input encodings),

$$\boxed{S = -J^{-1} U, \qquad J S = -U.}$$

### 1.3 Recovering `J`

Right-multiplying `J S = -U` by the Moore-Penrose pseudo-inverse `S⁺`,

$$J S S^+ = -U S^+ \quad \Rightarrow \quad J P_X = -U S^+, \qquad P_X := S S^+.$$

`P_X = S S⁺` is the orthogonal projector onto `range(S)`; this is a standard property of the pseudo-inverse for real matrices. Consequently:

$$\boxed{\text{The experiment identifies the action } J P_X = -U S^+ \text{ on } X := \text{range}(S).}$$

If `rank(S) = d` (full identification), `P_X = I` and the full `J` is recovered. If `rank(S) < d`, `J`'s action outside `range(S)` is unconstrained.

**Numerical verification** (in-repo, `pytest` acceptance tests):
- `test_ACCEPTANCE_exact_full_rank_recovery_and_report` — full-rank case recovers `J` to machine precision.
- `test_ACCEPTANCE_partial_domain_has_correct_orientation_and_blocks_full_j` — partial-rank case recovers `J P_X` exactly, and refuses to expose `J` itself.

---

## 2. Identifiability partition — why `range(S)` and not `range(U)`?

*Code: `src/anchorop/types.py::MeasuredOperator.J`, `src/anchorop/types.py::AnchorReport`.*

Two subspaces are natural:

- `X = range(S)` — the **identified response subspace**. Directions in `X` are where the experiment observed a response.
- `Y = range(U)` — the **actuated input subspace**. Directions in `Y` are the perturbation directions the experiment excited.

The tool's central discipline is: **only `X` bounds what is identified**, not `Y`.

**Why**: From `J S = -U`, right-multiplying by `S⁺` gives the identified action on `X`. Left-multiplying by anything doesn't help — no operation on `U` alone constrains `J`'s action on any subspace outside `X`.

More formally: given the linear system `J S = -U`, treating `J` as the unknown (a `d × d` matrix, `d²` scalar parameters), the number of equations is `d · rank(S)`. The system is underdetermined by `d(d - rank(S))` parameters — the unidentified subspace has dimension `d - rank(S)`, extending in the null space of `S⁺` from any solution.

**Geometric intuition:** think of `J` as a linear map. Each column of `S` is an observed response direction. `J` acting on that column must produce the negative of the corresponding `U` column. This constrains `J` on the span of `S` columns — nothing else.

**Code consequence:** `MeasuredOperator.J` raises `IdentifiabilityError` unless the report certifies full effective response-domain rank. `MeasuredOperator.identified_action` always returns `J P_X`.

---

## 3. Regularization and the `rank_tol` parameter

*Code: `src/anchorop/identifiability.py::_svd`, `_tsvd_entry`.*

### 3.1 SVD-based pseudo-inverse

Let `S = U_S Σ V_S^T` be the SVD, `Σ = diag(σ_1, ..., σ_{min(d,m)})`. The Moore-Penrose pseudo-inverse is

$$S^+ = V_S \Sigma^+ U_S^T,$$

where `Σ⁺` inverts nonzero singular values and zeros the rest.

For small `σ_i`, `1/σ_i` amplifies whatever noise is present in the direction of the `i`-th singular vector. This is the *classical ill-conditioning* problem.

### 3.2 Numerical rank at tolerance

Given a tolerance `τ`, the "numerical rank" is

$$r(S; \tau) = \# \{ i : \sigma_i > \tau \}.$$

Two natural choices for `τ`:

- **Machine-precision default**: `τ_ε = ε · max(d, m) · σ_max ≈ 2×10⁻¹³ · σ_max`. Any `σ_i` above this counts as identified. This accepts almost every singular direction as signal.
- **Scientific default (`rank_tol=1×10⁻²`)**: `τ = 0.01 · σ_max`. Only accept singular directions above 1% of the leading direction. This corresponds to a physical noise-floor argument: below 1% of the leading response, singular directions are typically dominated by shared measurement noise across guides.

### 3.3 Why `1×10⁻²` specifically

On a synthetic setup with rank-2 truth plus 1% additive noise (validated in repo):

```
singular values of S: [15.50, 9.72, 0.057, 0.041, 0.032, 0.026]
ratio to max:         [1.00, 0.63, 0.004, 0.003, 0.002, 0.002]

eps-based numerical rank: 6 (accepts all noise directions as signal)
rank_tol=1e-2 numerical rank: 2 (correctly identifies the true rank)
```

Specific numbers here are illustrative from one seed; the pattern — two large singular values from the true rank, four small ones two orders of magnitude below — is stable across seeds and characterizes the "noise floor cliff" the tolerance is designed to catch.

Real Perturb-seq guide libraries produce singular-value ratios spanning 3-4 orders of magnitude. Directions below ~1% of `σ_max` are typically dominated by shared per-cell noise + guide correlation structure, not by network signal. The `1×10⁻²` cutoff is preregistered in `PREREGISTRATION.md` before any real-data analysis.

### 3.4 The Tikhonov alternative

The Tikhonov-regularized pseudo-inverse is

$$S_{\alpha}^{+} = V_S \, \text{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \alpha}\right) U_S^T.$$

For `α → 0`, this recovers `S⁺` (truncated at rank_tol). For finite `α`, small singular directions are damped rather than truncated. The tool retains both regularization families and reports the full regularization path; a scientifically justified selection rule (generalized cross-validation for Tikhonov, `rank_tol`-based rank for TSVD) is applied by default.

---

## 4. Additive vs intervention perturbation model

*Discussed in `SPEC.md` §"Modeling assumption caveat" and quantified in `MANUSCRIPT.md` §4.1.*

This is the most consequential modeling assumption in anchor-op. It affects the interpretation of every measured eigenvalue.

### 4.1 The two models

**Additive-input model** (what `anchor-op` fits):

$$\frac{dz}{dt} = J (z - z^*) + u_g, \qquad u_g = -\kappa \, z_g^* \, e_g,$$

where `e_g` is the `g`-th standard basis vector. The intervention is modeled as adding a constant sink to gene *g*'s dynamics — the perturbation is a *forcing term*.

**Hard-clamp intervention** (what CRISPRi biology actually does):

Clamp gene *g*'s expression at `(1-κ) z_g*` (reduced by fraction κ). Other genes then evolve under the *unmodified* dynamics `dz/dt = h(z)` subject to the fixed constraint. Following through the linearization,

$$\frac{dz_{-g}}{dt} = J_{11} (z_{-g} - z^*_{-g}) + J_{12} (z_g - z^*_g),$$

where `J_11 = J[-g, -g]` (block excluding gene *g*), `J_12 = J[-g, g]` (column *g* minus the *g*-th entry). Substituting `z_g = (1-κ) z_g*` (so `z_g - z_g* = -κ z_g*`),

$$\frac{dz_{-g}}{dt} = J_{11} \Delta z_{-g} - \kappa \, z_g^* \, J_{12}.$$

At the new fixed point,

$$\Delta z_{-g}^{\text{int}} = \kappa \, z_g^* \, J_{11}^{-1} J_{12}, \qquad \Delta z_g^{\text{int}} = -\kappa \, z_g^*.$$

### 4.2 The additive-model response, explicit

From `Δz = -J⁻¹ u_g` with `u_g = -κ z_g* e_g`,

$$\Delta z^{\text{add}} = \kappa \, z_g^* \, J^{-1} e_g = \kappa \, z_g^* \, (J^{-1})_{:, g}.$$

Using the standard block-matrix inverse formula, for `J = [[J_11, J_12], [J_21, J_22]]` and Schur complement `c := J_22 - J_21 J_{11}^{-1} J_{12}`,

$$(J^{-1})_{-g, g} = -\frac{1}{c} J_{11}^{-1} J_{12}, \qquad (J^{-1})_{g, g} = \frac{1}{c}.$$

Thus

$$\Delta z^{\text{add}}_{-g} = -\frac{\kappa \, z_g^*}{c} J_{11}^{-1} J_{12}, \qquad \Delta z^{\text{add}}_g = \frac{\kappa \, z_g^*}{c}.$$

### 4.3 When do they agree?

Compare component-wise:

| Component | Additive | Intervention | Ratio |
|---|---|---|---|
| Non-`g` | `-κ z_g* J_{11}⁻¹ J_{12} / c` | `κ z_g* J_{11}⁻¹ J_{12}` | `-1/c` |
| `g` itself | `κ z_g* / c` | `-κ z_g*` | `-1/c` |

**The two models agree exactly if and only if `c = -1`.** For a diagonal `J` with target-gene self-decay rate `J_22 = -1` (unit timescale), `c = -1` trivially and the models coincide. For any other value of `c`, additive and intervention differ by a fixed multiplicative factor `-1/c` that does not depend on `κ`.

### 4.4 The bias is κ-independent

**Numerical verification** (in-repo). On a small synthetic system (`d=4`, target `g=2`, non-normal `J`, Schur complement `c = -1.4082` computed directly, so predicted ratio `-1/c = 0.7101`), the empirical per-component ratio `Δz^add / Δz^int` matches the predicted `-1/c` **exactly** at every κ tested:

```
κ=0.001:  add/int per-component ratio = [0.7101, 0.7101, 0.7101, 0.7101]
κ=0.01:   add/int per-component ratio = [0.7101, 0.7101, 0.7101, 0.7101]
κ=0.1:    add/int per-component ratio = [0.7101, 0.7101, 0.7101, 0.7101]
κ=0.5:    add/int per-component ratio = [0.7101, 0.7101, 0.7101, 0.7101]
```

The Frobenius relative difference `||Δz^add - Δz^int||_F / ||Δz^int||_F = |1 - (-1/c)| ≈ 0.29` in this example is also κ-independent.

**Consequence:** the additive-input fit's bias does not vanish at small κ. Restricting to a weak-perturbation subset does not resolve the mismatch. This corrects an intuitive but wrong statement in early drafts of `SPEC.md`.

### 4.5 Empirical bias magnitude

On synthetic gene-space data with a realistic `d=8`, non-normal `J`, and known intervention-model ground truth, fitting under the additive model gives:

- Frobenius error `‖J_hat - J‖_F / ‖J‖_F ≈ 0.45`
- Leading real eigenvalue: `Re(λ_max)^true = -1.17` → additive-fit `Re(λ_max)^hat = -0.69`

The bias systematically **shifts eigenvalues toward zero** relative to the intervention-model truth. Practical implication: an additive-input fit that reports `Re(λ_max) ≈ 0` (weakly hyperbolic or weakly stable) is *consistent with* a truly-damped operator whose eigenvalues were pushed toward the imaginary axis by the model mismatch. Sign claims near zero are the fragile regime.

---

## 5. Program-space intervention model — under-identified from projected observations

*This is a negative result: proves that a naive fix for the additive-vs-intervention mismatch does not work in the same coordinate system.*

Setup: assume the gene-space Jacobian has the low-rank structure

$$J^{\text{gene}} = W J^{\text{prog}} W^T,$$

where `W` is the (orthonormal) program basis and `J^prog` is the `d × d` program-space Jacobian. This is the natural assumption if the dynamics live in the span of the programs.

### 5.1 Intervention response with rank-`d` `J^gene`

For gene *g* clamped at `(1-κ) z_g*`, apply the derivation of §4.1:

$$J_{11} = J^{\text{gene}}[-g, -g] = W_{-g} J^{\text{prog}} W_{-g}^T, \quad J_{12} = W_{-g} J^{\text{prog}} W_g,$$

where `W_{-g}` is `W` with row *g* removed and `W_g` is row *g* as a `d`-vector.

The intervention equation `J_11 x = κ z_g* J_12` in the response `x = Δz^{-g}` has solutions of the form `x = W_{-g} y` for some `y ∈ R^d` (assuming responses live in `span(W)`). Substituting:

$$W_{-g} J^{\text{prog}} W_{-g}^T W_{-g} y = \kappa z_g^* W_{-g} J^{\text{prog}} W_g.$$

Using `W^T W = I` (orthonormal) → `W_{-g}^T W_{-g} = I - W_g W_g^T`, and assuming `J^prog` invertible, this reduces to

$$(I - W_g W_g^T) y = \kappa z_g^* W_g,$$

which by the Sherman-Morrison identity has solution

$$y = \frac{\kappa z_g^*}{1 - \|W_g\|^2} W_g.$$

Therefore `Δz_{-g} = W_{-g} · κ z_g* W_g / (1 - ‖W_g‖²)`, and `Δz_g = -κ z_g*`.

### 5.2 The projected response is identically zero

Project back to program space: `Δz^prog = W^T · Δe`. Splitting into the *g*-row and the rest,

$$W^T \Delta e = W_g \cdot (-\kappa z_g^*) + W_{-g}^T \cdot \Delta e_{-g}.$$

Substituting `Δe_{-g} = W_{-g} · κ z_g* W_g / (1 - ‖W_g‖²)` and using `W_{-g}^T W_{-g} = I - W_g W_g^T`,

$$W^T \Delta e = -\kappa z_g^* W_g + (I - W_g W_g^T) \cdot \frac{\kappa z_g^*}{1 - \|W_g\|^2} W_g.$$

Simplify: `(I - W_g W_g^T) W_g = W_g - W_g \|W_g\|^2 = W_g (1 - \|W_g\|^2)`. Therefore

$$W^T \Delta e = -\kappa z_g^* W_g + \frac{\kappa z_g^* (1 - \|W_g\|^2)}{1 - \|W_g\|^2} W_g = -\kappa z_g^* W_g + \kappa z_g^* W_g = 0.$$

**The projected intervention response is identically zero, independent of `J^prog`.**

### 5.3 Numerical verification

On three random choices of `J^prog` with `n_genes = 20`, `d = 5`, orthonormal `W`:

```
trial 0: max ||W^T Δe|| over all g = 2.68×10⁻¹⁵
trial 1: max ||W^T Δe|| over all g = 4.88×10⁻¹⁵
trial 2: max ||W^T Δe|| over all g = 4.70×10⁻¹⁵
```

All at machine precision, confirming the analytic result.

### 5.4 Implication

The natural rank-`d` program-space intervention model is exactly under-identified from program-space observations. **You cannot fit an intervention-model `J^prog` from projected data alone under the standard rank assumption.**

This is a genuine methodological obstruction, not a numerical accident: the projection commutes with the intervention geometry in exactly the way that annihilates the observable signal. Escaping this requires one of:

- Working in **full gene space** with a non-low-rank `J^gene` (adds `O(G²)` parameters — computationally hopeless at genome scale).
- Adding **external constraints** (sparsity, ChIP-seq / ATAC-seq priors on which entries of `J` are nonzero).
- Combining multiple perturbation modalities (CRISPRi + CRISPRa) whose response geometries differ.

None of these are shipped in the current release. The additive-input model, with its documented bias, is the tool's operating point.

---

## 6. Null-corrected linearity check

*Code: `src/anchorop/measure.py::linearity_check`, `_split_rel_diff`.*

### 6.1 The naive linearity check

Split guides into weak- and strong-efficiency halves by median. Fit an operator for each half:

$$J_A = -U_A S_A^+, \quad J_B = -U_B S_B^+.$$

Compare on their common identified subspace `P_common`:

$$\text{rel\_diff} = \frac{\|(J_A - J_B) P_{\text{common}}\|_F}{\tfrac{1}{2}(\|J_A P_{\text{common}}\|_F + \|J_B P_{\text{common}}\|_F)}.$$

The symmetric denominator matters — an asymmetric normalization would depend on which half is arbitrarily "reference" and would make the null distribution below dependent on labeling. Documented in the code as a specific design choice.

### 6.2 What perfect linearity predicts

**Under perfect linearity with noiseless observations from a single operator `J`, and identical target sets in both halves:** `J_A = J_B = J` on their common subspace, and `rel_diff = 0`. Numerical validation: on noiseless synthetic data, random 50/50 splits give `rel_diff = 0` to machine precision.

**Under perfect linearity with realistic data:**
- Different guide halves sample different columns of `J`. Even if `J` is the same operator, `J_A` and `J_B` are estimates of *different projections* of `J`. Their disagreement on the common subspace reflects **finite-sample variance + biological heterogeneity across guides**, not linearity failure.

### 6.3 The random-split null

Draw `n_null` random 50/50 splits of the same guide set. For each split, compute `rel_diff` using the same procedure. This distribution captures:

- Finite-sample variance
- Biological heterogeneity across guides (different targets → different sub-operators)
- Any dataset-specific correlation structure

By construction, the random-null distribution is **agnostic to whether the split correlates with κ**. It measures only the noise floor of the diagnostic itself.

Let `μ = median(null_distribution)` and `σ = std(null_distribution)`.

### 6.4 Excess above null

The interpretable quantity is

$$\boxed{\text{excess} = \text{rel\_diff}_{\text{obs}} - \mu, \qquad z = \frac{\text{rel\_diff}_{\text{obs}} - \text{mean(null)}}{\sigma}.}$$

`excess > 0` means the observed disagreement is larger than what a random split of the same guide set produces. This is the "linearity failure signal" — the part where efficiency-based binning reveals a *systematic* difference (dose-response nonlinearity, model mismatch) that random binning does not.

`excess ≤ 0` means the observed disagreement is within the random-null distribution. No linearity failure signal above the finite-sample floor.

### 6.5 The passed criterion

Preregistered: `excess ≤ 0.25` → pass. This threshold is set *before* any real-data analysis and cannot be relaxed post-hoc. See `PREREGISTRATION.md`.

**Empirical result on Replogle 2022 essential-gene screens:**

| Dataset | `rel_diff` | `μ` (null median) | `σ` (null std) | `excess` | `z` | Pass? |
|---|---:|---:|---:|---:|---:|:---:|
| K562 essential | 1.466 | 1.353 | 0.034 | **+0.113** | +3.34 | **✓** |
| RPE1 essential | 1.571 | 1.384 | 0.035 | **+0.187** | +5.46 | **✓** |

Both cell lines pass the null-corrected criterion despite having raw `rel_diff` values that would look catastrophic (5-6× the naive threshold) if not null-corrected. The excess is small (about 8-12% of the raw value) but statistically significant (z > 3), consistent with the additive-vs-intervention bias direction documented in §4.

---

## 7. Archetype geometry

*Code: `src/anchorop/archetypes.py::fit_archetypes`, `src/anchorop/_utils.py::project_to_simplex`.*

### 7.1 Simplex-constrained reconstruction

Given a collection of `n` measured operator features `{x_i ∈ R^p}` (either symmetric-part matrix reshaped to `p = d²`, or spectral summary of dimension `p = 2d`), archetypes seek a small set of extremal profiles `{A_j ∈ R^p}_{j=1}^k` such that every state is expressible as a convex combination:

$$x_i \approx \sum_{j=1}^k w_{ij} A_j, \qquad w_{ij} \ge 0, \; \sum_j w_{ij} = 1.$$

The simplex constraint on `w` distinguishes archetypes from PCA (which allows arbitrary signed coefficients) and from NMF (which requires only non-negativity, not sum-to-one). Geometrically, every state lies in the convex hull of the archetypes.

### 7.2 Simplex projection

Given a candidate weight vector `v ∈ R^k`, project to the probability simplex `{w ≥ 0, sum(w) = 1}` by

$$w_j = \max(0, \; v_j - \theta),$$

where `θ` is chosen so that `sum_j w_j = 1`. This has a closed-form solution:

1. Sort `v` in descending order: `v_(1) ≥ v_(2) ≥ ... ≥ v_(k)`.
2. Compute cumulative sums `c_r = v_(1) + ... + v_(r)`.
3. Find largest `ρ` such that `v_(ρ) > (c_ρ - 1) / ρ`.
4. Set `θ = (c_ρ - 1) / ρ`, then `w_j = max(0, v_j - θ)`.

This is the Euclidean projection onto the simplex; standard result [Duchi et al., ICML 2008].

Numerical verification: for random `v ∈ R^6`, the projection always sums exactly to 1 with non-negative entries (validated in repo).

### 7.3 Farthest-point traversal (FPT) for observed-extreme archetypes

Rather than fit arbitrary points as archetypes (Cutler-Breiman archetypal analysis), we select **observed** extremes via farthest-point traversal:

1. Center the data. Start with the point farthest from the centroid as archetype 1.
2. For `j = 2, ..., k`: pick the data point maximally distant from the closest of the previously-chosen archetypes.

Advantages over Cutler-Breiman:
- **Deterministic**: no local minima from alternating-optimization.
- **Observed-extreme**: archetypes are actual measurements, not synthesized profiles.
- **Fast**: `O(n · k)`.

Disadvantages: less flexible; may miss archetypes that lie in the *interior convex hull* rather than on the observed extremes. Acceptable trade-off for the current small-`n` (2-10 cell states) regime.

### 7.4 Cross-validation for `k`

Selecting the number of archetypes cannot use an information criterion (AIC/BIC) because at `d = 30` the criterion overfits: with `n` states and `p = d²` features, the model has many more parameters than the state count constrains.

Instead: **guide-held-out equation residual**. For operator archetypes, hold out a subset of guides from each state's measurement, fit the archetypes on the remaining guides' operators, and evaluate on the held-out guides using the residual `‖J_reconstructed · S_test + U_test‖_F`. Select the `k` that minimizes this held-out residual.

This is a stricter test than in-sample reconstruction: it requires the archetypes to explain *new perturbation responses*, not just the fitted operator matrices.

### 7.5 Transfer test

Fit archetypes on source-state operators; project target-state operators onto the source-fitted simplex; compare with a direct target-state refit at the same `k`. Small transfer error / refit error ratio means the source vocabulary generalizes.

On the currently available two essential-gene measurements (K562 essential + RPE1 essential), the transfer test is under-powered (only 2 states, need ≥ 4 for source/target split with `k = 2+`). What we can report: the CV rule selects `k = 1` — a single common archetype describes both states, consistent with cell-line-invariant essential-gene operator geometry at the linear-response idealization.

---

## Postscript — open problems

Three math-shaped open problems the current release does not solve:

1. **Fitting the intervention model in program space.** Ruled out at the natural rank-`d` assumption in §5. Requires either full gene-space fitting with sparsity priors, or hybrid additive+intervention observations with additional structural constraints.

2. **Analytic bias correction for the additive-input model.** The bias factor `-1/c` from §4 depends on the Schur complement of `J`, which itself is what we're trying to measure. An iterative correction (fit `J` under additive, compute implied `c`, back-correct) has not been analyzed for stability.

3. **State-dependent Jacobians.** The tool assumes a single `J` per measurement (per cell state). Cells with diverse baseline expressions have different local `J`'s. A per-cell operator field would require per-cell response estimates (currently aggregated per guide) and a smoothness prior over the cell manifold.

All three are legitimate research directions rather than fixable oversights. Each would extend the tool's scope substantially; each would also require its own identifiability analysis.

---

## Appendix: Why `poisson_mle` is biased downward under independent zero-inflation

Let `X ~ Poisson(λ)` be the noise-free count model and `Y = X · B` where `B ~ Bernoulli(1 − π)` is an independent dropout indicator: with probability `π`, the count is masked to zero regardless of the underlying `λ`. The detection rate is `Pr[Y > 0] = (1 − π) · (1 − e^{−λ})`.

The `poisson_mle` estimator uses `λ̂ = −log(1 − Pr[Y > 0])`, i.e., it treats the observed detection rate as if it came from a pure Poisson without dropout. Substituting the true detection rate:

```
λ̂ = −log(1 − (1 − π)(1 − e^{−λ})) = −log(π + (1 − π) e^{−λ}).
```

Taylor-expanding in `π` around `π = 0` (small dropout):

```
λ̂ = −log(e^{−λ} + π (1 − e^{−λ})) = λ − log(1 + π (e^λ − 1)) ≈ λ − π (e^λ − 1).
```

For any `λ > 0`, the correction term `π (e^λ − 1)` is strictly positive, so **`λ̂ < λ` under zero-inflation**: `poisson_mle` systematically underestimates `λ` when there is a nonzero dropout probability independent of `λ`. Applied to `κ̂ = 1 − λ̂_pert / λ̂_ctrl`:

- If dropout `π` is the same in perturbed and control conditions, both `λ̂` values are biased in the same direction. To first order in `π`, `λ̂_ctrl ≈ λ_ctrl − π(e^{λ_ctrl} − 1)` and `λ̂_pert ≈ λ_pert − π(e^{λ_pert} − 1)`. Since `λ_pert = (1 − κ) λ_ctrl < λ_ctrl`, the correction on `λ̂_ctrl` is larger in absolute terms, so `λ̂_pert / λ̂_ctrl > λ_pert / λ_ctrl = 1 − κ`, and therefore `κ̂ = 1 − λ̂_pert/λ̂_ctrl < κ`.
- Contrast with `mean_ratio`: `E[Y_ctrl] = (1 − π) λ_ctrl`, `E[Y_pert] = (1 − π)(1 − κ) λ_ctrl`, so the ratio `E[Y_pert] / E[Y_ctrl] = 1 − κ` regardless of `π`. The dropout fraction cancels in the moment ratio, and `mean_ratio` is unbiased.

This is the analytic underpinning for §2.3's recommendation of `mean_ratio` over `poisson_mle` on count data with dropout, and for the `poisson_mle` docstring in `src/anchorop/measure.py` noting that the estimator is "conservative under independent zero-inflation" (attributes some structural zeros to Poisson, understates `κ`).
