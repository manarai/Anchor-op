# Additive-input encoding fails target-held-out prediction on genome-scale Perturb-seq screens

**Authors.** *TO BE FILLED IN BEFORE SUBMISSION.*
**Affiliations.** *TO BE FILLED IN BEFORE SUBMISSION.*
**Corresponding author.** *TO BE FILLED IN BEFORE SUBMISSION.*

---

## Abstract

Pooled Perturb-seq screens measure how cells respond to hundreds of genetic knockdowns, and a common modeling step reads these responses through a linear settled-state model with an additive input for each perturbed gene. Under that model the response operator is identifiable in closed form, `J·P_X = −U·S⁺`. We tested whether it predicts held-out perturbations in three screens: Replogle K562 and RPE1 essential-gene screens, and the titrated Jost 2020 screen. We used target-held-out prediction with nested cross-validation and compared each screen against a positive control, a linear ground truth simulated at the same signal-to-noise ratio, inputs and noise. The model failed on all three. Held-out error stayed at the predict-zero baseline (ρ = 0.96 with 5-fold standard deviation 0.02; 1.00 with SD 0.006; 1.00 with SD 0.002 for K562, RPE1, and Jost at d = 30), while the matched linear truth reached 0.18, 0.53, and 0.72. The failure persists when targets are sampled at random (K562 real ρ = 1.00, SD 0.009 vs matched linear 0.43), when input dimension is reduced so the problem is well overdetermined (Jost d = 5 real ρ = 1.01 vs matched linear 0.32), and when perturbation strength is removed from the fit by column-normalizing the design, so mis-estimated knockdown efficiency cannot explain it. It is also not a noise problem: at the observed signal amplitude, the same estimator recovers a known operator's interactions (interaction-only Frobenius cosine 0.98 [K562] and 0.74 [RPE1] against a shift-1 cross-replicate null near zero). On Jost's titrated design the model partly interpolates dose along known target directions (guide-level ρ = 0.66 against 0.22 for the matched linear-truth control on the same folds). Whether the failure comes from projecting targets onto expression programs or from nonlinear dose responses remains open. anchor-op releases the matched controls and an identifiability gate that checks the rank of the inputs as well as the responses.

---

## 1. Introduction

Pooled Perturb-seq screens deliver an intervention→response map at genome scale by pairing CRISPR-based perturbations with single-cell RNA-seq [1]. A common modeling step reads these responses through a linear settled-state model of regulatory dynamics: knocking down gene *g* at efficiency `κ_g` shifts program-space state by `Δz_g = −J⁻¹u_g`, where the perturbation input is `u_g = −κ_g Wᵀδ_g`, with `W` a control-derived program basis and `δ_g` a one-hot on gene *g*. Under this additive-input steady-state assumption, stacking guides gives a sensitivity matrix `S = −J⁻¹U`, and regularized inversion returns the operator's action on the identified response subspace, `J·P_X = −U·S⁺` with `X = range(S)`. Because this action is closed-form, the pipeline that computes it can be run on any Perturb-seq screen with a defined program basis and a calibrated efficiency estimator.

Work in this area has focused on whether the linear-response assumption holds and how many cells per guide might suffice to establish it. Both are secondary. The primary question is whether the fit predicts held-out perturbations. Answering it requires a fold-honest evaluation: fit `A = −U·S⁺` on training guides, predict `U_test = −A·S_test` on held-out guides, and compare the resulting held-out ρ to a positive control that runs the same procedure on a linear ground truth simulated at the dataset's own guide inputs, efficiency, and per-entry noise. Under any additive-input linear model the positive control must beat the predict-zero baseline of ρ = 1; otherwise the identifiability calculation is not informative on the data at hand.

We ran that control on three current Perturb-seq screens: the essential-gene libraries in Replogle et al. 2022 (K562 and RPE1 lines) [1] and the titrated dose-response library in Jost et al. 2020 [2]. On all three, the fitted encoding fails to beat the predict-zero baseline on held-out prediction, while the matched-SNR linear-truth control does. The failure is not a noise-level artifact: at the observed signal amplitude the same estimator on a linear ground truth cleanly recovers the interaction structure of the operator. It is not created by the widely used convention of selecting the 200 targets with the most cells: a random-200 K562 refit reproduces the failure. It is not created by identifiability under-determination on Jost's smaller target set: a d-sweep truncating the coordinate system to a comfortably overdetermined regime leaves real held-out ρ at the predict-zero floor. And it is not created by mis-scaling of the perturbation-strength estimate: column-normalizing the design before the fit removes all per-guide scale that a κ error could carry, and held-out ρ does not fall on any of the three screens.

Two candidate explanations remain live. Under the additive-input encoding, projecting the target-direction one-hot onto d = 30 control-derived programs leaves ~99% of the target direction outside the retained subspace, so the encoded input may carry little of the response-relevant signal — a *projection* failure. Independently, the linear settled-state assumption may be the wrong shape for saturating or threshold-dominated dose responses that govern actual regulatory dynamics — a *dose* failure. We describe the tests that would distinguish them and leave both to future work.

Two subsidiary correctness issues bear on any tool in this space and are handled in the software. Eps-based numerical rank on collinear guide libraries can silently declare full rank; the identifiability gate now refuses to declare full-domain recovery unless both the response and the input have full rank. And knockdown-efficiency estimators can be applied outside their valid data format; the efficiency router chooses `mean_ratio` on count data and a `detection_rate` proxy score on pre-scaled residual data, with the second labelled as a signed distributional shift rather than a fractional-knockdown estimate.

**anchor-op** implements the pipeline, the identifiability discipline, the efficiency-routing regime, the matched-geometry positive control, and the target-held-out nested cross-validation as a reusable set of diagnostics. All numeric claims in this manuscript are round-form values; the exact values, the JSON they were computed from, and the script that produced each are listed in a machine-readable provenance table (Code availability).

---

## 2. Results

### 2.1 Target-held-out prediction fails against the matched linear control

For each screen we fit the additive-input operator `A = −U·S⁺` using truncated SVD, with the retained rank chosen by nested cross-validation inside each outer training fold (Methods). Both outer and inner folds are target-grouped, so no sibling sgRNA leaks between train and test. The rank grid includes zero, which represents the predict-zero baseline `A ≡ 0` and gives held-out ρ = 1 exactly; nested CV therefore selects "shrink to zero" whenever the fitted model does worse than predict-zero on inner validation.

We report pooled held-out ρ with its across-outer-fold standard deviation on the real data, and mean ± SD across 15 independent matched-SNR linear-truth simulations. Table 1 summarizes the result on the three screens as originally released.

**Table 1. Target-held-out held-out ρ under nested cross-validation.**

| Screen (n retained sgRNAs / n targets) | Real ρ (5-fold SD) | Median picked TSVD rank | Matched linear-truth ρ (mean, SD; N = 15) |
|---|---:|---:|---:|
| Replogle K562 essential (188 / 188) | 0.96 (0.02) | 3 | 0.18 (0.003) |
| Replogle RPE1 essential (153 / 153) | 1.00 (0.006) | 1 | 0.53 (0.016) |
| Jost 2020 (122 / 25) | 1.00 (0.002) | 0 | 0.72 (0.043) |

Every real ρ sits at or above the predict-zero baseline within one fold-level SD. On K562 the median picked rank is 3; on RPE1 it is 1; on Jost it is 0 — nested CV picks "shrink to zero" as the best per-fold model, so the fitted operator loses to no model at all. Under the same estimator on the same U and per-entry σ, a matched-SNR linear ground truth reaches held-out ρ well below the predict-zero baseline in every case.

The three screens differ in library size and design. Replogle-essential libraries retain one sgRNA per target after the released target-level aggregation step. Jost carries several sgRNAs per target at graded activities, spanning `κ ∈ [0.07, 1.00]` with count-based estimates from `mean_ratio` on UMIs — roughly twice Replogle's `κ` range in a design purpose-built for dose response — and 25 targets total. Their per-entry noise anchors, estimated from within-guide cell-level split-half bootstraps in each screen's own basis, are 0.24 (K562), 0.35 (RPE1), and 0.066 (Jost, per sgRNA). The failure is not restricted to a specific library size, design, or noise level.

### 2.2 The failure is not a noise problem

At the amplitude of the released simulation ensemble the noise-free response is much smaller than the observed response — a factor of ~360 on K562 by median column norm. At that amplitude the signal-to-noise ratio is low and any TSVD estimator, linear-truth or not, has almost no chance of recovering the operator. A signal-scale sweep varying the multiplier `α_S` on the noise-free response confirms this and identifies the α_S that reproduces the observed median column norm of `S`.

At the data-matched amplitude the same TSVD estimator on a linear ground truth cleanly recovers the operator: full-matrix Frobenius cosine 0.99 (K562) and 0.91 (RPE1) at N = 200 replicates. Those full-matrix cosines are dominated by the shared diagonal shift carried by every replicate in the ensemble; their shift-1 cross-replicate null is around 0.70. The *interaction-only* cosine — computed after subtracting the diagonal from both `A` and `J` — isolates the recoverable structure. It reaches 0.98 (SD 0.003) on K562 and 0.74 (SD 0.023) on RPE1 at matched amplitude, against a shift-1 cross-replicate null of 0.003 in both cases (N = 200).

The estimator can recover a known operator's interactions at the signal amplitude of the observed response. The gap between "an estimator that recovers a linear operator's interactions at matched SNR" and "an estimator that fails held-out prediction on real data at the same SNR" is the point. Under the additive-input encoding, the model fits `A` and predicts `U = −A·S` on held-out guides. On the linear-truth positive control that prediction succeeds; on the real data it does not. The failure lies somewhere between the data-generating process and the additive-input model, not in the fitting machinery.

### 2.3 The failure is not target selection, input dimension, or κ mis-scaling

**Selection bias.** The publicly released essential-gene notebooks pick the 200 targets with the most cells after a minimum-cell filter. In essential-gene screens, high cell count correlates with mild fitness defect, so the selection preferentially retains weakly-affected targets. A K562 refit on 200 targets sampled uniformly at random from the 1,740 qualifying targets (all with ≥ 60 cells) gives real nested-CV ρ = 1.00 (5-fold SD 0.009), median picked rank 0, against a matched linear-truth ρ = 0.43 (SD 0.011, N = 15). The failure survives the random selection.

**Design underdetermination.** Jost has 25 target directions and d = 30 program dimensions; under 5-fold target-grouped CV each outer training set spans only ~20 target directions, which is fewer than d. If the encoding failed on Jost only because of this under-determination, the failure should disappear when the coordinate system is truncated to a smaller d where training targets comfortably exceed d. We swept d ∈ {5, 10, 15, 20, 25, 30} on Jost, recomputing α_S at each d so the matched linear-truth control tracks the same signal amplitude at each point.

**Table 2. Jost d-sweep under target-grouped nested cross-validation.**

| d | Real ρ (pooled) | Median picked rank | Matched linear-truth ρ (mean, SD; N = 15) | Regime |
|---:|---:|---:|---:|:---|
| 5 | 1.01 | 0 | 0.32 (0.038) | overdetermined |
| 10 | 1.00 | 0 | 0.85 (0.067) | overdetermined |
| 15 | 0.99 | 2 | 0.46 (0.031) | overdetermined |
| 20 | 1.00 | 0 | 0.55 (0.061) | overdetermined |
| 25 | 1.00 | 1 | 0.69 (0.037) | underdetermined |
| 30 | 1.00 | 0 | 0.73 (0.043) | underdetermined |

At d = 5 the identifiability regime is comfortably overdetermined and the matched linear truth reaches its cleanest ρ (0.32), yet real Jost ρ is 1.01. The encoding fails independent of the identifiability regime.

**κ mis-scaling.** The Replogle-essential pipeline uses a residual-space `detection_rate` proxy for κ (Methods), which is a signed distributional-shift statistic rather than a fractional-knockdown estimate. If mis-scaled per-guide κ were the source of the failure, column-normalizing S and U before the fit — which removes all per-guide scale from the fitting problem — should improve held-out ρ. It does not. On Replogle the transform barely moves ρ: baseline ρ = 1.11 → 1.09 on K562 (Δρ = −0.03) and 1.21 → 1.13 on RPE1 (Δρ = −0.08). Column-normalizing a matched linear-truth control on the same design gives ρ ≈ 0.17 on both, so the direction-only estimator itself is not the source of the ρ ≈ 1 outcome. On Jost the same transform *worsens* the direction-only ρ, which means Jost's count-based κ carries real scale information the transform discards. κ quality differs qualitatively between the two designs — Replogle's residual-space proxy is uninformative; Jost's count-based `mean_ratio` is informative — but κ mis-scaling cannot on its own account for the encoding's failure on either.

### 2.4 The pipeline captures within-target dose interpolation on Jost

Jost 2020 pairs each of its 25 targets with 3–6 sgRNAs at graded activities. Under target-held-out prediction (the setting in §2.1–2.3) a held-out target's sgRNAs have no siblings in training, and the model is asked to predict a new perturbation direction. Under *guide-level* folds — where every held-out sgRNA has siblings for its own target in training — the model is instead asked to interpolate amplitude along a *known* direction. It predicts held-out `U_test` given `S_test` along that known direction. That interpolation ρ is 0.66 on real Jost, against 0.22 (SD 0.008, N = 15) for the matched linear-truth control on the same guide-level folds.

Two things follow. The pipeline does capture within-target amplitude structure when a target's direction is available: 0.66 is meaningfully below the predict-zero baseline of 1 and well above the null. But the amplitude the pipeline captures is not operator recovery. Under any additive-input linear model, every guide for one target moves the system along the same direction — `κ_g` rescales columns, `Wᵀδ_g` is fixed — and the guide-level fold split reduces the prediction task to one that a per-target one-parameter fit could pass. We label the guide-level Jost result "within-target dose interpolation along a known target direction" and keep it separate from the target-held-out claim that governs the main result.

### 2.5 Two commonly used diagnostics are uncalibrated at these signal-to-noise ratios

Two summary diagnostics widely used with additive-input pipelines are not calibrated to the noise levels of current Perturb-seq screens.

**A bin-split linearity threshold.** The `rel_diff` statistic splits guides at the median efficiency, fits an operator per half, and returns the Frobenius-normalized difference on the shared identified subspace. A prespecified threshold `rel_diff ≤ 0.25` was chosen at preregistration to declare the additive-input model consistent with the data. Real data on Replogle gives `rel_diff` = 1.47 (K562) and 1.57 (RPE1); on Jost it is 1.26. Every screen fails the threshold. But so does a matched linear ground truth at each screen's own signal amplitude and noise: 0.33 on K562, 0.98 on RPE1, and 1.05 (SD 0.03) on Jost. On RPE1 and Jost even a perfectly linear generator fails the 0.25 threshold; only on K562 does it drop below 1.0. The threshold cannot separate a linear model from a nonlinear one at these signal-to-noise ratios regardless of ground truth. Under the target-held-out nested-CV comparison in §2.1 the linear truth reaches ρ well below 1 on all three screens; the diagnostic failure is a threshold-calibration issue, not a data verdict.

**A single-scalar condition number.** A common summary of an additive-input measurement is the condition number of the retained response matrix under the preregistered `rank_tol = 1×10⁻²`. On Jost this number is small (~3.4) at the screen's own noise, which reads as "well-conditioned" and would ordinarily support downstream operator interpretation. It is misleading. At Jost's per-sgRNA noise the noise budget per column (`σ · √d ≈ 0.36`) exceeds the noise-free per-column response norm (~0.05). A noise-dominated matrix has a random-matrix singular-value distribution and an O(1) condition number for the same reason a Ginibre matrix does — the small condition number reflects the noise, not the operator. On the noise-free response the same truncation gives condition ~65; on the input alone, ~75. The identifiability gate we ship now refuses to declare "full-domain identified" unless both the input rank and the retained response rank equal `d` at the preregistered tolerance. Jost has input rank 24 at that tolerance, so the gate now correctly labels the Jost fit as partially identified, and the condition-number summary should be interpreted alongside the input and response ranks separately rather than as a single scalar.

The general point: uncalibrated single-number diagnostics can mislead at Perturb-seq signal-to-noise ratios. The calibrated form is the paired comparison of real held-out ρ against a matched-SNR linear-truth control under the same fitting recipe (Table 1).

---

## 3. Discussion

### 3.1 What the result says

Under the additive-input encoding `u_g = −κ_g Wᵀδ_g` and truncated-SVD fitting on three current Perturb-seq screens, target-held-out prediction of a new perturbation does not beat the predict-zero baseline held-out ρ = 1, while a matched-SNR linear-truth positive control on the same U, κ, and per-entry noise reaches ρ well below 1 on every screen. The failure is not a noise-level artifact, is not created by the top-cell-count target selection convention, is not created by identifiability under-determination on Jost's smaller target set, and is not created by mis-scaling of the κ proxy on Replogle or by the count-based κ on Jost.

Two positive findings survive from the same measurements. Under the same matched-SNR positive control, the fitting machinery cleanly recovers the interaction structure of a known operator: interaction-only Frobenius cosine 0.98 on K562 and 0.74 on RPE1, against a shift-1 cross-replicate null of ~0.003. On Jost's titrated design the pipeline captures within-target dose interpolation along a known target direction: guide-level ρ = 0.66 against 0.22 for the matched linear-truth control on the same folds. Neither is target-held-out recovery of the operator; both are legitimate summaries of what the pipeline does capture at these amplitudes.

The result is strongly incompatible with reading fitted spectra or edges as quantitatively estimated operators on the tested screens under the tested encoding. It does not rule out recovery under an alternative input encoding, an alternative model class, a lower-dimensional target, a stronger structural prior, or a different assay design.

### 3.2 What is not yet decided

Two candidate mechanisms are consistent with the data, and neither is decided by what we ran.

**Projection failure.** The additive-input encoding pushes a target-direction one-hot through a d = 30 control-derived program basis. The projected input `Wᵀδ_g` has median column norm ~0.09 on the essential-gene screens — about 99% of the target direction lies outside the retained subspace. If the target's response-relevant direction is not among the top d control-derived programs, the encoded input carries little response-relevant signal and the fitting task becomes ill-posed on target-held-out folds. A footprint encoding `u_g ∝ Wᵀ Σ_ctrl δ_g`, where `Σ_ctrl` is the control-cell gene-space covariance, projects the target's co-expression *neighbourhood* rather than the target alone; it would test whether the projection is the binding constraint.

**Dose non-linearity.** The linear settled-state assumption `S = −J⁻¹U` may not describe the assay's actual dynamics if the response saturates or thresholds in dose. Under any additive-linear model, all sgRNAs for one target move the system along the same direction — κ rescales columns, the direction is fixed. Within-target guide-replicate direction concordance on Jost, compared against a between-target null and a within-guide split-half noise ceiling from the same cell-level bootstraps, tests this directly: strong concordance well above the between-target null and near the noise ceiling is evidence *for* the linear-direction assumption; weak concordance is evidence against it.

A footprint encoding rewrites the input and tests projection. A within-target concordance diagnostic tests the direction assumption. Both are cheap; neither was required to establish the primary claim.

### 3.3 Limitations

The result concerns three screens, one encoding class, and one estimator family. Structural alternatives — a diagonal-only, block-modular, symmetric, or GRN-constrained operator; a much smaller d; a footprint or `Σ_ctrl`-weighted encoding; a nonlinear dose-response family; a Perturb-seq design with true replicate targets and calibrated dose titration — were not tested and could behave differently. The identifiability calculation itself, a rank statement about `range(S)`, is correct as a matter of linear algebra and unaffected by the encoding failure.

Two per-screen caveats. Replogle-essential libraries retain one sgRNA per target after the released aggregation step, so within-target replicate diagnostics do not apply to them. Jost has 25 targets, small enough that a target-held-out CV places most of the design outside the training set and the fit sits close to under-determined at d = 30; the d-sweep controls for this at d = 5 but does not turn Jost into a general genome-scale test. Cross-screen quantitative comparison of specific ρ values requires care because per-cell noise structure differs across cell lines.

The additive-input model itself is a modelling choice. CRISPRi is closer to a clamp on target transcript than to an additive forcing term. Under a program-space clamp with orthonormal `W` and a rank-deficient `J_gene = W·J_prog·Wᵀ`, the intervention model is exactly under-identified from projected observations. This is a valid algebraic result under those assumptions rather than a general impossibility; alternative encoder/decoder conventions or explicit gene-space dynamics can change the conclusion.

We report every real held-out ρ with its across-outer-fold standard deviation and every simulated positive-control ρ with its cross-replicate standard deviation. Interpretation carries the same reservation: target-held-out held-out ρ on a real Perturb-seq screen is the quantity that a matched-SNR linear-truth positive control makes calibrated.

---

## 4. Methods

### 4.1 Framework

For expression `E ∈ ℝ^(n×G)` and a control-derived program basis `W ∈ ℝ^(G×d)` with orthonormal columns, program coordinates are `z = Wᵀe`. For a guide targeting *g* at efficiency `κ_g ∈ (0, 1]`, the perturbation input is `u_g = −κ_g Wᵀδ_g`; stacking *m* guides gives `U ∈ ℝ^(d×m)`. Under the additive-input, steady-state assumption `S = −J⁻¹U`, so `J·S = −U`. Right-multiplying by the pseudoinverse identifies the operator's action on `X = range(S)`:

```
J·P_X = −U·S⁺
```

`J` is not identified outside `X`. anchor-op returns this action and blocks access to the full `J` unless both the actuated input subspace `range(U)` and the retained response subspace `range(S)` have rank `d` at the preregistered tolerance (§4.2).

### 4.2 Regularization and the identifiability gate

`S⁺` is computed by truncated SVD, with the full regularization path retained. A singular direction counts as identified only if `σ_i > rank_tol · σ_max(S)`. The default `rank_tol = 1×10⁻²` was preregistered to prevent eps-based numerical rank from silently accepting below-noise directions on collinear guide libraries. The identifiability gate now labels a fit as full-domain-identified only when both `input_rank(U)` and the effective retained response rank equal `d` at the same tolerance; noise-inflated retained rank on rank-deficient `U` was previously accepted and is not. Regression tests cover both the noise-inflated failure mode and the heavy-Tikhonov collapse.

### 4.3 Efficiency estimation

Three estimators with an auto-router. `mean_ratio` (κ = 1 − mean_pert / mean_ctrl) is asymptotically unbiased under Poisson and consistent under independent zero-inflation. `poisson_mle` matches under pure Poisson. `detection_rate` (Pr[X_ctrl > 0] − Pr[X_pert > 0]) is not a κ estimator on count data but is a valid signed distributional-shift statistic on pre-scaled residual data. The `"auto"` router chooses `detection_rate` when ≥ 2% of entries are negative (the pre-scaled-residual signature), else `mean_ratio`. On Replogle-essential's pre-scaled residual matrix the router chooses `detection_rate`; on Jost 2020's UMI counts it chooses `mean_ratio`. Where the manuscript reports a per-screen `κ` distribution, Replogle values are the proxy score and Jost values are the count-based estimate; they are not commensurable on a common fractional-knockdown axis and are not compared as such.

### 4.4 Per-entry noise anchors

Within-guide cell-level split-half bootstraps on each screen: for each target with ≥ 20 cells, split cells into equal halves, compute half-Δz vectors `d₁, d₂` in that screen's own PCA basis fit on non-targeting controls, and take the median per-entry standard deviation `‖d₁ − d₂‖_F / (2√d)` across targets. This gives σ = 0.24 (K562), 0.35 (RPE1), and 0.066 (Jost, per sgRNA). The three anchors are used in every matched-SNR positive control (§4.6).

### 4.5 Target-held-out nested cross-validation

Outer folds: 5 target-grouped folds. All sgRNAs of a target are held out together, so no sibling sgRNA leaks between train and test. Inner folds within each outer training set: 3 target-grouped folds by the same rule. The TSVD rank is chosen per outer fold from the grid `{0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}` by minimizing pooled inner-MSE of `A·S_val + U_val`. `r = 0` returns `A ≡ 0` and gives outer-fold ρ = 1 exactly, which lets nested CV select "shrink to zero" whenever the fitted model does worse than predict-zero on inner validation. We refit at the selected rank on the full outer training set and report pooled ρ across outer folds with the across-outer-fold standard deviation.

The Jost d-sweep in §2.3 truncates S and U to the first `d' ∈ {5, 10, 15, 20, 25, 30}` rows before running the same recipe. The K562 random-200 refit in §2.3 rebuilds the entire pipeline (variance-based highly-variable-gene selection with the sampled target genes force-kept in the feature matrix, PCA on non-targeting controls at d = 30, per-guide response construction with the proxy-efficiency path used in the released essential-gene notebooks) on 200 targets sampled uniformly at random from the 1,740 targets with ≥ 60 cells.

### 4.6 Matched-SNR linear-truth positive control

For each screen we compute the observed median column norm of `S`. For each of 15 independent simulations, we draw `J_ref = G − 1.5·I` with `G` entrywise `N(0, 1/d)` and set `J = J_ref / α_S`, choosing `α_S` so the median column norm of the noise-free `S_true = −J⁻¹U` matches the observed median at each screen's own `U`. We add per-entry noise at the screen's bootstrapped σ (§4.4) and run the same nested-CV recipe (§4.5) on this synthetic `(S_obs, U)` pair. We report the mean and standard deviation of held-out ρ across the 15 simulations.

The interaction-only cosine reported in §2.2 uses the same synthetic ground-truth ensemble at matched α_S and reports Frobenius cosine after subtracting the diagonal from both `A` and `J`, with the shift-1 cross-replicate null as its baseline (N = 200 replicates). The signal-scale sweep in §2.2 varies α_S across a log grid and reports the cos-versus-α_S curve and the α_S implied by the observed median column norm.

### 4.7 Direction-only refit

Column-normalize `S` and `U` — divide each column by its own Frobenius norm — then run the same nested-CV recipe (§4.5). This removes all per-guide scale from the fitting problem. On the same design the matched-SNR linear-truth control is column-normalized in the same way and evaluated by the same recipe; both are reported in §2.3.

### 4.8 Preregistered vs post hoc

Preregistered: `rank_tol = 1×10⁻²`, TSVD regularization, `d = 30` PCA basis fit on non-targeting controls, the raw bin-split `rel_diff ≤ 0.25` threshold. Post hoc: per-screen σ anchors (§4.4, replacing an earlier common σ); the α_S signal-scale sweep and matched-α positive control (§4.6); the target-grouped nested-CV recipe and the rank-0 baseline (§4.5); the interaction-only cosine analysis (§4.6); the direction-only refit (§4.7); the K562 random-200 refit and the Jost d-sweep (both executed and reported in §2.3); the identifiability gate now requiring `rank(U) = d` as well as retained `rank(S) = d`, a bug fix affecting the identifiability label on rank-deficient inputs with no numerical-recovery changes. A preregistration amendment is shipped with the release.

### 4.9 Data availability

Replogle et al. 2022: gwps.wi.mit.edu / Figshare Plus deposit 20029387 (K562 essential sampled at day 6; RPE1 essential sampled at day 7) [1]. Jost et al. 2020: GEO GSE132080 [2].

### 4.10 Code availability

anchor-op is available at https://github.com/manarai/anchor-op under the MIT license. A pinned conda `environment.yml` (Python 3.11), a 59-test pytest suite, per-figure regeneration scripts, and every JSON that a numeric claim traces to are shipped with the release. Every numeric claim in this manuscript that is not derived in the text has an entry in `results/recheck/number_provenance.csv` giving the value stored in a JSON in `results/recheck/`, the reproduction script that produced it, and a note on rounding where applicable. Round-form numbers cited in the abstract and Results are drawn from the un-rounded values in that table.

## Author contributions

*TO BE FILLED IN BEFORE SUBMISSION.*

## Competing interests

*TO BE FILLED IN BEFORE SUBMISSION.*

## Acknowledgements

*TO BE FILLED IN BEFORE SUBMISSION.*

## Funding

*TO BE FILLED IN BEFORE SUBMISSION.*

## References

[1] Replogle JM, Saunders RA, Pogson AN, et al. (2022) Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq. *Cell* 185(14):2559–2575.e28.

[2] Jost M, Santos DA, Saunders RA, et al. (2020) Titrating gene expression using libraries of systematically attenuated CRISPR guide RNAs. *Nature Biotechnology* 38:355–364.
