# Preregistration amendment — 2026-09-27

`PREREGISTRATION.md` is untouched. This file records analysis
decisions made after preregistration, in response to the
external audit bundle.

Preregistration reference commit: whatever hash `PREREGISTRATION.md`
carried at the top of the branch when this recheck ran (see
`git log -- PREREGISTRATION.md`).

## Post hoc analyses

1. **Paired-bootstrap cross-replicate null** (10 000 resamples,
   shift = 1) — added in Round 2, applied here to cos_full and cos_1
   per dataset. Preregistration specified an empirical null but not
   the paired-bootstrap inference; declared post hoc.
2. **Per-dataset σ anchors** — K562 σ = 0.240, RPE1 σ = 0.352 (from
   Fig. S19). Replaces the original σ = 0.266 anchor. Declared post
   hoc.
3. **Signal-scale (α_S) sweep** — `reproduction/30_recheck_F1_F4.py`
   parameterizes the ground-truth Jacobian as `J_true = J_ref / α_S`,
   so that the noise-free response `S_true = α_S · (−J_ref⁻¹ U)`
   scales linearly with α_S. `α̂_data` puts median simulated ‖S‖ at
   the observed median. New in this audit; declared post hoc.
4. **F4 target-input orthogonality checks** — median `|cos(s_g, u_g)|`
   at chance, held-out ρ after mean-response projection removal. New
   in this audit; declared post hoc.
5. **F7 target-selection sensitivity** — random-200 and
   fitness-stratified-200 comparison scripts. New in this audit;
   declared post hoc. Not executed in this pass (see
   `reproduction/35_recheck_F7_selection_sensitivity.py`).

## Post hoc code changes

- **`src/anchorop/identifiability.py`** — the
  `full_domain_identified` gate now requires `input_rank == d` as
  well as `effective_response_rank == d`. Rationale: on Jost-like
  data (25 targets, 6 sgRNAs per target sharing `Wᵀδ_g`), rank(U) <
  d while a noisy S can still hit numerical rank d. The prior gate
  falsely declared full identification. New regression tests:
  `tests/test_full_domain_gate.py`. This is a bug fix, not a
  preregistration change to the estimator — no analysis result
  changes; only the identifiability label attached to Jost-shaped
  measurements changes.

## Non-changes

The preregistered decisions below are unchanged:
- `rank_tol = 1e-2`.
- Regularizer `reg = "tsvd"`, path selection.
- `d = 30` PCA basis fit on non-targeting controls.

## Amendment — footprint-encoding preregistration (2026-09-27)

This section is being committed **before any footprint-encoding code
is run**. The specifications below are locked in by this commit; the
run in `reproduction/45_footprint_encoding.py` and its outputs must
follow exactly.

**Primary encoding.**
`u_g = −κ_g · Wᵀ · Σ_ctrl · δ_g`, where:
- `δ_g` is the gene-space one-hot on the target `g`;
- `Σ_ctrl` is the gene–gene covariance of *non-targeting-control cells
  only*, computed in the same HVG feature space that fits the program
  basis, with target genes force-kept in the feature set;
- `W` is the same control-derived program basis used elsewhere;
- `κ_g` is the same efficiency estimate used elsewhere on each screen
  (Replogle `detection_rate` proxy; Jost `mean_ratio`).
No perturbed cells enter `Σ_ctrl`. `Σ_ctrl` for each screen is
recomputed from the h5ad the screen was originally fit from and saved
alongside the measurement pkl.

**Metric.**
Target-grouped nested cross-validation held-out ρ under exactly the
recipe of Table 1 in `MANUSCRIPT.md` §2.1: 5-fold target-grouped
outer folds; 3-fold target-grouped inner folds; TSVD rank chosen per
outer fold from the grid `{0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}`
(rank 0 = predict-zero baseline).

**Controls (all four run on every screen).**
- (a) *Current encoding baseline.* `u_g_current = −κ_g · Wᵀ · δ_g`,
  identical to Table 1. Reported here as the reference against which
  the footprint encoding is compared.
- (b) *Shuffled-footprint null.* At least 100 permutations of the
  target → footprint mapping (each replaces `Σ_ctrl · δ_g` with
  `Σ_ctrl · δ_{σ(g)}` for a random permutation `σ`). For each
  permutation, refit the nested-CV pipeline and record pooled ρ. The
  full null distribution is retained; success is judged against its
  2.5th percentile.
- (c) *Random-direction null.* At least 100 draws of random unit
  vectors `r_g ∈ ℝ^d` for each guide, rescaled so column norm of `r_g`
  matches the column norm of `u_g_foot` for that guide. Refit and
  record pooled ρ per draw.
- (d) *Matched-SNR linear-truth on footprint U.* Recompute `α_S` from
  the median column norm of the footprint `U` on the same real `S`;
  simulate `J_ref = G − 1.5·I`, set `J = J_ref / α_S`, compute
  `S_true = −J⁻¹U_foot`, add noise at each screen's own σ, run the
  same nested-CV recipe. 15 replicates.

**Success criterion, per screen.**
Footprint nested-CV ρ **≤ 0.90 AND below the 2.5th percentile of the
shuffled-footprint null**. Both conditions must hold. This threshold
is fixed here before running.

**Decision rule.**
- Success on ≥ 2 screens → the projection mechanism becomes a Result
  in the manuscript (moved from Open Questions to §2 / §3).
- Failure on all three, or success on 0–1 → report that the footprint
  encoding does not rescue held-out prediction and that projection
  failure stays open in Discussion §3.2.
- Mixed outcomes → report per screen with the exact ρ, the null
  percentile it sits at, and the success/failure verdict for that
  screen. No cross-screen ranking.

**Direction-only variant.** A direction-only refit of the footprint
encoding is also reported alongside — same recipe as `§4.7`, column-
normalize `S` and `U_foot` before nested-CV — but it is not part of
the success criterion above.

**Deliverables.** `reproduction/45_footprint_encoding.py`,
`results/recheck/F_footprint.json`, and an update to `RECHECK_LOG.md`
containing exactly the results as they come out. If a bug is
discovered post-hoc, the fix, the rationale, and all reruns of the
controls are appended to `RECHECK_LOG.md` transparently and this
amendment section is not edited.

---

## Amendment — comparator panel, positive-control ensembles, and random-panel distribution (2026-09-28)

This section is being committed **before any of the Steps 1–3 code
below is run**. The specifications lock in the recipes, success
criteria, and decision rules independently of the outcomes.

### Step 1 — Comparator panel

All comparators use the same target-grouped outer folds (K = 5) and
target-grouped inner folds (K_inner = 3) as Table 1. Hyperparameters
are chosen only on inner folds. Nothing from held-out targets enters
any fit, gene embedding, or PCA.

**1a. Inverse direction, current metric.**
`ρ = ‖A·S_test + U_test‖_F / ‖U_test‖_F`. Predict-zero baseline
= 1. Two estimators:
- TSVD with the existing rank grid `{0, 1, 2, 3, 5, 8, 12, 16, 20,
  25, 30}` (already reported in Table 1).
- Ridge / Tikhonov, `λ` chosen from a log grid on inner folds, with
  `λ → ∞` (fit reduces to `A = 0`, ρ = 1) included so nested CV can
  select the predict-zero baseline.

Reporting: both next to the matched-linear-truth ρ per screen.

**1b. Forward direction.**
`ρ_fwd = ‖Ŝ_test − S_test‖_F / ‖S_test‖_F`. Baselines:
- predict-zero (`Ŝ_test = 0` → ρ_fwd = 1);
- **predict-training-mean response** — the column mean of training
  `S`, broadcast across `n_test` columns. This is REQUIRED. About
  half of the response energy is a shared mode, so beating zero is
  trivial; success is defined against the training-mean baseline.

Encodings tested, each fit with ridge (`λ` by inner CV):
- **Fixed.** `Ŝ_test = B·U_test`, with `U_g = Wᵀδ_g` and `B` fit
  on training.
- **Footprint.** Same, with `U_g = Wᵀ Σ_ctrl δ_g` (Σ_ctrl from NT
  controls only, same HVG feature space, targets force-kept).
- **Learned.** Following the linear-baseline model of Ahlmann-Eltze
  et al. 2025 (structure per their published code at
  github.com/const-ae/linear_perturbation_prediction-Paper). The
  gene embedding is derived from a PCA on the training targets'
  gene-space pseudobulk responses only, refit within every outer
  fold. The held-out target's embedding is that gene's row in the
  training-derived embedding — this is legitimate because it is a
  feature-gene, not a response. Prediction is in gene space; project
  through `W` to program space for the metric. Also report the
  gene-space ρ_fwd.

Nulls for the learned encoding: **shuffled-embedding null** —
permute target embeddings across targets, ≥100 permutations, refit
per permutation.

Matched linear truth: ρ_fwd for each encoding's U where defined.

**Preregistered success criterion, per screen, per encoding:**
ρ_fwd is more than 2 outer-fold SDs below the training-mean baseline
AND below the 2.5th percentile of its shuffled-embedding null (learned
encoding only; the fixed and footprint encodings need no shuffled
null because the encoding is not learned from training responses).

**Decision rule:**
- Learned succeeds on ≥ 2 of 3 screens **and** fixed does not:
  promote to a Result. Update title and abstract to state that the
  fixed program-space encoding fails but a learned encoding recovers
  held-out prediction.
- Learned also fails: report that neither fixed nor learned linear
  encodings predict held-out targets beyond the training mean on
  these screens.
- Mixed outcome: per-screen reporting; no cross-screen ranking.

### Step 2 — Positive-control ensembles at matched amplitude

Rerun the matched-SNR linear-truth control (nested-CV ρ, and
interaction-only Frobenius cosine vs cross-replicate null) on K562,
RPE1, and Jost across five ground-truth ensembles:
- dense;
- sparse-10% (Bernoulli off-diagonal mask);
- sparse-2%;
- rank-5 (`J_int = B·V` with `B ∈ ℝ^(d×5)`, `V ∈ ℝ^(5×d)` both
  scaled by 1/√d);
- block-modular (5 blocks of size ~d/5, dense within blocks with a
  weak between-block coupling; specifically, per-entry Gaussian at
  1/√d within blocks and at 1/(5·√d) between blocks).

α_S is re-derived per ensemble per screen so median column-norm of
`S_true` matches the observed median. Draws with spectral abscissa
of `J_true` above zero are rejected and re-drawn; report the
rejection fraction per ensemble. N ≥ 15 accepted draws per cell.

**Criterion:** the conclusion "the matched linear truth beats
predict-zero at matched SNR" should hold across all five ensembles.
Report any (screen × ensemble) cell where it does not.

### Step 3 — Random-panel distribution (K562)

Fit anchor-op once on all approximately 1,740 qualifying K562
essential targets (≥ 60 cells, same filters), saving `Σ_ctrl` from
that fit. Then subsample 50 random 200-target panels from that
single measurement (fixed `Σ_ctrl`, fixed basis W, fixed κ). For
each panel, report:
- real nested-CV ρ under Table 1's recipe;
- matched-linear-truth ρ (N = 5 sims per panel).

Report the distribution across the 50 panels: median, 5th–95th
percentile band. Preregistered criterion: report the **fraction of
panels where real ρ < 0.95**. No hidden threshold; the fraction is
the reportable number.

### Step 4 — Non-essential check (optional)

If a Replogle genome-wide K562 h5ad is available locally and time
allows, repeat Table 1 on 200 random non-essential targets. If not,
add to Limitations exactly: "Both Replogle screens are essential-gene
libraries dominated by a shared growth/stress response; whether the
result holds for non-essential perturbations is untested."

### Deliverables and reporting rule

All Steps 1–3 outputs go to `results/recheck/F_step[1-3]_*.json` and
new/updated figures. `RECHECK_LOG.md` receives the results exactly as
they come out. If a bug is found post-hoc, the fix, the rationale, and
all reruns of the affected controls are appended transparently; this
amendment section is not edited.

---

## Preregistration design flaw noted (not a change, but a caveat)

- **`rel_diff ≤ 0.25` threshold is not calibrated to any realistic
  SNR.** Under a matched-SNR linear-truth simulation on each dataset
  (`reproduction/30_recheck_F1_F4.py` for Replogle;
  `reproduction/39_recheck_jost_grouped_folds.py` for Jost), the
  linear truth itself gives `rel_diff` of 1.05–1.57 — well above 0.25.
  Every observed screen (real rel_diff 1.26–1.57) fails the threshold,
  but so does the linear truth. The threshold is unreachable at these
  SNRs regardless of ground truth: it is a preregistration design
  flaw, not a data verdict. The manuscript's linearity narrative uses
  the nested-CV ρ comparison
  (`reproduction/40_nested_cv_rho.py`) and the α_S signal-scale sweep
  as the calibrated substitutes; the preregistered rel_diff line is
  retained for continuity with `PREREGISTRATION.md` but reported
  alongside its matched-linear-truth value so the reader can judge.
