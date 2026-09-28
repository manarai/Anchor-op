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
