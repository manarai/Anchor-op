# Recheck log — audit response, 2026-09-27

**Seed:** `SEED_BASE = 20260927` (per-script) / `20260810` (per script 21).
**Test suite:** `pytest -q` → **58 passed** after the F5 gate fix. Two new
tests added in `tests/test_full_domain_gate.py`.
**All output files:** `results/recheck/*.json`, `results/recheck/number_provenance.csv`.

## Summary of verdicts

| ID | Verdict | Notes |
|----|---------|-------|
| F1 | **CONFIRMED** | Positive-control signal is ~358× (K562) / ~194× (RPE1) weaker than the real responses at the published `α=1` amplitude. |
| F2 | **CONFIRMED** | At data-matched signal amplitude, the operator is recovered: dense-ensemble cos_full ≈ 0.99 (K562) / 0.91 (RPE1). Cross-replicate paired null ≈ 0. |
| F3 | **CONFIRMED** | Linearity diagnostic on real data (ρ = 1.11 K562 / 1.21 RPE1) reflects a real linearity-vs-real-data gap, not overfitting or noise-limited behavior. A linear-truth simulation at matched SNR gives ρ ≈ 0.18 (K562) / 0.53 (RPE1). |
| F4 | **CONFIRMED** | Top singular direction carries 51% of `S` energy for both cell lines. Between-target mean cosine +0.38 (K562) / +0.14 (RPE1). Median `|cos(s_g, u_g)|` is 0.115 (K562) / 0.122 (RPE1) against chance ≈ 0.118 — literally at chance for K562. ρ *rises* after mean-response projection removal. |
| F5 | **CONFIRMED with a wording caveat** | Jost `U` at d=30 has 25 singular values ≥ 1e-6 relative and 5 essentially zero (~1e-9). Under the preregistered `rank_tol = 1e-2`, `input_rank = 24` (the 25th SV = 0.0049 is below the threshold). The gate bug is real: `full_domain = (effective_response_rank == d)` ignored `input_rank`. Fixed to require both. New tests pass. |
| F6 | **CONFIRMED** | Median `\|\|u\|\|` = 0.068 (Jost) vs 0.029 (K562). κ median = 0.82 (Jost, count-based) vs 0.32 (K562, detection-shift proxy). Same-σ comparison in §2.7 is not a same-SNR comparison. |
| F7 | **CONFIRMED (code path)**, rerun deferred | `qualifying[:N_TARGETS_KEEP]` in both notebooks slices the descending `value_counts()`, keeping the 200 targets with the most cells. In essential screens, high cell count correlates with mild fitness defect. Full h5ad refit (~10 GB per file, 6 fits total) was not run in this pass; the runnable script is at `reproduction/35_recheck_F7_selection_sensitivity.py`. |

**F1–F3 all confirmed → the manuscript is restructured.** See
`MANUSCRIPT_RESTRUCTURE_PLAN.md`. Consistency edits to
`MANUSCRIPT.md` / `README.md` are only the items that do not depend on
the restructure decision.

## F1 — Signal amplitude
Command: `python reproduction/30_recheck_F1_F4.py`
Output: `results/recheck/F1_K562_essential.json`,
`results/recheck/F1_RPE1_essential.json`, `results/recheck/F1_F4_summary.json`

| Quantity | K562 | RPE1 | Reviewer's claim |
|---|---:|---:|---|
| median ‖S_col‖ observed | 9.297 | 3.909 | 9.3 / 3.9 |
| median ‖S_col‖ sim, α=1, dense | 0.02597 | 0.02014 | ~0.026 / ~0.020 |
| ratio (obs / sim) | 358.0× | 194.1× | 200–350× |
| median ‖U_col‖ | 0.0293 | 0.0217 | 0.029 |
| median κ | 0.319 | 0.324 | 0.32 |

**Refutation attempts.**
- *σ basis mismatch.* Same basis object as the stored `S` (the pkl carries
  both). Median ‖S_col‖ / σ = 9.30 / 0.24 ≈ 39 (K562), 3.91 / 0.352 ≈ 11
  (RPE1). σ is in the same PCA coordinates as `S`; no rescaling.
- *Methods §4.4 says 4.26.* That earlier number is not reproduced by the
  current pkl and is either stale (from the σ = 0.266 anchor) or from a
  different aggregation (e.g. target rather than per-column). Flagged for
  the consistency audit.
- *U scale.* `median ‖u_col‖ = 0.0293` for K562 matches κ_median × sqrt(30/3142)
  ≈ 0.32 × 0.098 ≈ 0.031 for a single-target `W^T δ_g`. Consistent with
  the derivation `u_g = −κ_g Wᵀδ_g` and the `z*` factor being 1 in this
  encoding. This confirms the Introduction's omission-of-`z*` note.

## F2 — Signal-scale sweep
Command: `python reproduction/30_recheck_F1_F4.py` (F2 section)
Output: `results/recheck/F2_K562_essential.json`, `results/recheck/F2_RPE1_essential.json`.

Parameterization: `J_true(α_S) = J_ref / α_S`, so `‖S_true‖ ∝ α_S`. The
published simulation is α_S = 1. `α̂_data` puts median sim ‖S‖ at the
observed median: **K562 α̂ = 368.7, RPE1 α̂ = 199.3**.

Full sweep (dense ensemble):

| α_S | K562 cos_full | K562 mag_ratio | RPE1 cos_full | RPE1 mag_ratio |
|---:|---:|---:|---:|---:|
| 1 | +0.035 | 0.007 | +0.019 | 0.005 |
| 10 | +0.300 | 0.082 | +0.181 | 0.050 |
| 30 | +0.584 | 0.256 | +0.413 | 0.159 |
| 100 | +0.889 | 0.621 | +0.763 | 0.452 |
| 200 | +0.970 | 0.824 | +0.913 | 0.682 |
| 300 | +0.988 | 0.904 | +0.959 | 0.801 |
| 500 | +0.996 | 0.960 | +0.987 | 0.905 |
| 1000 | +0.999 | 0.989 | +0.997 | 0.972 |

Cross-replicate paired null at high α is +0.70 (dense J shares the −1.5I
diagonal across draws, so full-matrix cosine to another replicate's J is
≈ ‖−cI‖² / ‖J‖² ≈ 0.7). Paired-diff over the null tops out at ~0.30. This
mirrors reviewer's "interaction-only cosine ≈ 0.97/0.59, cross-replicate
null ≈ 0" — the manuscript's dense positive control at α = 1 is a low-SNR
regime, and the negative result is a signal-amplitude conclusion, not an
operator-recovery conclusion.

**Refutation attempts.**
- *TSVD vs bare pinv.* The sweep uses `regularized_pseudoinverse` (the
  package TSVD path with `rank_tol=1e-2`), not `numpy.linalg.pinv`. Bare
  pinv would recover perfectly at any α at σ = 0 — irrelevant here.
- *N = 200 (dense; smaller for full sweep, N = 50 to keep runtime bounded).*
  Cross-replicate paired null was included; the "matched" cell is well
  above the null at any α ≥ 100.

Reviewer's rescale-J-to-match-observed conclusion is upheld.

## F3 — Linearity diagnostic

Command: `python reproduction/30_recheck_F1_F4.py` (F3 section)
Output: `results/recheck/F3_K562_essential.json`, `results/recheck/F3_RPE1_essential.json`.

| Cell line | ρ (real) | ρ, linear-sim @ α=1 | ρ, linear-sim @ matched α̂ | rel_diff (real) | rel_diff, matched |
|---|---:|---:|---:|---:|---:|
| K562 | 1.114 | 1.120 ± 0.007 | **0.181 ± 0.004** | 1.466 | 0.334 |
| RPE1 | 1.208 | 1.150 ± 0.011 | **0.531 ± 0.012** | 1.571 | 0.984 |

Interpretation. A linear-truth simulation at the *same* SNR as the real
data gives ρ ≪ 1 — the additive-input encoding held-out prediction *would
pass* if a linear truth ran the data. Real data is at ρ ≈ 1.1–1.2. That
gap is not noise, not overfitting (900 parameters vs ~150 training guides,
but ρ_lin at matched SNR is 0.18, not >1), and not sample size.

**Refutation attempts.**
- *Overfitting.* 30×30 A on ~150 training guides would give large ρ if
  overfitting. But at matched SNR the linear sim gives ρ = 0.18, so
  overfitting is not the source of ρ > 1 on real data.
- *Noise-limited.* ρ at published SNR (α = 1) is 1.12 for the linear
  simulation — matches real data. This is the "noise-dominates,
  everything looks the same" regime. Once α is set to the real signal
  amplitude, the linear-truth ρ falls to 0.18/0.53 while real ρ stays
  at 1.1/1.2. That's the rejection.

## F4 — Shared mode, target-input orthogonality

Command: `python reproduction/30_recheck_F1_F4.py` (F4 section)
Output: `results/recheck/F4_K562_essential.json`, `results/recheck/F4_RPE1_essential.json`.

| Quantity | K562 | RPE1 | Reviewer's claim |
|---|---:|---:|---|
| top SV energy fraction of S | 0.510 | 0.512 | 0.51 (K562) |
| between-target mean cosine | +0.383 | +0.143 | +0.38 / +0.14 |
| median \|cos(s_g, u_g)\| | 0.115 | 0.122 | 0.115 / 0.122 |
| chance \|cos\| (2/π·√(d−1)) | 0.118 | 0.118 | ~0.146 (reviewer used a different chance) |
| ρ (real) | 1.114 | 1.208 | 1.12 / 1.26 |
| ρ after mean-response projection removed | **1.156** | **1.229** | ~1.12 / 1.26 (approximately unchanged) |

Reviewer said "held-out ρ *stays* at 1.12 / 1.26 after projecting out the
mean-response direction." Recomputation: ρ *rises slightly* (1.11 → 1.16;
1.21 → 1.23) — the shared mode was contributing a small amount of
predictable variance, and its removal slightly worsens held-out
prediction. Directionally the same conclusion (removing the shared mode
does not rescue linearity), but the exact numbers are 1.156 / 1.229, not
1.12 / 1.26. Recorded as F4 confirmed with corrected numbers.

**Three readings of F4, and the correct test for each.** F4 is
correlational and consistent with at least three mechanisms. A previous
version of this log claimed within-target concordance separated (a)
from (c); that was wrong. Under any additive-linear model, all guides
for one target move the system along the same direction (κ rescales,
`Wᵀδ_g` is fixed) — only amplitude differs. So concordance separates
(b) from {a, c}, not (a) from (c).

  (a) *Projection failure.* `Wᵀδ_g` puts ~99% of the target-direction
      unit vector outside the 30-PC subspace; the encoded input has
      little response-relevant signal.
      **Test:** footprint encoding `u_g ∝ Wᵀ Σ_ctrl δ_g` (or a
      gene-space model on a small target set).
  (b) *Dose-dependent non-linearity.* The additive settled-state
      assumption `S = −J⁻¹U` is the wrong shape for the assay's actual
      dynamics.
      **Test:** within-target direction concordance vs a between-target
      null and a within-guide split-half noise ceiling. Low
      within-target direction concordance (below noise ceiling and near
      between-target null) is evidence for (b).
  (c) *κ-proxy mis-scaling.* The `detection_rate` κ used on
      Replogle-essential pre-scaled data may not track actual input
      strength; response amplitude follows fitness effects rather than
      knockdown efficiency. κ only rescales columns of U.
      **Test:** direction-only held-out ρ — refit after
      column-normalizing S and U, or fit with a free per-guide scale.
      If ρ still fails, (c) is not the explanation.

Order of tests: (c) settles fastest — it runs in minutes on the
Replogle pkls already tracked. (b) needs the concordance-hardening
extension + Jost. (a) needs the footprint-encoding comparison. See
`MANUSCRIPT_RESTRUCTURE_PLAN.md` for the run order and blocker list.

**Reading (c) result** (`python reproduction/38_direction_only_rho.py`,
output `results/recheck/F_direction_only_rho.json`).
Column-normalizing S and U before the fit removes all per-column scale
that κ could carry. If (c) were live, held-out ρ should drop
substantially under this transform. It does not:

| Dataset | baseline ρ | direction-only ρ | Δρ | control linear-sim ρ (direction-only) |
|---|---:|---:|---:|---:|
| K562 essential | 1.114 | 1.087 | −0.027 | 0.169 |
| RPE1 essential | 1.208 | 1.128 | −0.080 | 0.172 |
| Jost 2020 (below, F3 rerun) | 0.666 | 0.755 | +0.089 | — |

The estimator still recovers a linear ground truth on the same column-
normalized design (control ρ ≈ 0.17 on both Replogle cell lines), so the
direction-only path is a working estimator. Real Replogle holds ρ ≈ 1.1
under it. **Reading (c) — κ-proxy mis-scaling — is ruled out for both
K562 and RPE1 essential.** On Jost, direction-only ρ actually *rises*
by 0.09 (real 0.666 → direction-only 0.755) — the per-column scale
information was slightly *helping* the fit, not hiding a κ error, so
reading (c) is ruled out on Jost too under Jost's count-based κ.

The remaining live readings are (a) projection failure and (b) dose-
dependent non-linearity; the tests that distinguish them are still
outstanding.

## F3 on Jost 2020, end to end (produced by
`reproduction/37_recheck_F3_jost_end_to_end.py --basis pca_controls`;
output `results/recheck/F3_Jost_pca_controls.json` and
`results/jost_measurement.pkl`)

Fit configuration (matches Replogle's for a like-for-like comparison):
- PCA on 2,446 log-normalized non-targeting-control cells; d = 30.
- HVG selection at n_top = 3,000, with all 25 Jost target genes
  force-kept in the feature matrix (otherwise `measure_operator` drops
  93/100 sgRNAs for `target_gene_absent_from_response_matrix`).
- `measure_operator` with `aggregate_replicate_guides=False` — the
  default `True` collapsed per-sgRNA responses to one-per-target on the
  first attempts and defeated the whole point of running Jost.

Numbers:

| Quantity | Jost | K562 (comparison) |
|---|---:|---:|
| n_sgRNAs retained | 122 (of 124) | 188 |
| n_targets | 25 | 188 (one guide per target after Replogle aggregation) |
| rank(U) at rank_tol = 1e-2 | 25 (`input_subspace_dim = 25`) | 30 |
| `full_domain_identified` | False (correct under fixed gate) | True |
| σ_per_sgRNA (bootstrap in Jost's PCA basis) | 0.066 | 0.240 |
| median ‖S_col‖ observed | 1.44 | 9.30 |
| median ‖S_col‖ sim @ α=1 | 0.049 | 0.026 |
| α_hat_S (data-matched) | 29.6 | 369 |
| Real ρ_pooled | **0.666** | 1.114 |
| Real ρ_direction-only | 0.755 (Δ = +0.089) | 1.087 (Δ = −0.027) |
| Real `rel_diff` | 1.260 | 1.466 |
| Matched-SNR linear-truth ρ_overall | **0.220** | 0.181 |
| Matched-SNR linear-truth ρ_identified_subspace | 0.213 | — |
| Real – matched-linear-truth ρ gap | **0.45** | 0.93 |

**Sibling-sgRNA leakage — ρ = 0.67 was flattered.**
The 0.67 above was computed by guide-level 5-fold CV. Jost has ~5 sgRNAs
per target sharing `Wᵀδ_g`, so every held-out sgRNA had siblings in
training and the held-out prediction was interpolating dose along a
known direction. Rerun with **target-grouped folds** (all sgRNAs of a
target held out together;
`reproduction/39_recheck_jost_grouped_folds.py`, output
`results/recheck/F3_Jost_grouped_folds.json`):

| Metric | Real Jost | Linear-truth @ matched α = 29.6 | Real − linear |
|---|---:|---:|---:|
| guide folds (leaked) | 0.655 | 0.218 | +0.44 |
| **target-grouped folds** | **2.43** | 0.697 | +1.73 |
| direction-only guide folds | 0.774 | 0.492 | +0.28 |
| direction-only target-grouped folds | 1.37 | 0.829 | +0.54 |

On target-held-out prediction, real Jost ρ is **2.43** — worse than
predicting zero (ρ = 1). The 0.67 was pure within-target dose
interpolation, not new-perturbation prediction. Replogle's numbers
already correspond to target-held-out (one guide per target after
aggregation).

**rel_diff calibration on Jost.** Real Jost rel_diff = 1.26 vs matched-
SNR linear-truth rel_diff = 1.05 ± 0.03 (N = 15). Real fails the
preregistered 0.25 threshold, but so does the linear truth: rel_diff
is not calibrated to Jost's SNR — the F3 trap applies.

**rank(U) reconciliation.**
- F5 recheck reported rank(U) = 24 at rank_tol = 1e-2 on the **raw
  124-sgRNA U** before the efficiency filter.
- F3-Jost and the grouped-fold recheck both report rank = 25 on the
  **retained 122-sgRNA U** after the min-cells / min-κ filter.
Both are correct; they refer to different matrices.

**Reading-(c) verdict on Jost, revised.** Jost uses count-based κ
(`mean_ratio`), the more trusted estimator. Under guide folds,
direction-only ρ goes 0.66 → 0.77 (worsens by 0.11). Under target
folds it goes 2.43 → 1.37 (improves markedly, but from a worse-than-
zero baseline). Column-normalizing the linear-truth control at matched
α also costs it substantially (target-grouped ρ 0.70 → 0.83), which
Replogle's controls did not show (both stayed near 0.17). Wider κ
range *is* informative on Jost, and normalization discards real signal
there. **Reading (c) is ruled out on Jost — κ is not a mis-scaling
artifact — but the direction-only test is not the same "cheap
diagnostic" it was on Replogle**; on Jost it discards information
rather than removing an error.

Restated for Replogle: on K562/RPE1 essential (detection-rate κ on
pre-scaled data), the direction-only test barely moves ρ (Δ ≈ −0.03 /
−0.08), so the κ proxy is *uninformative* on those datasets — and
therefore cannot account for their failure, which persists without
it. Reading (c) is ruled out as a *sufficient* explanation for Replogle
but the proxy quality is genuinely a concern separate from the
mechanism story.

**Central-claim consequence.** Two follow-on reruns (below) show that
the target-grouped ρ values above are still not directly comparable
across datasets:
- ρ well above 1 mostly measures the estimator's variance under
  ill-conditioned training S. Choosing regularization per fold by
  nested CV shrinks the "> 1" excess and lets the encoding be tested
  at its best-achievable point.
- Jost's ρ = 2.43 at d = 30 sits in an underdetermined regime (20
  training targets across 25 total < d = 30), so its magnitude is
  partly a design property, not an encoding statement.
Each dataset must be compared to its **own** matched-SNR linear-truth
control at its best achievable regularization, not to the other
datasets. See the two sections that follow for the fair comparison.

## F3 nested-CV regularized ρ, all three datasets — corrected version

Two things fixed in this revision:
- Inner CV was previously guide-random within the outer training set.
  On Jost, that leaked sibling sgRNAs between inner-train and inner-
  val (the same failure mode the outer fix removed). Inner CV is now
  target-grouped too.
- The rank grid now includes rank 0. Rank 0 means A ≡ 0, giving ρ = 1
  exactly. Nested CV can now legitimately pick "shrink to zero" when
  the fit is worse than predict-zero — turning "> 1 excess" into a
  diagnostic that gets absorbed by the rank choice instead of showing
  up in the pooled ρ.

Recipe: outer 5-fold target-grouped; inner 3-fold target-grouped;
grid ∈ {0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}. Script:
`reproduction/40_nested_cv_rho.py`.

| Dataset | Real ρ (nested-CV) | Median picked rank | Matched linear-truth ρ (N=15) |
|---|---:|---:|---:|
| K562 essential | **0.96** | 3 | 0.18 ± 0.003 |
| RPE1 essential | **1.00** | 1 | 0.53 ± 0.016 |
| Jost 2020 (d=30) | **1.00** | 0 | 0.72 ± 0.027 |

On Jost, nested CV picks rank 0 for the median fold — the fit is
worse than predict-zero, and A ≡ 0 wins. That collapses the earlier
Jost ρ = 2.43 / 2.23 to ρ = 1.00, confirming those numbers were
entirely estimator variance under an ill-conditioned training S.

**The corrected picture:** on all three datasets, real ρ cannot get
below 1 (the predict-zero baseline); on all three, a matched-SNR
linear truth reaches ρ well below 1 under the same nested-CV
estimator on the same U and σ.

## F3 Jost d-sweep under target-grouped folds — corrected version

Same fix (target-grouped inner + rank 0). Script:
`reproduction/41_jost_d_sweep_grouped.py`. Truncate to d ∈
{5, 10, 15, 20, 25, 30}; α̂ recomputed at each d.

| d | Real ρ | Linear-truth ρ (N=15) | α̂ | Regime |
|---:|---:|---:|---:|:---|
| 5  | 1.01 | 0.32 | 24.7 | overdetermined |
| 10 | 1.00 | 0.85 | 6.1  | overdetermined |
| 15 | 0.99 | 0.46 | 43.7 | overdetermined |
| 20 | 1.00 | 0.55 | 36.6 | overdetermined |
| 25 | 1.00 | 0.69 | 28.7 | underdetermined |
| 30 | 1.00 | 0.73 | 29.6 | underdetermined |

Real Jost ρ pinned at 1.00 across the whole sweep — the encoding
never beats predict-zero, at any d. The matched linear-truth control
reaches 0.32 at d = 5 (best SNR regime) and stays 0.46–0.85 elsewhere.
Design underdetermination is not the driver: even at d = 5 where
identifiability is comfortably overdetermined, real Jost is at the
predict-zero floor.

## F2 interaction-only cosine at matched α (produced by
`reproduction/43_interaction_only_cosine.py`, output
`results/recheck/F_interaction_only_cosine.json`).

Reviewer note (2026-09-27): the full-matrix cos ≈ 0.99 / 0.91 at
matched α is dominated by the shared −cI diagonal, so its shift-1
cross-replicate null is ≈ 0.70. Report the **interaction-only**
cosine — subtract the diagonal from both A and J — against a
near-zero null instead.

| Dataset | Full cos | Shift-1 null (full) | Interaction-only cos | Interaction-only null | Paired-diff |
|---|---:|---:|---:|---:|---:|
| K562 essential | 0.993 | 0.696 | **0.976** | 0.003 | 0.973 |
| RPE1 essential | 0.912 | 0.675 | **0.738** | 0.003 | 0.735 |

(N = 200; dense J ensemble; α_S 369 / 199; σ 0.240 / 0.352). K562 lands
where the reviewer predicted (0.97); RPE1 came out at 0.74 rather than
their memory-based 0.59, but the qualitative point holds — the
estimator recovers the operator's interactions cleanly at matched SNR,
so noise dominance alone does not preclude recovery on either
Replogle line.

## K562 nested-CV real ρ, fold-to-fold spread

From `results/recheck/F_nested_cv_rho.json` (5 outer target-grouped
folds; picked ranks 3, 3, 3, 5, 3): per-fold ρ mean 0.964, SD 0.019,
min 0.944, max 0.991. Report as "K562 real ρ = 0.96 (5-fold SD 0.02)"
in the abstract. RPE1 5-fold SD is 0.006; Jost is 0.002 — both close
enough to the predict-zero floor that a CI would be uninformative.

## Footprint-encoding test (produced by
`reproduction/45_footprint_encoding.py`, output
`results/recheck/F_footprint.json`; preregistered 2026-09-27 in
`PREREGISTRATION_AMENDMENT.md` before any code ran).

Encoding: `u_g = −κ_g · Wᵀ · Σ_ctrl · δ_g`, `Σ_ctrl` from NT controls
only, same HVG feature space, targets force-kept, same `W`, same `κ`.
`Σ_ctrl` reloaded from each screen's original h5ad (K562 random-200
loaded the cached `results/k562_random200_sigma_ctrl.npz`; K562
top-200, RPE1, and Jost each computed and saved a fresh cache).

Recipe: same target-grouped nested-CV as Table 1 (5-fold outer,
3-fold inner, grid `{0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}`).
Controls per screen: (a) current-encoding baseline (recomputed for
comparability), (b) shuffled-footprint null at N = 100 permutations
of the target→footprint map, (c) random-direction null at N = 100
draws of unit vectors rescaled to the footprint column norms,
(d) matched-SNR linear-truth on footprint U at N = 15.

Success criterion (preregistered): footprint ρ ≤ 0.90 AND below the
2.5th percentile of the shuffled-footprint null. Both required.

**Verdicts.**

| Screen | Footprint ρ | Shuffled p025 | Random p025 | Linear-truth ρ (SD) | Verdict |
|---|---:|---:|---:|---:|:---|
| K562 top-200 | 0.81 | 0.97 | — | — | **success** |
| K562 random-200 | 1.00 | 0.99 | — | — | failure |
| RPE1 | 1.00 | 0.98 | — | — | failure |
| Jost 2020 | 1.00 | 0.98 | — | — | failure |

(Random-direction null and matched-linear-truth stats live in
`F_footprint.json`; the shuffled-footprint 2.5th percentile is the
decisive comparator per preregistration.)

**Decision rule outcome.** Success on 1 of 4 fits. This is below the
preregistered ≥ 2-screen threshold that would promote the projection
mechanism to a Result. Under the preregistered "mixed outcomes"
branch, we report per screen and keep projection failure as an open
question in Discussion §3.2. The K562 top-200 success is treated as
a positive signal for the projection reading on that specific fit,
not as a general mechanism claim; the K562 random-200 failure on the
same cell line under a random-200 selection is the more informative
data point, since it isolates the effect of the selection convention.
No parameters or thresholds were changed after the results came in.

## F7 K562 random-200 refit (produced by
`reproduction/42_f7_k562_random200.py`, output
`results/recheck/F7_K562_random200.json`,
`results/k562_random200_measurement.pkl`,
`results/k562_random200_sigma_ctrl.npz`).

Selection sensitivity check against the notebook's
`qualifying[:N_TARGETS_KEEP]` (top-200 by cell count). Random sample of
200 targets from the 1,740 qualifying (≥ 60 cells) pool at
`SEED = 20260927`. Full refit: variance-based HVG on the pre-scaled
matrix (force-keeping the sampled targets — 180 of 200 are in the HVG
feature-matrix), PCA on non-targeting controls (d = 30), the proxy-
efficiency path used in `examples/01b`, TSVD at `rank_tol = 1e-2`.
`Σ_ctrl` saved for the optional footprint-encoding follow-on.

| Quantity | K562 random-200 | K562 published top-200 |
|---|---:|---:|
| n_targets kept | 200 | 200 |
| n_guides retained | 180 | 188 |
| input_subspace_dim | 30 | 30 |
| `full_domain_identified` | True | True |
| α̂_S (data-matched) | 165.5 | 369 |
| Real nested-CV ρ | **1.005** | 0.96 |
| Median picked rank | 0 | 3 |
| Matched-SNR linear-truth ρ | 0.433 ± 0.011 | 0.181 ± 0.003 |
| Real − linear gap | 0.57 | 0.78 |

**Same qualitative outcome** as top-200-by-count: real ρ is pinned at
the predict-zero floor (nested CV even picks A ≡ 0 as best model),
while a matched linear truth reaches ρ well below 1 under the same
estimator. The failure is not created by the top-cell-count selection.

The gap is smaller on random-200 (0.57 vs 0.78) because α̂ is lower
(165 vs 369): random-200 targets have smaller-amplitude responses on
average, so the matched linear-truth control also operates at lower
SNR and its held-out ρ is higher. Signal amplitude is a property of
the selection, not of the encoding.

Cross-dataset ranking is retracted: each dataset compared only against
its own control shows the same qualitative picture — real held-out ρ
sits around 1 while a matched linear truth sits well below.

**Reading-(c) verdict, restated for the record.**
- On Replogle (detection-rate κ on pre-scaled data): direction-only
  refits barely move ρ (Δ ≈ −0.03 / −0.08). The κ proxy is
  **uninformative** — it cannot account for the failure (which
  persists without it) but is also not contributing useful signal.
- On Jost (count-based κ on UMIs): direction-only refits worsen ρ,
  and column-normalizing the linear-truth control also costs it
  substantially (target-grouped matched-α ρ 0.70 → 0.83). Count-based
  κ carries **real** information. Reading (c) is ruled out as a
  sufficient explanation on both, but κ *quality* is a real
  cross-dataset difference worth keeping in the manuscript.

**rel_diff calibration** — Jost real 1.26 vs matched-SNR linear-truth
1.05 ± 0.03. The preregistered `rel_diff ≤ 0.25` threshold is
unreachable at every one of the tested SNRs regardless of ground
truth. Documented in `PREREGISTRATION_AMENDMENT.md` as a design flaw
in the preregistration itself.

**Refutation attempts.**
- *Chance level.* Reviewer wrote "chance ~0.146"; my computation with the
  standard formula for `E|cos|` between unit vectors in R^d is
  `2 / (π √(d−1))` = 0.118 at d = 30. Empirical chance under uniform
  spherical draws confirms 0.118, not 0.146. Either way the observed
  median (0.115 K562) is *at or below* chance — the reviewer's qualitative
  conclusion is unchanged and actually stronger under my chance value.
- *"target lies outside 30-PC span" phrasing.* Every target in both pkls
  is present in the HVG feature matrix (0 out of 188 K562, 0 out of 153
  RPE1 dropped). But the target's projected input norm `‖Wᵀδ_g‖ ≈ 0.090
  (K562), 0.075 (RPE1)` means 99% of the target-direction unit vector
  lies outside the 30-PC subspace. So the correct wording is
  "the target direction is nearly orthogonal to the retained PCs" —
  not "excluded from HVG." Documented under the Manuscript consistency
  audit.
- Footprint encoding `u_g_foot ∝ Wᵀ Σ_ctrl δ_g` is exploratory and
  requires `Σ_ctrl`, which is not exported to `results/`. Deferred; the
  Bucket-C decision on encoding change stays with the author.

## F5 — Jost U rank and full_domain_identified gate

Command: `python reproduction/33_recheck_F5_jost_rank.py`
Output: `results/recheck/F5_jost_gate.json`,
`results/recheck/F5_jost_note.json`.

Jost U at d=30:
- singular values: last 5 are 5.3e-9, 4.1e-9, 3.0e-9, 2.6e-9, 1.9e-9;
- rank at 1e-2 relative tolerance = 24 (the 25th SV = 0.00493 is below
  1e-2 · σ_max = 0.00808);
- rank at 1e-6 relative or absolute = 25;
- 25 targets, 124 sgRNAs (3–6 per target).

The reviewer wrote "rank 25 at d=30." Under the preregistered
`rank_tol = 1e-2` used everywhere in the manuscript, rank is 24 not 25.
The material claim — that rank(U) < d — is unchanged.

**Gate fix.** `src/anchorop/identifiability.py` line 238:

```
- full_domain = selected.effective_rank == d
+ full_domain = (selected.effective_rank == d) and (input_rank == d)
```

`tests/test_full_domain_gate.py` adds two regression tests, one for the
noise-inflated-S-hides-rank-deficient-U failure and one that guards the
positive path. Full suite: 58 passed (was 56).

Downstream. `manuscript_figures/jost_matched_geometry_nulls.json` — the
§2.7 result — is unaffected in its per-cell cosines (they use
`identified_action` already), but every cell that reported
`full_domain_identified = True` under the old gate must now be relabeled
False. This is caught by the code path; §2.7 prose needs a note that
Jost's identified subspace is at most 25-dimensional, not d = 30.

**Condition number 3.38 vs the manuscript's 55.6 — reviewer flagged.**
Investigation (`python reproduction/33_recheck_F5_jost_rank.py`, plus an
extended trace):

| Object | Condition |
|---|---:|
| Jost `U` (all 30 SVs) | 4.3 × 10⁸ |
| Jost `U` after `rank_tol = 1e-2` truncation | **75.4** |
| noise-free `S = −J⁻¹U` (dense J, c = 1.5, seed = 20260927) | 6.9 × 10⁸ |
| noise-free `S` after rank_tol = 1e-2 truncation | **65.2** |
| noisy `S` at σ = 0.036, no truncation | 3.38 |
| noisy `S` at σ = 0.036, rank_tol = 1e-2 | 3.38 |
| `report.condition_number` (what F5 recheck printed) | 3.38 |

The 3.38 that my recheck reported is the ratio of largest to smallest
retained singular value of the *noisy* S at Jost's own σ. There are two
things making the number look healthy, and both are the same lesson as F1:

- **Noise-dominated S looks well-conditioned like a random matrix.** At
  the published J scale, Jost's noise-free per-guide response has norm
  ≈ 0.06, while the per-guide noise budget is σ·√d ≈ 0.036·√30 ≈ 0.20 —
  three to four times the signal. A noisy matrix at that SNR behaves as
  random Gaussian for its singular-value distribution, and a Gaussian
  matrix's expected singular-value ratio is O(1). The 3.38 is the
  small-c end of that random regime, not a statement about the operator.

- **Rank(U) < d is hidden by the same noise flood.** The five near-zero
  directions of the true S are raised by noise to `σ/√d ≈ 0.007`,
  and the rank_tol = 1e-2 filter still discards them (they sit below
  1e-2 · σ_max), leaving 30 numerically well-behaved directions.
  The report's `condition_number` ignores `input_subspace_dim`, so
  3.38 is technically correct for retained S but misleading as an
  "identifiability condition."

Same lesson as F1's amplitude gap: **a diagnostic computed at the wrong
SNR looks healthy.** The paper's "condition 55.6" and the recheck's
3.38 are two symptoms of the same mis-scaling, not two different
observations.

The manuscript's 55.6 is likely from the noise-free S or from U itself
after rank_tol truncation (both give values in the 65–75 range, close to
55.6 modulo seed / cell-line noise ratio). Neither the 3.38 nor the 55.6
is the natural quantity to report; the honest one is the condition of the
**identified action** on the 24-dimensional subspace where both rank(U)
and rank(retained S) survive. Flagged in `number_provenance.csv` and in
the restructure plan for the §2.7 rewrite; **do not** commit either
number into a public commit without re-deriving it.

## F6 — Jost input amplitude and common-SNR comparison

Command: `python reproduction/34_recheck_F6_jost_amplitude.py`
Output: `results/recheck/F6_input_amplitudes.json`.

| Dataset | median ‖u‖ | median κ | Frob ‖U‖ |
|---|---:|---:|---:|
| K562_essential | 0.0293 | 0.319 | 0.545 |
| RPE1_essential | 0.0217 | 0.324 | 0.417 |
| Jost_2020 | 0.0682 | 0.819 | 1.392 |

Median ‖u‖ Jost/K562 = 2.33×; median κ Jost/K562 = 2.57×. Reviewer's
numbers match. The §2.7 "same σ" comparison in the manuscript
implicitly holds ‖u‖ fixed; it does not.

A measured Jost `S` is not in `results/` (only `jost_u_at_d30.pkl` is
tracked). Reporting Jost's observed ρ, observed SNR and any real
same-SNR comparison to Replogle is not possible without exporting the
full Jost measurement bundle. `reproduction/34…` will pick it up
automatically once a `jost_measurement.pkl` or
`jost_essential_measurement.pkl` lands in `results/`.

**Important consequence for the restructure.** With only Jost's `U`
tracked, the "encoding fails held-out prediction" central claim rests on
Replogle alone (K562, RPE1). The two direct extensions to that claim —
Jost's real ρ, and Jost's observed SNR against σ = 0.036 — are both
gated on exporting `S_jost`. Recommend running end-to-end Jost `measure_operator`
with the same rank_tol = 1e-2 and dumping the pkl before the abstract is
rewritten around a Perturb-seq-wide claim.

## F7 — Target-selection bias

Code audit. `examples/01b_measure_k562_replogle.ipynb` cell 7 and
`examples/01c_measure_rpe1_replogle.ipynb` cell 7 both call
`qualifying = target_counts[target_counts >= MIN_CELLS_PER_TARGET].index.tolist()`
then `qualifying = qualifying[:N_TARGETS_KEEP]`.
`value_counts()` returns descending, so this keeps the 200 targets with the
most cells. In essential-gene screens, high cell count correlates with
mild fitness defect.

Full rerun on random-200 and fitness-stratified-200 was not run in this
pass. The h5ads are 10 GB each and a full refit (HVG selection + PCA +
`measure_operator` with bootstrap=100 in the notebook) takes ~15–25 min
per (dataset, selection). Six fits total. Ran the code path only.
`reproduction/35_recheck_F7_selection_sensitivity.py` is the runnable
script; add to `reproduction/run_all.sh` after refit.

## Descriptive-concordance hardening (backed_exploratory)

Command: `python reproduction/36_concordance_hardening.py`
Output: `results/recheck/F_concordance_null.json`.

- The Replogle-essential pipeline as used in `01b/01c` aggregates to
  target-level (`AGGREGATE_TO_TARGET=True`, one "guide" per target). The
  stored pkls therefore have **zero** targets with ≥ 2 guides. Within-target
  guide-replicate concordance is not measurable on the current pkls.
- Between-target null is computed from a `guide_responses.csv` when
  supplied; the script does not silently invent one.
- Within-guide split-half ceiling requires a ~30-line extension to
  `backed_exploratory_response_analysis.py`. Not applied in this pass;
  the extension is described in the script's docstring.
- Primary dataset for this diagnostic is Jost 2020 (25 targets × 3–6
  sgRNAs), not Replogle.

## Analysis changes: preregistered vs post hoc

See `PREREGISTRATION_AMENDMENT.md` for the dated list. In brief:
- **preregistered:** `rank_tol = 1e-2`, TSVD `reg="tsvd"`, `d = 30`.
- **post hoc analyses:** paired bootstrap on cross-replicate null; the
  per-dataset σ anchors (K562 0.240, RPE1 0.352); the α_S signal-scale
  sweep (`reproduction/30_recheck_F1_F4.py`).
- **post hoc code changes:** the full_domain_identified gate fix.

## Number provenance

`results/recheck/number_provenance.csv` maps each numeric claim in
`MANUSCRIPT.md` and `README.md` to its source file. Any number without a
source is flagged with `SOURCE_MISSING`.

---

# v0.3.0 addendum — Steps 1–3 (preregistered 2026-09-28, commit 08560c3)

**Seed:** `SEED = 20260928` (Steps 1–3 scripts).
**Preregistration:** `PREREGISTRATION_AMENDMENT.md` §"Amendment — comparator
panel, positive-control ensembles, and random-panel distribution
(2026-09-28)"; committed before any Step 1–3 code ran.

## Step 1 — Comparator panel

Command: `python reproduction/51_comparator_panel.py`
Output: `results/recheck/F_step1_comparator_panel.json`

### Inverse direction (ρ = ‖A·S_test + U_test‖/‖U_test‖; predict-zero = 1)

| Screen | TSVD (real) | Ridge (real) | TSVD (matched linear) | Ridge (matched linear) |
|---|---:|---:|---:|---:|
| K562 essential | 0.9673 | 0.9615 | 0.1825 | 0.1825 |
| RPE1 essential | 1.0011 | 0.9990 | 0.5364 | 0.5322 |
| Jost 2020 | 1.0004 | 0.9941 | 0.7285 | 0.7033 |

Ridge and TSVD track each other closely on the real data; both sit near
the predict-zero baseline on all three screens. The matched-linear-truth
controls sit well below 1 (K562 0.18, RPE1 0.53, Jost 0.72), matching the
Table 1 read that the inverse-direction identifiability failure is a data
property, not an estimator artefact.

### Forward direction (ρ_fwd = ‖Ŝ_test − S_test‖/‖S_test‖)

Baselines: predict-zero (ρ_fwd = 1) and predict-training-mean (column
mean of training S, broadcast). Success = ρ_fwd more than 2 outer-fold
SDs below the training-mean baseline AND (learned only) below the
shuffled-embedding null's 2.5th percentile.

| Screen | Train-mean baseline | Fixed | Footprint | Learned (prog) | Learned (gene) | Shuffled null p2.5 |
|---|---:|---:|---:|---:|---:|---:|
| K562 essential | 0.8382 | 0.8826 | 0.8538 | 1.0004 | 1.0002 | 0.9999 |
| RPE1 essential | 0.9341 | 0.9892 | 0.9859 | 1.0003 | 1.0001 | 1.0000 |
| Jost 2020 | 0.9681 | 1.0010 | 0.9843 | 1.0029 | 1.0047 | 1.0002 |

**Verdict (all 9 encoding × screen cells): success = False.** No encoding
sits more than 2 outer-fold SDs below the training-mean baseline on any
screen; the learned encoding does not fall below its own shuffled-null
2.5th percentile on any screen.

**Decision rule invoked:** *"Learned also fails: report that neither
fixed nor learned linear encodings predict held-out targets beyond the
training mean on these screens."* No promotion of a learned-encoding
result; no title / abstract update.

Matched-linear-truth ρ_fwd under each encoding's U (for reference):
K562 fixed 0.5798, footprint 0.3909; the matched linear generator recovers
signal that no fitted encoding on the real data does, corroborating the
target-held-out failure being a data-side property (identifiability at
these SNRs, plus the shared-mode dominance of training responses) rather
than an encoding-choice failure.

## Step 2 — Positive-control ensembles at matched amplitude

Command: `python reproduction/52_positive_control_ensembles.py`
Output: `results/recheck/F_step2_ensembles.json`

α_S is re-derived per (screen × ensemble) so median column-norm of
`S_true` matches the observed median. Draws with spectral abscissa of
`J_true` above zero are rejected and re-drawn; N ≥ 15 accepted per cell
(actual rejection rate 0.0 for every cell — the diagonal −1.5·I dominates
in every ensemble tested).

| Screen | Ensemble | α_S | Linear-truth nested-CV ρ (mean, SD, N=15) | Interaction-only cos (mean, SD, N=200) | Cross-replicate null (mean, SD) | Beats predict-zero |
|---|---|---:|---:|---:|---:|:-:|
| K562 essential | dense | 287.6 | 0.229 (0.005) | 0.956 (0.006) | −0.005 (0.035) | ✓ |
| K562 essential | sparse_10 | 467.8 | 0.123 (0.001) | 0.939 (0.011) | +0.005 (0.034) | ✓ |
| K562 essential | sparse_2 | 474.1 | 0.120 (0.002) | 0.773 (0.072) | −0.002 (0.039) | ✓ |
| K562 essential | rank_5 | 459.5 | 0.127 (0.003) | 0.957 (0.008) | +0.003 (0.035) | ✓ |
| K562 essential | block_modular | 440.6 | 0.134 (0.004) | 0.960 (0.006) | +0.000 (0.069) | ✓ |
| RPE1 essential | dense | 151.0 | 0.651 (0.017) | 0.615 (0.029) | +0.004 (0.035) | ✓ |
| RPE1 essential | sparse_10 | 263.7 | 0.387 (0.007) | 0.565 (0.041) | +0.002 (0.033) | ✓ |
| RPE1 essential | sparse_2 | 271.3 | 0.374 (0.006) | 0.304 (0.064) | −0.002 (0.035) | ✓ |
| RPE1 essential | rank_5 | 255.6 | 0.399 (0.009) | 0.614 (0.040) | −0.002 (0.034) | ✓ |
| RPE1 essential | block_modular | 236.4 | 0.424 (0.013) | 0.631 (0.030) | −0.002 (0.048) | ✓ |
| Jost 2020 | dense | 23.6 | 0.734 (0.028) | 0.348 (0.034) | +0.001 (0.034) | ✓ |
| Jost 2020 | sparse_10 | 36.8 | 0.616 (0.017) | 0.178 (0.040) | +0.002 (0.033) | ✓ |
| Jost 2020 | sparse_2 | 37.1 | 0.611 (0.020) | 0.079 (0.037) | −0.002 (0.032) | ✓ |
| Jost 2020 | rank_5 | 35.6 | 0.626 (0.026) | 0.203 (0.046) | +0.004 (0.033) | ✓ |
| Jost 2020 | block_modular | 33.6 | 0.652 (0.029) | 0.232 (0.040) | −0.000 (0.044) | ✓ |

**Verdict:** the linear truth beats predict-zero on every (screen × ensemble)
cell (15/15). The conclusion "the linear truth beats predict-zero at
matched SNR" is robust across dense, sparse-10%, sparse-2%, rank-5, and
block-modular ensembles.

The interaction-only Frobenius cosine sits well above the cross-replicate
paired null in every cell. Amplitude decreases in sparser ensembles
(sparse_2 pulls Jost from 0.35 → 0.08 and RPE1 from 0.62 → 0.30, both
still above the null band), but no ensemble flips the sign of the
conclusion.

## Step 3 — Random-panel distribution (K562 essential)

Command: `python reproduction/53_random_panel_distribution.py`
Output: `results/recheck/F_step3_random_panels.json`

Fit anchor-op once on the qualifying K562 essential targets (all with
≥ 60 cells; `n_qualifying_targets = 1581` after the pipeline's per-target
retention filter — the preregistration's "~1,740" estimate came from the
`≥ 60 cells` filter alone; the pipeline drops a further 159 targets whose
per-guide response passes the additional QC steps in
`measure_operator`). `Σ_ctrl` saved from that single measurement.
Fifty random 200-target panels are drawn from that measurement (fixed
`Σ_ctrl`, fixed basis W, fixed κ). Nested-CV under Table 1's recipe per
panel; N = 5 matched-linear-truth simulations per panel.

| Distribution | Median | 5th – 95th percentile | Min | Max |
|---|---:|---:|---:|---:|
| Real nested-CV ρ | 0.9988 | 0.9855 – 1.0054 | 0.9698 | 1.0076 |
| Matched linear-truth ρ (per-panel mean, N = 5) | 0.3700 | 0.3136 – 0.4416 | — | — |

**Preregistered criterion (fraction of panels with real ρ < 0.95):**
**0 / 50 (0.0%).** No panel drops below 0.95; the tightest panel real
ρ is 0.9698. The matched-linear-truth ρ sits well below the predict-zero
baseline in every panel (5th–95th percentile band 0.31–0.44).

The random-200 point estimate reported in the previous manuscript (§2.3
Fig 4a: real ρ = 1.00, matched linear-truth ρ = 0.43 (SD 0.011, N = 15))
is inside this distribution; the previous number sat at the upper end of
the 50-panel matched-linear-truth band (0.43 is above the 95th percentile
0.44 by less than one SD — same recipe, one draw). The preregistered
verdict against target selection stands: the failure is not driven by
picking one favourable random 200 out of the qualifying pool.

## Step 4 — Non-essential check

Not run. The Replogle genome-wide K562 h5ad is not available locally
(only `K562_essential_normalized_singlecell_01.h5ad` and
`rpe1_normalized_singlecell_01.h5ad` are on disk under `examples/data/`).
Per the preregistered fallback, the Limitations paragraph in the
manuscript takes the sentence:

> Both Replogle screens are essential-gene libraries dominated by a
> shared growth/stress response; whether the result holds for
> non-essential perturbations is untested.

