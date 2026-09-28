# Manuscript restructure plan (draft)

**Trigger:** F1–F3 all confirmed in `RECHECK_LOG.md`. The published
negative-recovery result is a **signal-amplitude** conclusion, not a
statement about the estimator or about biological Jacobian
recoverability. This plan lays out the reshape; **no full-text rewrite
will happen until you sign off.**

**Sign-off checklist.** Before I touch a section, I need your explicit
"go" on:
1. the revised central claim (below),
2. the keep / cut / rewrite list (below),
3. the figure list (below),
4. the abstract (below).

Redirect any item and I'll re-plan.

---

## 1. Revised central claim (one sentence)

**Under target-held-out prediction with nested cross-validation, the
projected additive-input encoding `u_g = −κ_g Wᵀδ_g` cannot reach
held-out ρ below 1 — the predict-zero baseline — on any of the three
Perturb-seq screens tested (K562 real ρ = 0.96, 5-fold SD 0.02, vs
matched linear-truth 0.18; RPE1 1.00 vs 0.53; Jost 1.00 vs 0.72 at d
= 30, and 1.01 vs 0.32 at d = 5 where the identifiability regime is
comfortably overdetermined), while the same estimator on a matched-SNR
linear ground truth over the same U and σ reaches ρ well below 1 in
every case. Under the same matched-SNR positive control the estimator
recovers a known operator's interaction-only structure (K562 cos_int
= 0.98, RPE1 cos_int = 0.74 against a shift-1 cross-replicate null
near 0.003), so noise dominance alone does not preclude recovery.**

**Two secondary findings kept in the manuscript.**
- **Jost within-target dose interpolation.** Guide-level (siblings
  leaked) ρ = 0.66. Labelled as a distinct diagnostic that predicts
  along known target directions rather than predicting new
  perturbations. Not a recovery statement about the operator.
- **κ-quality contrast between designs.** Direction-only refits show
  Replogle's detection-rate κ is uninformative (column-normalization
  barely moves ρ) while Jost's count-based κ carries real information
  (column-normalization worsens ρ, and also costs the linear-truth
  control substantially on Jost). Reading (c) — κ-proxy mis-scaling —
  is ruled out as a *sufficient* explanation on all three, but κ
  quality is a real cross-dataset difference worth reporting.

**Selection-bias robustness (F7).** K562 random-200 refit
(`reproduction/42_f7_k562_random200.py`): resampling 200 of the 1,740
qualifying targets uniformly at random gives real nested-CV ρ = 1.005
vs matched linear-truth 0.433 — same qualitative outcome as the
notebook's top-200-by-count selection (0.96 vs 0.18). The failure is
not created by the top-cell-count selection.

**rel_diff.** Preregistered `rel_diff ≤ 0.25` threshold is unreachable
at every tested SNR regardless of ground truth. Documented in
`PREREGISTRATION_AMENDMENT.md` as a design flaw in the preregistration.

**Analysis freeze — 2026-09-27.** No further analyses before the
manuscript is written. The remaining mechanism questions (footprint
encoding on Replogle, within-target concordance on Jost with a
split-half ceiling) move to Open Questions (§5).

**Wording caveats to keep the claim honest about what F4 supports.**
The mechanism — "the input encoding is the binding constraint" — is one
hypothesis consistent with F4. It is not the *only* hypothesis. There are
at least three readings, and each is distinguished by a **different**
test — an earlier version of this plan claimed within-target concordance
separated (a) from (c); it does not.

  (a) **Projection failure.** `Wᵀδ_g` puts ~99% of the target-direction
      unit vector outside the 30-PC subspace; the encoded input has
      little response-relevant signal.
      **Distinguishing test:** footprint encoding `u_g ∝ Wᵀ Σ_ctrl δ_g`
      or a gene-space model on a small target set.
  (b) **Dose-dependent non-linearity.** The linear settled-state model
      `S = −J⁻¹U` is the wrong shape for the assay's actual dynamics
      (saturation, dose-response, etc.).
      **Distinguishing test:** within-target guide-replicate direction
      concordance. Under any additive-linear model, every guide for one
      target moves the system along the same direction — only the
      amplitude differs. So a low within-target direction concordance
      is evidence *for* (b), and does not on its own bear on (a) vs (c).
      Requires the descriptive-concordance hardening extension to be
      run on Jost.
  (c) **κ-proxy mis-scaling.** The `detection_rate` κ proxy on Replogle-
      essential's pre-scaled data may not track actual input strength;
      response amplitude then follows fitness effects rather than
      knockdown efficiency. `κ` only rescales columns of `U`, so
      direction is untouched.
      **Distinguishing test:** *direction-only* held-out ρ. Refit
      after column-normalizing `S` and `U` (or with a free per-guide
      scale). If ρ still fails, (c) cannot be the explanation, since
      κ only affects scale.

The load-bearing sequence for pulling apart the three readings:
  1. Run direction-only ρ on the Replogle pkls we already have. Minutes.
     Settles (c) either way. If ρ improves substantially under
     direction-only fitting, (c) is a live candidate and κ-free tests
     are needed downstream; if it doesn't, (c) is out and we move on.
  2. Repeat direction-only ρ on Jost (once its measurement pkl is
     produced) with count-based κ — an independent, more trusted
     version of the same test.
  3. Run the concordance hardening on Jost — separates (b) from
     {a, c}, in the direction that within-target concordance well
     above the between-target null and near the split-half ceiling
     argues *against* (b), which then puts weight on (a).
  4. Footprint-encoding comparison — distinguishes (a) directly.

All four tests are cheap-to-medium. Only (1) is runnable today; (2)
depends on the Jost measurement pkl; (3) depends on the split-half
ceiling extension; (4) is a small script change.

Also: the "Replogle" qualifier is load-bearing. Without Jost's measured
`S`, the same-SNR real-vs-linear ρ comparison has not been run on Jost,
so a Perturb-seq-wide claim overreaches on current evidence.

Compare the old central claim (essentially "operator recovery is
practically absent at Replogle scale"), which is still true at the
published α = 1 amplitude but rests entirely on the fact that the
noise-free simulated response `‖S‖ ≈ 0.026` is ~350× smaller than the
observed response `‖S‖ ≈ 9.30`. A matched-signal simulation recovers the
operator (cos ≈ 0.99 at α_S ≈ 369 for K562). The paper's original
negative result is therefore an SNR conclusion, and reads as *stronger*
when reframed as: real data has plenty of signal per column, but the
additive-input encoding fails held-out prediction anyway.

## 2. Keep / cut / rewrite

Section by section. "Keep" = no change; "trim" = shorten under the new
central claim; "rewrite" = new material; "cut" = remove entirely.

### Cut
- **Table 2 (§2.5 / §3.4) — cell-density projections** (68k, 1.7M
  cells/guide, cos_1 crossings by ensemble). All predicated on the
  σ → cells/guide extrapolation from a positive control that doesn't
  match the observed SNR. Under the new framing the density figure is
  no longer load-bearing.
- **Fig. S18 stability-shift sweep (in its current form)**. Replace with
  the α_S SNR sweep (see figure list). The c-dependence discussion is
  interesting but should not be a *headline* Discussion beat.
- **The 68k / 1.7M and 189 / 2,725 cell-count projections** in the
  Discussion. Same reason.
- **The "densities that our default c = 1.5 estimate declares
  subcritical" thread** in §3.1 and Discussion — it survives only as a
  side note.

### Rewrite
- **Abstract.** New abstract below (≤ 250 words). Signal-amplitude
  framing; recovery-at-matched-SNR is the positive result; encoding
  failure at real-data SNR is the negative result; introduces F4 (shared
  mode, target-input orthogonality) and F5 (Jost U rank).
- **§1 Introduction.** Add a paragraph on the `u_g = −κ_g Wᵀδ_g`
  amplitude budget: ‖Wᵀδ_g‖ ≈ 0.09 (K562) at d = 30 puts ~99% of the
  target-direction unit vector outside the 30-PC subspace even when the
  target *is* in the HVG matrix. Motivates the encoding-failure story.
- **§2.2 "no draw-specific recovery" → "encoding fails held-out
  prediction at matched SNR"** (new §2.2). The α_S sweep is the load-
  bearing figure (see Fig. 2 in the new figure list). Cross-replicate
  null still relevant but as a supporting comparator, not the headline
  test.
- **§2.6 linearity diagnostic**. Rewrite around the F3 comparison: real
  data ρ ≈ 1.11 / 1.21 vs matched-SNR linear-truth ρ ≈ 0.18 / 0.53.
  Explicitly rule out overfitting (900 params, ~150 guides; matched-SNR
  linear-sim ρ is 0.18, not > 1). This is the strongest form of the
  linearity result and the reviewer flagged that the current §2.6
  compares against the *wrong* baseline.
- **§2.7 Jost.** Rewrite for two claims: (a) Jost's U has rank 25 at
  d = 30 — the identified subspace is at most 25-dimensional, not d;
  §2.7's "condition 55.6" needs a fresh derivation under the corrected
  gate. (b) Jost's median ‖u‖ = 0.068 is 2.3× K562's — the current
  "same σ" comparison is not a same-SNR comparison; the redo runs on a
  common SNR axis. Retain the direct-Jost recovery result at Jost's own
  σ (statistically detectable full-operator paired difference).
- **§3.2 Discussion / mechanism section**. Rewrite around F4: response
  is dominated by a shared mode (top SV 51%), target-input orthogonality
  puts `|cos(s_g, u_g)|` at chance (~0.12 vs chance 0.118), and the ρ
  gap is not rescued by removing the shared mode (F4 confirms 1.16 /
  1.23 after mean-response projection). This is the *mechanistic* story
  the paper doesn't currently tell.
- **§3.3 "constructed proposition" on hard-clamp derivation**. Keep,
  but sharpen with F4 as motivation: the empirical failure mode is
  consistent with the constructed proposition rather than a general
  impossibility. Also add the F5 identifiability-gate note.

### Add
- **New §2.3 F4 material** — shared-mode dominance, target-input
  orthogonality, footprint-encoding note. Small figure (see Fig. 3
  new).
- **New §2.4 F5 material** — rank-deficient U cases (Jost); gate fix.
  This is a paragraph, not a section.
- **New Discussion beat: what would fix it?** — three concrete
  suggestions:
  (a) input encoding that better projects the target direction
      (`u_g_foot ∝ Wᵀ Σ_ctrl δ_g` co-expression footprint; exploratory in
      F4);
  (b) higher rank(U) libraries (Jost's wide-κ design does help at Jost
      SNR, per §2.7 direct result);
  (c) a target-aware basis (per-target consensus modes; the shared-mode
      finding motivates it).

### Keep
- **§1.1 scope statement.** Already aligned with the new framing.
- **§2.1 identification / range(S) discipline.** Positive result;
  unchanged.
- **Methods §4.1–§4.7.** Except: §4.4's per-entry SNR context paragraph
  (already patched to reference F1 recheck); §4.6 needs the "rank-5 B, C
  scaled by 1/√d" and "fixed −1.5I shift" corrections (reviewer flagged;
  code-vs-prose mismatch); §4.7 unchanged.
- **All identifiability-discipline claims** (TSVD-vs-Tikhonov,
  `is_hard_projector`, rank_tol path). Positive contributions.

### Trim
- **§2.5 matched-scale linearity failure signal.** Keep the finding
  (real data fails held-out prediction where matched-SNR linear truth
  passes) but the ~48k cells/guide projection is now moot.
- **Discussion §3.4 Design implications.** Cut the numbers, keep the
  spirit ("do this control on your data").
- **Table 1 / N counts.** Consolidate to one N = 200 story (or split
  N = 200 empirical null from N = 15 per-direction breakdown with an
  explicit table title clarifying which is which). The reviewer flagged
  the N = 15 vs N = 200 mismatch.

## 3. Figure list (proposed)

Each figure with its generating script (existing or new).

| # | Title | Generating script |
|---|---|---|
| Fig. 1 | Pipeline schematic + identification discipline | `reproduction/00_fig0_pipeline_schematic.py` (existing) |
| Fig. 2 (**new headline**) | α_S signal-scale sweep: cos_full and cos_1 vs α_S at K562 σ, dense ensemble; data-implied α_S = 369 marked; the observed-vs-published-simulation gap | new: `reproduction/40_fig2_alpha_sweep.py` (build from `reproduction/30_recheck_F1_F4.py` output `F2_*.json`) |
| Fig. 3 (**new**) | Shared-mode dominance and target-input orthogonality: top-SV energy bar; between-target cos histogram vs `|cos(s_g, u_g)|` histogram at chance; ρ before/after mean-response projection removal | new: `reproduction/41_fig3_shared_mode.py` (build from `F4_*.json`) |
| Fig. 4 | Linearity diagnostic — real data vs matched-SNR linear truth | rework of Fig. 3 in the current draft: `reproduction/22_fig1_and_fig3_composites.py` + `reproduction/30_recheck_F1_F4.py` |
| Fig. S1 | rank_tol sweep | existing (`05_figS1_rank_tol_sweep.py`) |
| Fig. S9 | Replogle vs Jost per-target diagnostics | existing |
| Fig. S13 | Operator recovery per ensemble | existing (`13_operator_recovery.py`) |
| Fig. S17 | Noise-model sensitivity | existing |
| Fig. S19 | Per-dataset σ bootstrap | existing (`19_sigma_bootstrap_all_datasets.py`) |
| Fig. S20 (**new**) | Jost U singular spectrum at d = 30 (5 near-zero SVs) | new: `reproduction/42_figS20_jost_rank.py` (from `F5_jost_gate.json`) |
| — | Table 2 density projections | **remove** |
| — | Fig. S18 c-sweep (current form) | **remove or fold** |

## 4. Title and abstract (final draft; ~250 words)

**Proposed title.** "Additive-input encoding fails target-held-out
prediction on genome-scale Perturb-seq screens." (Alternate: "A
matched-geometry recovery check locates an encoding failure in
Perturb-seq operator estimation.") The previous title framed a
cell-density limit and no longer matches the claim.

**Abstract (245 words).**

Pooled Perturb-seq screens measure how cells respond to hundreds of
genetic knockdowns, and a common modeling step reads these responses
through a linear settled-state model with an additive input for each
perturbed gene. Under that model the response operator is identifiable
in closed form, `J·P_X = −U·S⁺`. We tested whether it predicts
held-out perturbations in three screens: Replogle K562 and RPE1
essential-gene screens, and the titrated Jost 2020 screen. We used
target-held-out prediction with nested cross-validation and compared
each screen against a positive control, a linear ground truth simulated
at the same signal-to-noise ratio, inputs and noise. The model failed
on all three. Held-out error stayed at the predict-zero baseline (ρ =
0.96, 5-fold SD 0.02; 1.00; 1.00 for K562, RPE1, Jost), while the
matched linear truth reached 0.18, 0.53 and 0.72. The failure persists
when targets are sampled at random (K562 real 1.00 vs matched linear
0.43), when input dimension is reduced so the problem is well
overdetermined (Jost d = 5 real 1.01 vs matched linear 0.32), and when
perturbation strength is removed from the fit, so mis-estimated
knockdown efficiency cannot explain it. It is also not a noise
problem: at the observed signal amplitude, the same estimator recovers
a known operator's interactions (interaction-only cosine 0.98 [K562]
and 0.74 [RPE1] against a null near zero). On Jost's titrated design
the model partly interpolates dose along known target directions
(ρ = 0.66, against 0.22 for the matched linear control). Whether the
failure comes from projecting targets onto expression programs or from
nonlinear dose responses remains open. **anchor-op** releases the
matched controls and an identifiability gate that checks the rank of
the inputs as well as the responses.

**Numbers cited in the abstract (all traced in
`results/recheck/number_provenance.csv`):**
- Real nested-CV ρ, K562 0.96 (5-fold SD 0.02; per-fold min 0.94,
  max 0.99; `results/recheck/F_nested_cv_rho.json`).
- Real nested-CV ρ, RPE1 1.00, Jost 1.00 (same file).
- Matched linear-truth ρ, K562 0.18, RPE1 0.53, Jost 0.72 (same).
- Random-200 K562 real ρ 1.00, matched 0.43
  (`results/recheck/F7_K562_random200.json`).
- Jost d = 5, real ρ 1.01, matched 0.32
  (`results/recheck/F_jost_d_sweep_grouped.json`).
- Interaction-only cosine at matched α: K562 0.98, RPE1 0.74, null
  ≈ 0.003 (`results/recheck/F_interaction_only_cosine.json`).
- Jost dose interpolation (guide-level folds): real ρ 0.66, matched
  linear-truth 0.22
  (`results/recheck/F3_Jost_grouped_folds.json`).

---

## 5. Open questions (Discussion section, not blockers)

Per the reviewer's freeze instruction, mechanism dissection moves
here as future work, not as blockers on this paper.

- **What distinguishes projection failure (reading a) from
  dose-dependent non-linearity (reading b)?** The data rule out
  κ-proxy mis-scaling as a sufficient explanation (reading c) on all
  three datasets. Two candidate next experiments:
    - *Footprint encoding* `u_g ∝ Wᵀ Σ_ctrl δ_g` on Replogle. Tests
      (a) directly by rewriting the input to include the target's
      co-expression footprint rather than a one-hot. `Σ_ctrl` was
      exported by the F7 refit (`results/k562_random200_sigma_ctrl.npz`)
      so this is a small script.
    - *Within-target guide-replicate direction concordance on Jost*
      against a between-target null and a within-guide split-half
      noise ceiling. Under any additive-linear model, all guides for
      a target move along the same direction. Low within-target
      concordance below the noise ceiling and near the between-target
      null is evidence for (b). Requires a ~30-line extension to
      `backed_exploratory_response_analysis.py` and a run on Jost.
- **Jost NMF-basis sensitivity.** The Jost F3 result was fit in a PCA-
  on-controls basis to match Replogle. `reproduction/37_recheck_F3_jost_end_to_end.py
  --basis nmf_qr` is available as a sensitivity check. Not blocking.
- **Bucket-C items from Round 2** (cross-replicate-null decomposition,
  equivalence margin for the noise-model comparison, Path A vs Path
  B). Author-decision items; independent of the mechanism story.

These are labelled as **next-paper** work; the current paper's central
claim ("here's a robust failure, here's what it isn't, and here's
what would tell the remaining explanations apart") is complete on the
numbers already in hand.

## Notes

- Ground rule for the rewrite: **no numeric claim without a source
  entry in `results/recheck/number_provenance.csv`.** Anything without a
  source gets flagged with `SOURCE_MISSING` and either sourced or
  removed before the section is signed off.
- Ground rule for prose: **no inline file paths in the manuscript
  body.** Traceability lives in the CSV and in the Code-availability
  section's "Number provenance" pointer; sentences quote rounded
  numbers (e.g. "~140-fold", "cos_1 = 0.09 at c = 3.0"). The CSV
  carries the un-rounded values.
- **Reading (c) settled — 2026-09-27.** Direction-only held-out ρ on
  the Replogle pkls and on Jost. On Replogle the detection-rate κ
  proxy is uninformative; on Jost the count-based κ is informative.
  κ-proxy mis-scaling ruled out as a sufficient explanation on all
  three. Live readings: (a) and (b).
- **Blocker 1 landed — Jost F3 end-to-end.**
  `reproduction/37_recheck_F3_jost_end_to_end.py --basis pca_controls`
  produced `results/jost_measurement.pkl` and
  `results/recheck/F3_Jost_pca_controls.json`. σ_per_sgRNA = 0.066 in
  Jost's PCA basis; `input_subspace_dim = 25`,
  `full_domain_identified = False`.
- **Blocker 2 landed — nested-CV regularized ρ.**
  `reproduction/40_nested_cv_rho.py`, output
  `results/recheck/F_nested_cv_rho.json`. Target-grouped outer folds
  on all three; inner CV picks TSVD rank. Real ρ: K562 = 0.96, RPE1 =
  1.00, Jost = 2.23. Matched-SNR linear-truth: 0.18, 0.53, 0.69.
  Replogle's earlier > 1 ρ was noise amplification; even at best
  regularization it stays at 0.96/1.00.
- **Blocker 3 landed — Jost d-sweep.**
  `reproduction/41_jost_d_sweep_grouped.py`, output
  `results/recheck/F_jost_d_sweep_grouped.json`. Truncate to d ∈
  {5, 10, 15, 20, 25, 30}. At d = 5 (comfortably overdetermined,
  matched linear-truth ρ = 0.32), real Jost ρ = 1.18 — the encoding
  fails independent of design underdetermination. Cross-dataset
  ranking retracted throughout.
- **Blocker 4 landed — F7 K562 random-200 refit.**
  `reproduction/42_f7_k562_random200.py`, output
  `results/recheck/F7_K562_random200.json` and
  `results/k562_random200_measurement.pkl`. Real nested-CV ρ = 1.005
  vs matched linear-truth ρ = 0.433 — same qualitative outcome as
  top-200-by-count. Selection bias is not the source of the failure.
  `Σ_ctrl` saved to `results/k562_random200_sigma_ctrl.npz` for any
  future footprint-encoding follow-on.
- **Analysis freeze.** No further runs before the manuscript is
  written. The three items previously listed as blockers move to
  Open Questions:
    - Descriptive-concordance hardening + Jost run (separates reading
      b from {a, c}).
    - Footprint encoding on Replogle (tests reading a directly).
    - Optional Jost NMF+QR basis sensitivity check.
- The Bucket-C items in `REVISION_NOTES.md` (cross-replicate-null
  decomposition, equivalence margin, Path A vs Path B) still need your
  explicit decision independent of this restructure.
- The reviewer's proposed references (Gardner 2003, Kholodenko 2002,
  Tegnér 2003, Hyttinen 2012, Ahlmann-Eltze 2025, King 2026) are added
  as *proposed* entries in the References section and must be verified
  before submission. Gardner and Kholodenko are reviewer high-
  confidence; the other four are memory-only and need DOI lookup.
