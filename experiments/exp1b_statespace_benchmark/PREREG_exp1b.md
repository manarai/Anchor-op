# PREREG_exp1b — nonlinear state-space benchmark

Branch: `experiments/exp1b-statespace-benchmark` (forked from `experiments/exp1-statespace`).
Date committed: 2026-09-30, **before any arm fit**.
Supersedes nothing. **Not merged to `main`. Nothing from this experiment enters the anchor-op journal manuscript** (v0.3.4 on main, bioRxiv v0.3.2 frozen).

## 1. Question

Which state space, if any, lets the additive-input operator predict held-out knockdowns better than the paper-1 PCA(d = 30) and FA(d = 30, QR) reference arms?

Secondary: does the ranking match scJDO's embedding ranking (B0 extract above in `EXPERIMENT_LOG.md`)?

## 2. Prerequisite — satisfied

Part A corrections on main (v0.3.4, commit 7c6d6bc, 2026-09-30):
- PCA re-run through the `55_fa_table1_recheck.py` pipeline for apples-to-apples comparison with FA (same proxy-efficiency path, same in-basis σ and α_S).
- FA loadings QR-orthonormalized.
- Per-outer-fold SDs reported for Supp Table S3.

PCA and FA(QR) agree within fold SDs on all three screens (K562 real ρ 0.964 vs 0.968, matched 0.125 vs 0.120; RPE1 1.003 vs 1.003, matched 0.282 vs 0.289; Jost 0.998 vs 1.000, matched 0.726 vs 0.755).

**Reference arm for the pass criterion.** PCA and FA(QR) are empirically interchangeable on this task; we pick **FA(QR)** as the single reference arm because it is scJDO's default state space, so a reviewer comparing exp1b's conclusions against scJDO sees apples-to-apples.

## 3. Scope

- Screens: **K562 essential (primary), RPE1 essential (required replication), Jost 2020 (secondary; small, no claim rests on Jost alone).** Same h5ads, HVG recipe (3000 variance-based, force-included target genes), target set, min-cells-per-guide, min-knockdown-efficiency, rank_tol, outer/inner K, seeds as paper 1 Table 1.
- Latent d = 30 for every arm.
- Non-targeting control cells only for every fit. Perturbed cells enter the pipeline only via the measured `S` matrix after the arm's basis is frozen.
- **Nothing merges to main. Nothing from this experiment enters the anchor-op journal manuscript.**

## 4. Arms (all at d = 30)

Code: `experiments/exp1b_statespace_benchmark/arms.py` (committed in B1 before this PREREG).

| # | Arm | Input space | Ingestion / fit | Notes |
|---|---|---|---|---|
| 1 | PCA (log1p) | log1p | sklearn PCA on log1p NT controls | paper-1 reference, log1p variant |
| 2 | **FA (log1p, QR)** | log1p | sklearn FA + QR-orthonormalize loadings | **reference arm (scJDO default)** |
| 3a | LDVAE encoder | counts | `scvi.model.LinearSCVI(n_latent=30)` on raw counts, gem-group batch key; posterior-mean latent via `get_latent_representation()` | 3 training seeds (full benchmark) |
| 3b | LDVAE loadings | log1p | decoder loadings from the same trained LDVAE, QR-orthonormalized, used as a linear basis on log1p counts | reuses 3a's trained encoder — within-model comparison |
| 4 | scVI | counts | `scvi.model.SCVI(n_latent=30)` on raw counts, gem-group batch key; posterior-mean latent, no sampling | 3 training seeds |
| 5a | scGPT_human (fixed ingestion) | log1p | continuous-input head; np.argsort(..., kind='stable') for tie-breaking; target force-included at token position 0 for both ctrl and in-silico kd; PCA head to d = 30 fit on controls | deterministic given weights |
| 5b | scGPT (random weights, same ingestion) | log1p | identical architecture + ingestion to 5a, weights re-initialised with a fixed seed | pretraining control |

Knockdown definition (in silico, control cells only):
- log1p arms (PCA, FA, LDVAE-loadings, scGPT 5a/5b): scale target's log1p expression via `expm1 → multiply by (1 − κ) → log1p`.
- count arms (LDVAE-encoder, scVI): scale target's raw count by `(1 − κ)`; library size as observed; encode via posterior mean.
- κ ∈ {0.5, 0.7, 0.9}, primary κ = 0.7.
- Real perturbed cells are embedded with the same arm and the same ingestion as controls.

## 5. Protocol (identical to paper 1 §4.5)

- Target-grouped 5 outer × 3 inner nested CV.
- Rank grid including 0 for the inverse task: `{0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}`.
- Matched-SNR linear-truth control simulated **in each arm's own d = 30 space** with the arm's own σ (within-guide split-half bootstrap) and α_S (matched-median column-norm against a dense random `J_ref`).
- Baselines: predict-zero (ρ = 1) and predict-training-mean.
- ≥ 3 seeds for arms with training stochasticity (LDVAE encoder, LDVAE loadings via its encoder, scVI).

## 6. Metrics

Reported per (arm × screen × seed) and pooled across seeds (for stochastic arms):
- Inverse ρ (pooled across outer folds; per-fold SD).
- Forward ρ_fwd vs training mean (per-fold SD).
- Fraction of each arm's gap-to-matched-linear-truth closed on the forward task.
- Matched linear-truth ρ in that arm's own space (mean, SD across N = 15 reps).
- For VAE arms: SD across training seeds alongside fold SDs.

## 7. Pass criterion ("arm X improves the state space")

On **BOTH** K562 and RPE1:
- Forward ρ_fwd of arm X beats the FA(QR) reference arm by more than 2 outer-fold SDs **AND** beats the training-mean baseline by more than 2 outer-fold SDs.
- Arm X closes ≥ 20 % of its own gap to its own matched linear-truth control on the forward task.

Pretraining-specific pass (independent of 7): scGPT 5a beats scGPT 5b (random-weights twin) by more than 2 outer-fold SDs on both K562 and RPE1.

## 8. Ladder comparisons (reported regardless of pass/fail)

- LDVAE-encoder vs LDVAE-loadings — does a nonlinear encoder help *over its own linear decoder*?
- scVI vs LDVAE-encoder — does a nonlinear *decoder* help?
- scGPT 5a vs scGPT 5b — does pretraining help?

Each ladder reports both inverse and forward ρ with fold SDs; no multiple-comparison adjustment (the ladder is three prespecified pairs).

## 9. Secondary: Spearman vs scJDO ranking

See `EXPERIMENT_LOG.md § B0` for scJDO's panel (file + line refs) and ranking.

- Overlap arms: PCA, FA, LDVAE (map to LDVAE-encoder), scVI → n = 4.
- Report Spearman ρ with a 95 % CI from a bootstrap over the 4 overlap arms.
- Treated as suggestive, not inferential. The Spearman is a cross-task transfer question (scJDO scores developmental-trajectory diagnostics on bone-marrow hematopoiesis; exp1b scores target-held-out operator prediction on three Perturb-seq screens).
- Caveats recorded in EXPERIMENT_LOG.md § B0 (different d, different dataset, scJDO's FA is NOT QR-orthonormalized).

## 10. Feasibility gate (B2, pre-fitting)

Code: `experiments/exp1b_statespace_benchmark/feasibility.py`. Five clauses per (arm × target × κ); failing arms are dropped and reported:

- (a) determinism — identical input twice gives max |Δ| < 1e-6 in the d = 30 latent (posterior mean for VAEs; model.eval(), no sampling).
- (b) negative control — a gene with zero counts in all controls gives ‖u_z‖ < 1e-6.
- (c) non-triviality — ‖u_z‖ at κ = 0.7 above the cell-sampling noise floor (mean of 20 random disjoint 50-cell subset pairs).
- (d) stability — mean subset-to-subset cosine of u_z > 0.9 (same 20 pairs). For VAE arms also cross-seed cosine.
- (e) dose-grading — ‖u_z‖ strictly increases with κ ∈ {0.5, 0.7, 0.9}.

Targets: 6 by control-expression quantile (q10, q30, q50, q70, q90) + CDC27 (if in HVG feature set after force-include).

## 11. Leakage

- **LDVAE, scVI** — trained here on NT controls only. No pretraining corpus; no leakage.
- **scGPT_human (5a)** — pretraining corpus: CELLxGENE census snapshot (May 2023 `cellxgene_census_human-May23-08-36-2023`); checkpoint SHA256 `6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922`. Audit on exp1 (EXPERIMENT_LOG.md § Leakage) found 0 hits for Replogle, K562+CRISPR, Perturb-seq (7 mouse-thalamus-development hits only); scGPT training was MVC+generative self-supervised only, no `fine_tune`/`pert`/`perturbation`/`GEARS` keywords. Cui et al. 2024 Methods confirms no Perturb-seq fine-tuning on `scGPT_human`. **Low leakage risk**, cross-check documented in exp1.
- **scGPT random weights (5b)** — no pretraining, no leakage.

## 12. Expected outcome

Given Ahlmann-Eltze 2025 and paper 1 (now v0.3.4 on main), the expected outcome is **no arm passes the §7 criterion**. This is itself the finding worth reporting: operator-level target-held-out failure survives a broader set of representations than just PCA/FA.

If an arm does pass, the paper 1 story needs to be qualified: it is a property of the fixed-linear-encoding class, not of every reasonable state space.

## 13. Stop rule

One run. No tuning after results. Any post-hoc hyperparameter change invalidates the pass claim for that arm.

## 14. Compute venue

Feasibility (B2) runs locally. **Before the full benchmark, we stop and report the per-arm compute estimate** (VAE training × 3 seeds × 3 screens; scGPT embedding) so the user can confirm local vs cluster GPU. This stop point is enforced by B4.

## 15. Deliverables

- `experiments/exp1b_statespace_benchmark/EXPERIMENT_LOG.md` — per-step entries (B0 landed; B2–B4 to come).
- `experiments/exp1b_statespace_benchmark/PREREG_exp1b.md` — this document.
- `experiments/exp1b_statespace_benchmark/arms.py` — 7 arm classes (B1, committed).
- `experiments/exp1b_statespace_benchmark/feasibility.py` — 5-clause gate harness (B2, committed).
- `experiments/exp1b_statespace_benchmark/feasibility_table.json` — gate results (B4).
- `experiments/exp1b_statespace_benchmark/REPORT_exp1b.md` — arm × screen table, ladder comparisons, verdict against §7, Spearman vs scJDO (post-fit, gated on user confirmation of §14).
- One figure panel per screen.

Commit per step; push the branch; stop before the full benchmark.
