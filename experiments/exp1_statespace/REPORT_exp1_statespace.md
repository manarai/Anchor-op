# Experiment 1 report — state-space representations × anchor-op

Branch: `experiments/exp1-statespace`. **Not merged to `main`.** The v0.3.2 preprint on `main` is unchanged by this experiment.

## 1. Setup

Three state-space representations tested as the shared `d = 30` input space to the anchor-op fit on Replogle 2022 K562 essential-gene Perturb-seq (Figshare 20029387):

| Arm | encoder | status at closeout |
|---|---|---|
| PCA-30 (residual, paper 1 reference) | sklearn PCA fit on paper-1 pre-scaled control residuals | A1 reproduction gate PASS |
| PCA-lognorm | sklearn PCA fit on log1p-normalised raw-counts controls | A2 + A2b sanity gates PASS |
| scGPT → 30 | frozen scGPT_human whole-pretrain + PCA head to 30 | **A4 feasibility HALT** |

Preregistered 2026-09-30 in `PREREG_exp1_statespace.md` and `PREREG_amendment2.md`, committed before any fitting. All amendments and `A4` revisions are in `EXPERIMENT_LOG.md`. The amendments stand; no post-hoc edits were made to them after results.

## 2. Results

### 2.1 A1 reproduction gate (paper-1 PCA-30 on residual h5ad)

| Metric | A1 result | Paper-1 reference | Within tolerance ±0.03? |
|---|---:|---:|:-:|
| Real target-held-out ρ | **0.9658 ± 0.0241** | 0.96 | ✓ |
| Matched linear-truth ρ | **0.1818** | 0.18 | ✓ |

The branch's code reproduces Table 1 of the preprint for the PCA-30 arm. (`A1_reproduction_gate.json`.)

### 2.2 A2b PCA-lognorm sanity gate (re-anchored σ, α_S on log1p basis)

The h5ad used for the preprint is pre-scaled control residuals: all 188 target-gene control-cell means are at machine-precision zero (`|mean| ≤ 3.83e-9`). On this input a multiplicative knockdown-scale FD is 0 for every linear encoder, and scGPT cannot ingest residuals. Amendment 2 switched the whole exp1 pipeline to **log1p-normalised raw counts** from the same Figshare deposit.

With σ and α_S re-anchored on the log1p PCA-30 basis (within-guide split-half bootstrap for σ, matched-median column-norm for α_S):

| Setting | real ρ | matched-linear ρ | gap | σ | α_S |
|---|---:|---:|---:|---:|---:|
| A1 — residual (paper 1) | 0.9658 ± 0.0241 | 0.1818 | 0.78 | 0.240 | 369 |
| **A2b — log1p, re-anchored** | **0.9208 ± 0.0451** | **0.0773 ± 0.0019** | **0.84** | **0.0820** | **214.99** |

**Paper 1's operator-level failure holds on the log1p basis**. The log1p PCA-30 basis is better-conditioned for the matched-SNR control, so the gap is actually larger (0.84 vs 0.78). (`A2b_reanchored_sigma_alpha.json`.)

### 2.3 CELLxGENE leakage audit

Queried the latest CELLxGENE census (`cellxgene_census 1.18.0`, 1,852 datasets) for `Replogle`, `K562`, `CRISPR`, `perturb`:

| Query | Hits |
|---|---:|
| Replogle | 0 |
| K562 | 0 |
| CRISPR | 0 |
| perturb | 7 (all mouse-thalamus development, not Perturb-seq) |

scGPT_human's `args.json` records `data_source = /…/cellxgene/scb_strict/human` and `save_dir = cellxgene_census_human-May23-08-36-2023`, with `training_tasks = both` (MVC + generative, self-supervised only). No `fine_tune`, `pert`, `perturbation`, or `GEARS` keywords. **Low leakage risk**. User cross-checked against Cui et al. 2024 Methods via BYU access and confirmed.

### 2.4 scGPT A4 feasibility: HALT

With state-dict converted from flash_attn `Wqkv` layout to standard `in_proj` layout (0 unexpected keys; 10 missing keys are only `cls_decoder._decoder.*`, unused), scGPT_human runs on CPU. 200 K562 NT control cells → `ScGPTRep(d_out=30)`, PCA head retains **0.978** of 512-d native variance.

| Clause | Result |
|---|---|
| (i) non-trivial | PASS (‖u_z‖ = 0.0261 > forward-pass noise 0.0) |
| (ii) within-subset across κ | PASS (0.986, 0.986, 1.000) |
| (ii) **across two 50-cell subsets, same κ** | **FAIL — antiparallel** (κ=0.5: −0.740, κ=0.7: −0.760, κ=0.9: −0.767) |

**scGPT→30 arm halted per A4. No workaround.** (`A4_scgpt_feasibility.json`.)

## 3. Post-hoc diagnosis of the halt (closeout, not re-enabling)

Five items, all strictly diagnosis. The halt verdict stands.

### 3.1 Item 1 — tokenization drop-out

For CDC27 and five prespecified targets (10th / 30th / 50th / 70th / 90th percentile of control mean expression):

| Target | Control mean (log1p) | In top-1200 pre | **Dropped out at κ = 0.7** |
|---|---:|---:|---:|
| q10 CENPJ | 0.157 | 0.000 | 0.000 (never in top-k pre or post) |
| CDC27 | 0.404 | 0.105 | **0.105** |
| q30 PMPCB | 0.381 | 0.100 | 0.100 |
| q50 MED10 | 0.562 | 0.200 | 0.190 |
| q70 SMC4 | 0.976 | 0.590 | **0.550** |
| q90 NCL | 2.922 | 0.990 | 0.020 (robust until κ = 0.9: 0.345) |

Low-expression and mid-expression targets drop out of the top-1200 token set after κ-scaling in a large fraction of control cells.

### 3.2 Item 2 — "noise floor" at ratio 0.93

Mean ‖u_z‖ at κ=0.7 on CDC27 = 0.0319. Between-subset norm across 10 disjoint 25-cell subset pairs = 0.0342. **Ratio = 0.932**. The A4 "non_trivial" check as run compared against the deterministic forward-pass noise, which is **0** on identical input (confirmed in item 5a) — not against the cell-sampling-level variability that actually matters.

### 3.3 Item 3 — generality on 5 prespecified targets, same recipe

**0 / 5 pass A4 feasibility** at `n_subset = 50, κ ∈ {0.5, 0.7, 0.9}, cos ≥ 0.9`:

| Target | min subset cos |
|---|---:|
| q10 CENPJ | −0.642 |
| q30 PMPCB | −0.153 |
| q50 MED10 | −0.335 |
| q70 SMC4 | −0.028 |
| q90 NCL | **+0.399** (best case; still well below 0.9) |

### 3.4 Item 4 — CDC27 force-included at token position 0

Pinned CDC27 into position 0 of the token set (bypassing argsort). **PASSES.**

| Metric | CDC27 default (A4) | CDC27 force-included (item 4) |
|---|---:|---:|
| κ-cosines (within subset A) | 0.986, 0.986, 1.000 | 0.996, 0.980, 0.993 |
| **Subset cosines (A vs B at same κ)** | **−0.740, −0.760, −0.767** | **+0.995, +0.997, +0.999** |
| Stable? | ✗ | ✓ |

### 3.5 Item 5 — determinism, negative control, noise re-expression

- **(a) Determinism**: encode same 200 cells twice; **max |Δ| = 0** in both 512-d native and 30-d head. `model.training = False`; 37 Dropout modules, all `p = 0.0`. Pipeline is perfectly deterministic on identical input.

- **(b) CENPJ negative control**: CENPJ is never in top-1200 (item 1), so its in-silico knockdown should give `u_z = 0` exactly. Observed on 50 control cells: median 0, min 0, **max 0.535**; **9 / 50 cells have ‖Δ‖ > 1e-6**. Trace of the worst cell (23):
  - log1p-value positions that changed: **1** (CENPJ, as expected).
  - Top-k token-ID positions that differ between unperturbed and perturbed: **1,132 / 1,200**.
  - Top-k values identical to `atol = 1e-10`: True.
  - Padding mask identical: True.
  - Mechanism: `np.argsort(-|x|)` is **not stable** on ties; many near-zero or zero-expression genes tie, and a single-gene change (even one that stays out of the top-k) non-deterministically reshuffles which of the ties enter the trailing positions of the top-1200.

- **(c) Noise re-expression**: pipeline-determinism budget on CENPJ (max per-cell ‖Δ‖ = 0.5355) vs item 2 between-subset norm (0.0342) → **ratio 15.65**. The item 2 "noise floor" is dominated by **pipeline non-determinism from argsort tie-breaking**, not by cell-sampling variance.

(`A4_scgpt_item5_determinism.json`.)

## 4. Which conclusion holds

With the diagnosis in hand, the A4 halt is explained by **both (i) and (iii)** of the user's three candidates, with (iii) as the dominant mechanism and the apparent (ii) being a surface effect of (iii):

- **(i) Tokenization drop-out**: confirmed by item 1 (fraction dropped out ranges 0.00 to 0.55 across the quantile sweep) and sharply confirmed by item 4 (force-including CDC27 flips subset cos from −0.74 → +0.995).
- **(iii) Pipeline artefact — specifically, `argsort`-tie-breaking non-determinism in the top-k gene selection**: confirmed by item 5b and 5c. A single-gene no-op perturbation (CENPJ never in top-k) produces a 1,132-out-of-1,200 reshuffling of token IDs for the worst cell, and an embedding `‖Δ‖` 15× larger than the item 2 between-subset norm.
- **(ii) Knockdown response indistinguishable from cell-sampling noise**: *as initially described in item 2 this is TRUE on the surface* (ratio 0.93), but item 5 reveals the "noise" is 15× too large to be cell sampling — it is argsort-tie-breaking propagating a single-gene change into a nearly-complete reshuffle of the token set. (ii) is a measurement artefact of (iii), not an independent property.

The scGPT whole-pretrain embedding has structure the perturbation can hit when the target gene is reliably in the token set (item 4: `|cos|` ≈ 0.999). The failure to produce a stable `u_z` in A4 is not that scGPT "can't see" the operator-level signal in principle; it is that the ingestion path (top-`max_seq_len` gene selection with unstable argsort on an input with many ties) is incompatible with the knockdown-scale finite-difference definition in A4 for all but the most highly expressed targets.

## 5. Scope-respecting conclusion

**With this checkpoint and top-k tokenization, in-silico knockdown in scGPT embedding space did not give a stable, dose-graded input direction, so the encoder comparison could not run.**

- The halt is specific to: `scGPT_human` (May 2023 CELLxGENE census snapshot pretrain), `max_seq_len = 1200`, numpy's default (non-stable) argsort, and the knockdown-scale FD definition of `u_z` from PREREG amendment 2.
- No claim is made about pretrained foundation-model embeddings in general. No claim is made about scGPT with a different ingestion recipe (continuous-input head, HVG-fixed token set, force-included target gene, etc.).
- Paper 1's operator-level failure story is independently reproduced on the log1p PCA-30 basis via A2b (`gap = 0.84 vs 0.78`), consistent with the preprint.

## 6. Next-step options for the projection question that avoid this problem

**No implementation**, flagged for future paper design:

- **Gene-space operators restricted to targeted genes**. Fit the response operator directly on the ~1,581-dim subspace of targeted genes rather than on a `d = 30` program basis. Avoids the projection step tested in paper 1 §3.2 entirely, and does not depend on a cell-level embedding of control cells.
- **Jost within-target concordance** (described in paper 1 §3.2 "Dose non-linearity" / Methods §4.5c extensions): compare the direction of guide-level shifts within the same target across the 3–6 Jost sgRNAs against a between-target null and a within-guide split-half noise ceiling. This tests the linear-direction assumption on an orthogonal axis to the projection question.
- **Encoders with continuous-value inputs, not top-k binning**. Any encoder whose input depends on the full expression vector (not a top-k selection) is immune to the argsort-tie-breaking failure mode identified here. Examples: PCA / FA (already done, A2b), simple MLP on full expression, autoencoders trained with gene-order invariance.

## 7. Final status

- Branch `experiments/exp1-statespace` tip: this commit.
- v0.3.2 preprint on `main`: untouched. Nothing from exp1 enters it.
- Task 4 (fitting across arms × seeds × k): **not run**.
- scGPT → 30 arm: **HALT**. Does not enter Task 4.
- Linear-only Task 4: **not run** (paper 1 already covers PCA; A2b reproduces the paper-1 story on log1p).
- **No main merge**.

Deliverables committed on the branch:

```
PREREG_exp1_statespace.md          — original PREREG with A1–A6
PREREG_amendment2.md               — log1p amendment
EXPERIMENT_LOG.md                  — running log of all decisions + hashes
A1_reproduction_gate.{py,json}     — PCA-residual reproduction, PASS
TASK2_data_and_leakage.md          — data + initial leakage audit
task2b_lognorm_input_check.py
TASK2b_lognorm_input_check.{md,json} — log1p pipeline + non-triviality, PASS
A2_pca_lognorm_gate.{py,json}      — log1p PCA-30 gate, wrong anchors
A2b_reanchored_sigma_alpha.{py,json} — log1p PCA-30 gate, re-anchored, PASS
A4_scgpt_feasibility.{py,json}     — scGPT feasibility, HALT
A4_scgpt_diagnosis_posthoc.py
A4_scgpt_diagnosis.json            — items 1–4 of diagnosis
A4_scgpt_item5_determinism.{py,json} — item 5 of diagnosis
REPORT_exp1_statespace.md          — this report
weights/scGPT_human/*              — gitignored; local checkpoint only
```

Supporting code: `src/anchorop/state_space/*.py` (StateSpace ABC, PCARep, FARep, MultiomeRep stub, ScGPTRep with state-dict conversion loader, `_torchtext_shim`); `tests/test_state_space.py` (25 passed, 1 skipped — the skipped one is the obsolete autograd-Jacobian test from the original A4, kept skipped because A4 was revised to the knockdown-scale FD before any fitting).
