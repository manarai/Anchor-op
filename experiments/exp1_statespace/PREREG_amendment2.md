# PREREG amendment 2 — exp1 state-space, 2026-09-30

Branch: `experiments/exp1-statespace`. Amends `PREREG_exp1_statespace.md` (committed earlier today at `b04555e`) and is itself committed **before any fitting runs** on the new input representation. Scope: input data, arms, knockdown-scale definition, second sanity gate. Nothing from this document enters the v0.3.2 preprint.

## 0. Trigger

Verification run on the preprint's residual h5ad (`EXPERIMENT_LOG.md` 2026-09-30):

```
|mean(X_ctrl[:, g])| over the 188 target genes:
    min 3.87e-13, p05 4.54e-11, median 5.70e-10, p95 2.55e-09, max 3.83e-09
    all 188 targets have |mean| < 1e-6
```

The multiplicative A4 definition `u_z = mean_i[E(x_i · scale(target, 1 − κ)) − E(x_i)]` collapses to 0 for every linear encoder on this input (closed form `−κ · mean(X_ctrl[:, g]) · J[:, g] = 0`). scGPT also cannot ingest residuals. The PREREG as it stands cannot be executed on the preprint's h5ad.

## 1. Input data (replaces §2 of the original PREREG, A4 clause)

- **Primary data for all exp1 arms**: Replogle 2022 K562 essential-gene Perturb-seq *raw* single-cell counts, `K562_essential_raw_singlecell_01.h5ad` from Figshare Plus deposit 20029387 (same deposit as the residual h5ad). Download URL: `https://ndownloader.figshare.com/files/35773219`. Size 10.66 GB. Downloaded 2026-09-30 to `examples/data/`.
- **Cell / guide / target match**: identical cells, guides, and targets as the residual h5ad used in A1 (match by barcode / cell index). The script `experiments/exp1_statespace/task2b_lognorm_input_check.py` enforces the barcode match before any encoder is fit; mismatch halts the pipeline.
- **Normalisation**: per-cell CPM × 1e4 (scanpy `sc.pp.normalize_total(target_sum=1e4)`), then `sc.pp.log1p` (natural log). This is the scGPT input convention and the standard scverse pipeline. The method name, the `target_sum`, and the resulting non-zero-per-gene control-cell mean distribution are recorded in `EXPERIMENT_LOG.md` after the download finishes.
- **Counts to report** (in `TASK2b_lognorm_input_check.md`): n cells by category (non-targeting controls, perturbed, retained after ≥ 60 cells per target), n targets matched to the measurement bundle, median counts per cell, median counts per gene, percentage of genes with non-zero control-cell mean (expected close to 100 % on log1p-normalised counts).

## 2. Arms (replaces §1 of the original PREREG)

All arms are built on **identical log1p-normalised input** at the shared `d = 30`:

| Arm | class | d | input | role |
|---|---|---:|---|---|
| **PCA-lognorm** | `PCARep(dim=30)` | 30 | log1p-normalised K562 raw counts, fit on NT controls | **reference encoder** (replaces PCA-30 anchor-op reference) |
| **FA-lognorm** | `FARep(dim=30)` | 30 | same log1p input | second linear encoder for internal consistency |
| **scGPT→30** | `ScGPTRep(d_out=30)` | 30 | same log1p input → scGPT native 512 → PCA head to 30 | the encoder claim under test |
| *PCA-on-residuals (A1)* | `PCARep(dim=30)` | 30 | residual h5ad | reference row only; A1 already PASSED (ρ real 0.9658, matched linear 0.1818). Not used in the comparator. |

The **encoder claim** of exp1 (per §5 of the original PREREG) compares **scGPT→30 vs PCA-lognorm**, not vs the paper-1 PCA-on-residuals.

## 3. Knockdown-scale finite difference definition (replaces A4 §2 of the original PREREG)

For every arm (linear or scGPT), on log1p-normalised control input:

    δx_i[g] = log1p((1 − κ) · expm1(x_i[g])) − x_i[g]      (per cell i)
    u_z = mean_i [ E(x_i^perturbed) − E(x_i) ]
          where x_i^perturbed is x_i with the g-th entry replaced by log1p((1 − κ) · expm1(x_i[g]))

- Scaling is done in count space (the counts-side `(1 − κ)` multiplication); the embedding is then evaluated on the re-logged input.
- Primary κ = 0.7; sensitivity at κ ∈ {0.5, 0.9}. Control cells only; perturbed cells never enter.
- Code: `StateSpace.knockdown_scale_difference(X_ctrl, g, κ, input_space="log1p")`, default `input_space="log1p"`.

### Linear-arm consistency clause (replaces the earlier "−κ · mean · J @ δ_g" clause)

For a linear encoder `E(x) = (x − μ)·J.T`, the knockdown-scale FD under `input_space="log1p"` has the closed form

    u_z = mean_i[ log1p((1 − κ) · expm1(x_i[g])) − x_i[g] ] · J[:, g]

Enforced by `tests/test_state_space.py::test_linear_knockdown_scale_log1p_space` across `(PCARep, FARep) × κ ∈ {0.5, 0.7, 0.9}` to `atol = 1e-10`. Full suite currently **25 passed, 1 skipped**.

### Non-triviality assertion on real data

Before any Task 4 fitting, run `experiments/exp1_statespace/task2b_lognorm_input_check.py`:

- Compute `knockdown_scale_difference(X_ctrl, g, κ=0.7, input_space="log1p")` for every one of the 188 target genes on the log1p-normalised K562 raw-counts h5ad.
- Compute per-target `‖u_z‖`.
- Assertion: **‖u_z‖ > 0 on at least 95 % (179 / 188) of targets** for the PCA-lognorm arm. Report the per-target table and the fraction with `‖u_z‖ > 1e-6`.

If < 95 % pass, halt and report — the input preprocessing is not consistent with the knockdown-scale definition.

## 4. Second sanity gate — PCA-lognorm nested CV (new)

Script: `experiments/exp1_statespace/A2_pca_lognorm_gate.py`.

Same target-grouped 5-outer × 3-inner nested-CV recipe as A1, rank grid `{0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}`, seed 20260930, 15 matched-linear-truth replicates. Inputs: the measurement-bundle equivalent of the K562 essential measurement, but with the program basis fit on log1p-normalised NT controls (not on residuals). The `S` and `U` for each guide are built from the log1p-normalised raw counts (per-guide mean − per-NT-control mean, projected into the new basis).

**Gate**: on log1p-normalised input, PCA-lognorm must either reproduce the A1 numbers (real ρ ≈ 0.96, matched-linear ≈ 0.18, same tolerance 0.03) or differ measurably. The outcome is reported in `A2_pca_lognorm_gate.json` and discussed in `TASK2b_lognorm_input_check.md` *before* any scGPT comparison. The user reviews the gate result; the scGPT arm only runs after the review and Task 4 compute confirmation.

## 5. Everything else unchanged from the original PREREG

A2 (matched-SNR per arm), A3 (training-mean governs metrics, revised 2-SD rule), A5 (D̂ primary = split-half control replicate covariance), A6 (null is the expected outcome; report either way) are unchanged. Pass criterion §5 of the original PREREG — scGPT→30 beats PCA-reference (now PCA-lognorm) at k = 5 and k = 10 with 95 % CI excluding zero, beats training-mean by > 2 outer-fold SDs, and beats the best non-operator baseline — stands.

## 6. Deliverables for this amendment

Committed in one batch, before any new fitting on the log1p input:

1. `src/anchorop/state_space/base.py` — `knockdown_scale_difference(input_space=…)` with the log1p path.
2. `tests/test_state_space.py` — log1p consistency test (6 cases) + residual-zero test (1 case).
3. `experiments/exp1_statespace/EXPERIMENT_LOG.md` — residual-mean verification, CELLxGENE census audit (0 hits), scGPT_human SHA-256s and `args.json` lineage.
4. `experiments/exp1_statespace/PREREG_amendment2.md` — this file.
5. (deferred until raw-counts download finishes) `experiments/exp1_statespace/task2b_lognorm_input_check.py` + `TASK2b_lognorm_input_check.md` + `A2_pca_lognorm_gate.py` + `A2_pca_lognorm_gate.json`.

The commit hash of this amendment is recorded in `EXPERIMENT_LOG.md` immediately after it lands. Task 4 (fitting arms × seeds × k) still does not run without user authorisation.
