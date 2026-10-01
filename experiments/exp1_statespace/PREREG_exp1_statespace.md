# Preregistration — Experiment 1: state-space representations × anchor-op

Branch: `experiments/exp1-statespace`.
Committed **before any fitting runs** per the preregistration-first rule used throughout the anchor-op repository.
Scope: this experiment is independent of the v0.3.2 NAR GAB preprint on `main`. Nothing from this document enters the preprint.

## 0. Background and primary question

Anchor-op paper 1 (`main` at tag `v0.3.2`) shows that the fixed program-space additive-input encoding `u_g = −κ_g · Wᵀ · δ_g` fails target-held-out operator-level prediction on three Perturb-seq screens, and that under a comparator panel of fixed / footprint / learned linear encodings the forward-direction win is confined to K562 essential and is modest (~ 7 % of the training-mean-to-linear-truth gap). Ahlmann-Eltze et al. 2025 report that a simple linear baseline matches current deep-learning models on single-perturbation prediction in Replogle-type data.

**Primary question for experiment 1**: does a pretrained state space (frozen scGPT embedding, fit a PCA head on controls to the shared `d = 30`) help the operator-level anchor-op fit beat a PCA-30 anchor-op reference at k ∈ {5, 10} anchored knockdowns, after a predict-training-mean baseline? Secondary: does FA-30 differ from PCA-30 on the same task?

**Expected outcome, prespecified**: given Ahlmann-Eltze et al. 2025 and anchor-op paper 1, a **null result** (scGPT→30 does not help beyond PCA-30 or the training-mean baseline) is the expected outcome. We report the result as a test of the projection-failure hypothesis of paper 1's §3.2, whichever way it goes.

## 1. Arms

Each arm is a `StateSpace` object from `src/anchorop/state_space/` fit on the K562 essential non-targeting-control cells.

| Arm | class | native dim | PCA head | notes |
|---|---|---:|---:|---|
| **PCA-30** (reference) | `PCARep(dim=30)` | 30 (= output) | — | the paper-1 anchor-op basis |
| **FA-30** | `FARep(dim=30)` | 30 (= output) | — | `sklearn.decomposition.FactorAnalysis`; `pinv(components_)` encoder |
| **scGPT→30** | `ScGPTRep(d_out=30, checkpoint_path=…)` | native scGPT hidden dim (record on load) | PCA to 30 on control cell embeddings | frozen whole-human pretrain checkpoint; perturbation-fine-tuned variants prohibited (see A4) |

All arms use seeds **{0, 1, 2}**. The matched `d = 30` is mandatory across all arms (A2). The native scGPT dim and the variance retained by the PCA head are recorded in `EXPERIMENT_LOG.md`.

## 2. Anchoring recipe (per arm, per seed)

For each `k ∈ {0, 2, 5, 10, 20, d − 1}`:

1. Fix a random draw of `k` training knockdowns (fixed seeds; **5 draws per `k`**).
2. On each draw, estimate the operator action `A` from the measured shifts of the `k` training knockdowns under the anchor-op model `J = (−D / 2 + A) · Σ⁻¹`, with `D̂` the noise covariance per A5.
3. Predict held-out knockdowns' state-space shifts as `δ̂_z = −J⁻¹ · u_z`, where `u_z` is the arm's representation of the perturbation (A4).
4. Score against measured held-out shifts.

### Perturbation input `u_z` per arm (A4, revised 2026-09-30)

- Linear arms (`PCARep`, `FARep`): closed-form `u_z = J · u_gene`, where `u_gene` is a sparse-unit vector on the target gene with sign negative.
- `scGPT→30`: **knockdown-scale finite difference**

      u_z = mean_i [ E(x_i · scale(target_gene, 1 − κ)) − E(x_i) ]

  on control cells only, with primary κ = 0.7, sensitivity at κ = 0.5 and 0.9. Perturbed cells never enter.

- Linear-arm consistency clause: `StateSpace.knockdown_scale_difference(X_ctrl, g, κ)` for `PCARep` / `FARep` equals `−κ · mean(X_ctrl[:, g]) · J · δ_g` exactly, enforced by test to `atol = 1e-10`.

### scGPT A4 feasibility (runs BEFORE any anchor-op fitting on the scGPT arm)

Via `ScGPTRep.check_decode_direction_feasibility(X_control, target_gene_idx, kappas=(0.5, 0.7, 0.9), n_subset=50, noise_threshold_rel=0.1, cos_threshold=0.9)` on five non-targeting-control cells for a representative target gene:

- (i) **Non-triviality**: ‖u_z‖ at primary κ > encoder run-to-run noise norm on identical input.
- (ii) **Stability**: pairwise cosine of `u_z` across κ ∈ {0.5, 0.7, 0.9} and across two disjoint random subsets of control cells > 0.9 on every pair.

Halt the scGPT arm only if (i) or (ii) fails. **No workaround.** If a continuous-input path exists, the autograd Jacobian is reported alongside for comparison but is not required.

### Noise term `D̂` (A5)

- **Primary**: embedding covariance from split-half control replicates on each arm's state space.
- **Sensitivity (where available)**: Jacobian push-forward `J_E · D · J_Eᵀ` at control cells. For `PCARep` / `FARep` this is closed form. For `scGPT→30` it is reported only if the A4 continuous-input path exists; otherwise skipped.

## 3. Required baselines (A3)

All reported in the same table as the arms:

- **predict-zero** — `δ̂ = 0`.
- **predict-training-mean** — `δ̂ = mean(training-knockdown shifts)` in the arm's state space.
- **additive / ridge from control expression** — a linear regression of state-space shifts on control-cell expression at the target gene, ridge λ by inner CV.
- **nearest-training-knockdown** by target-gene similarity (scGPT gene embedding when available; co-expression on control cells as a non-leaky fallback).
- **PCA-30 anchor-op** — the reference arm for the encoder comparison.

The **training-mean baseline governs all metrics** (A3). Any encoder claim is calibrated against training-mean, not against predict-zero.

## 4. Metrics

- Held-out **R²** between predicted and measured state-space shifts, per knockdown, then median across knockdowns with 95 % bootstrap CI.
- Held-out **cosine** between predicted and measured shifts, per knockdown. The **primary cosine metric is per-knockdown cosine minus the training-mean prediction's cosine** (the shared response mode inflates raw cosine; subtract the baseline's contribution).
- **R²-vs-k curve** per arm, with dashed lines for every baseline.
- **Seed stability of J**: Frobenius correlation of fitted `J` across seeds, per arm.
- **Unanchored whitened-symmetric agreement**: Frobenius similarity of `−½ · Σ^(−1/2) · D̂ · Σ^(−1/2)` to the arm's `J`, per arm.

## 5. Pass criterion (A3, revised)

The scGPT→30 arm **passes** iff all three hold at k = 5 **and** at k = 10:

- scGPT→30 beats PCA-30 on held-out R² with the 95 % bootstrap CI of the difference excluding zero;
- AND scGPT→30 beats the predict-training-mean baseline by **more than 2 outer-fold SDs** of the baseline;
- AND scGPT→30 beats the best non-operator baseline (one of the four A3 baselines, excluding PCA-30) at both k = 5 and k = 10.

Anything else: **FAIL for the encoder claim**; report which clause failed, with the full R²-vs-k curve.

## 6. Stop rule

**One run per arm per seed.** No tuning after results. If a bug is found after a result is seen, fix it, log the fix and reason in `RECHECK_LOG.md`, mark the affected values SUPERSEDED, and rerun all affected arms — matching the preregistration-after-the-fact discipline from paper 1.

## A1 (reproduction gate — must PASS before any encoder arm)

Already run. PCA-30 reproduces K562 real ρ = 0.9658 (reference 0.96) and matched-linear ρ = 0.1818 (reference 0.18), both within tolerance 0.03. See `TASK2_data_and_leakage.md` and `A1_reproduction_gate.json`. Verdict **PASS**; encoder arms proceed under this gate.

## A2 (matched-SNR linear-truth per arm)

Each arm runs its own matched-SNR linear-truth positive control, simulated in that arm's state space at the arm's own observed shift amplitude and noise, run through the identical anchor-op recipe. Each arm is interpreted only relative to its own control.

## A3 (training-mean governs metrics)

Covered in §3 and §5. R² and cosine are reported for predict-zero, predict-training-mean, each A3 baseline, and each arm. Primary cosine is per-knockdown cosine minus the training-mean prediction's cosine.

## A4 (scGPT Jacobian feasibility, revised)

Covered in §2. The revision 2026-09-30 replaces the autograd-vs-FD Jacobian check with the knockdown-scale finite difference on control cells and runs the (i) non-triviality + (ii) stability tests on 5 control cells across κ ∈ {0.5, 0.7, 0.9}. No workaround; halt the arm if either test fails. Reason: scGPT bins expression values, so infinitesimal steps through the binning operation are not a valid Jacobian. Full record of the A4 revision is in `EXPERIMENT_LOG.md` 2026-09-30 entry.

## A5 (noise term D̂ per arm, revised)

Primary: embedding covariance from split-half control replicates. Sensitivity: Jacobian push-forward `J_E · D · J_Eᵀ` where a Jacobian exists.

## A6 (expected outcome stated)

A null result (scGPT→30 does not beat PCA-30 and does not beat the training-mean baseline at k = 5 and k = 10) is the expected outcome given Ahlmann-Eltze et al. 2025 and anchor-op paper 1. We report the result as a test of paper 1's projection-failure hypothesis (§3.2 of the preprint). We will report the outcome whichever way it falls — a positive result (scGPT→30 passes) is a stronger result than a null and is also expected to be surprising if it occurs.

## Deliverables (committed on this branch; no `main` merge until the user approves)

Order:

1. `experiments/exp1_statespace/EXPERIMENT_LOG.md` — running log (A4 revision recorded 2026-09-30).
2. `experiments/exp1_statespace/REPORT_task0_task1.md` — Task 0 + Task 1 scaffold report (filed 2026-09-30).
3. `experiments/exp1_statespace/TASK2_data_and_leakage.md` — Replogle K562 data provenance + scGPT leakage audit (this document).
4. `experiments/exp1_statespace/A1_reproduction_gate.{py,json}` — A1 PCA-30 reproduction gate script + result.
5. `experiments/exp1_statespace/PREREG_exp1_statespace.md` — this file.

The commit hash of this preregistration is recorded in `EXPERIMENT_LOG.md` immediately after it lands. **Task 4 (fitting across arms × seeds × k) does not run until user compute confirmation.**
