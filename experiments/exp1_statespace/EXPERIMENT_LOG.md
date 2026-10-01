# Experiment 1 — state-space representations × anchor-op

Running log for the "does a pretrained state space help under anchoring?" experiment. Branch: `experiments/exp1-statespace`. The v0.3.2 preprint on `main` is frozen; nothing from this experiment enters the NAR GAB submission.

## 2026-09-30 — Task 0 and Task 1 scaffolding

Branch cut from `main` at tag `v0.3.2` (commit `37e12bb`).

New package: `src/anchorop/state_space/`.

- `base.py` — `StateSpace` ABC with `name`, `dim`, `fit`, `encode`, `jacobian`, `decode_direction`. Shape conventions and semantic rules recorded in the module docstring.
- `linear.py` — `PCARep(d)` and `FARep(d)`. Both fit on non-targeting control cells via `sklearn.decomposition`. Closed-form constant Jacobian: `components_` for PCA, `pinv(components_).T` for FA. Per-cell Jacobian is a broadcast of that constant so the caller does not branch on linearity.
- `multiome.py` — `MultiomeRep` stub. Every call raises `NotImplementedError`.
- `scgpt.py` — `ScGPTRep(d_out)` scaffold with:
  - Lazy scGPT package import and checkpoint load (no hard dep on `torch` / `scgpt` for the rest of the catalog to work).
  - `ScGPTCheckpoint` dataclass recording checkpoint `name`, local `path`, SHA-256 hash, `native_dim`, and `continuous_input_supported` flag.
  - Three thin shims (`_load_scgpt_encoder`, `_scgpt_forward`, `_scgpt_continuous_forward`) that raise `NotImplementedError` and must be wired in to concrete scGPT release once the checkpoint is on disk.
  - **A4 feasibility check**: `check_jacobian_feasibility(X_control, n_cells=5, step_sizes=(1e-3, 1e-2, 1e-1), tol_cosine=0.9)`. If the checkpoint has no continuous-input path, the method returns a halt record immediately (no autograd through binning). If it does, autograd and finite-difference Jacobians are compared across step sizes on 5 cells; `halt_reason` is set when any (cell × step) cosine falls below `tol_cosine`. No workaround is attempted.

Tests: `tests/test_state_space.py`, 13 tests covering

- encode shape = `(n_cells, dim)` on PCA / FA.
- jacobian shape = `(n_cells, dim, n_genes)` on PCA / FA.
- Linear reps: `jacobian[0] @ u_gene == decode_direction(u_gene)` to float64 precision.
- Linear reps: `J[0] == J[-1]` (constant Jacobian).
- `PCARep` encode matches `sklearn.decomposition.PCA.transform` up to column sign.
- Fit-guard raises before use on PCA / FA.
- MultiomeRep raises `NotImplementedError` on every call.
- scGPT A4 feasibility test — **skipped** when `$SCGPT_CKPT` is unset (fine for a CPU-only dev machine); when set, runs on 5 cells with step sizes `{1e-3, 1e-2, 1e-1}` and asserts stability.

Full suite: `pytest -q` → **71 passed, 1 skipped**. The 59 pre-existing tests continue to pass.

### Deferred to Task 1 completion

The scGPT scaffold is in place but the actual loader (`_load_scgpt_encoder`, `_scgpt_forward`, `_scgpt_continuous_forward`) is not wired in because the checkpoint has not been downloaded locally yet. Deferred items:

- Install `pip install scgpt` and `torch` into the anchor-op env.
- Download the `scGPT_human` whole-pretrain checkpoint. Confirm it is NOT a perturbation-fine-tuned variant (A4 / #4 of the plan).
- Record `scGPT_human` SHA-256, native dim, and whether a continuous-input path exists in the scGPT release.
- Wire the three shim functions to the concrete scGPT API.
- Run `rep.check_jacobian_feasibility(...)` on five K562 non-targeting-control cells; log the per-cell cosine table across step sizes and the halt verdict.

A1 reproduction gate, Task 2 data + leakage, and Task 3 PREREG follow in that order on this branch.

## A4 revision — 2026-09-30, logged before PREREG commit

**Reason**: scGPT bins expression values before the first transformer layer. An infinitesimal step through the binning operation has zero derivative almost everywhere, so autograd through binning is not a valid Jacobian. Finite differences at a step larger than one bin partially work but confound the step size with the underlying bin structure.

**Revised feasibility check (replaces the Jacobian-vs-FD check in the original A4):**

For `ScGPTRep`, `decode_direction(u_gene, X, kappa=0.7)` is now a **knockdown-scale finite difference** on control cells:

    u_z = mean_i [ E(x_i · scale(target_gene, 1 − κ)) − E(x_i) ]

(multiplicative scale of the target gene's expression). Primary κ = 0.7; sensitivity at κ ∈ {0.5, 0.9}. Perturbed cells never enter.

The feasibility check (`ScGPTRep.check_decode_direction_feasibility`) is:

- (i) **Non-triviality** — ‖u_z‖ at primary κ must exceed the encoder's run-to-run noise norm on identical input (``σ_noise``, estimated by embedding the same control subset twice).
- (ii) **Stability** — pairwise cosine of the knockdown-scale direction across κ ∈ {0.5, 0.7, 0.9} and across two disjoint random subsets of control cells must exceed 0.9 on every pair.

Halt the scGPT arm only if (i) or (ii) fails. If a continuous-input path exists, the autograd Jacobian is reported alongside for comparison but it is not required.

**Linear-arm consistency clause** (added to A4 at the same time):
For `PCARep` / `FARep` the closed-form Jacobian is kept, and `knockdown_scale_difference(X_ctrl, g, κ)` is a method on `StateSpace` that for linear encoders reduces to the closed form

    u_z = −κ · mean(X_ctrl[:, g]) · J @ δ_g

Enforced by `tests/test_state_space.py` across `PCARep × FARep × κ ∈ {0.5, 0.7, 0.9}` to `atol=1e-10`. When the input is zero-centred residuals and `mean(X_ctrl[:, g]) = 0` by construction (the Replogle K562 essential h5ad we use is one such case), both sides vanish — the knockdown-scale definition assumes a positive-mean expression representation on the scGPT input path, which is what scGPT expects anyway. For the linear-arm anchor-op fit we continue to use the original `J @ u_gene` form via `decode_direction(u_gene)`.

**Noise term D̂ (A5 revision)**: the primary estimator is the embedding covariance from split-half control replicates; the Jacobian push-forward `J_E D J_Eᵀ` is a sensitivity check only where a Jacobian exists.

Code changes:

- `src/anchorop/state_space/base.py` — added `StateSpace.knockdown_scale_difference(X_ctrl, target_gene_idx, kappa)`.
- `src/anchorop/state_space/scgpt.py` — `decode_direction` now delegates to `knockdown_scale_difference`; `check_decode_direction_feasibility` runs the two (i)+(ii) tests on five control cells across κ ∈ {0.5, 0.7, 0.9}.
- `tests/test_state_space.py` — new parametric test `test_linear_knockdown_scale_matches_minus_kappa_mean_times_Jdelta` across (PCARep, FARep) × (0.5, 0.7, 0.9) = 6 cases. All pass to `atol=1e-10`.

Full suite after revision: `pytest -q` → **18 passed, 1 skipped** on the new state-space suite; the 59 pre-existing anchor-op tests continue to pass.

## Leakage check — pending

Grep the scGPT pretraining-corpus manifest (CellxGene + the scGPT README) for:

- "Replogle" (2022, 2023)
- "K562" + "Perturb-seq"
- GEO accession GSE264667 (Replogle 2022 Cell)
- Figshare Plus deposit 20029387 (Replogle 2022 K562 essential h5ad we use)
- "perturb-seq" / "perturbation" datasets listed in the scGPT pretraining description

Source URLs and the verdict will be committed here before any encoder fitting runs. Checkpoint lineage (whether it was perturbation-fine-tuned) is confirmed from the scGPT release notes for the specific SHA-256 downloaded.
