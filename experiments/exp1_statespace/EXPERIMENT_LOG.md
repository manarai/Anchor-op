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

## PREREG committed — 2026-09-30

Commit hash: **`b04555ea4d71910688ba9b230f7384efa4c2cf20`** on branch `experiments/exp1-statespace`.

Preregistered artefacts: `PREREG_exp1_statespace.md`, `TASK2_data_and_leakage.md`, `A1_reproduction_gate.{py,json}`, this `EXPERIMENT_LOG.md` with the A4 revision subsection. All five files land in the same commit, before any fitting runs.

A1 reproduction gate verdict (recorded in `A1_reproduction_gate.json`): **PASS**. Real K562 target-held-out ρ = 0.9658, matched linear-truth ρ = 0.1818, both within tolerance 0.03 of the preprint's Table 1 reference (0.96, 0.18).

**Task 4 is on hold** until user compute confirmation (per plan). The next steps on this branch (when Task 4 is authorised):

1. `pip install scgpt torch` into the already-created `anchor-op-scgpt` conda env.
2. Download the `scGPT_human` whole-pretrain checkpoint to `experiments/exp1_statespace/weights/scGPT_human/` (gitignored). Record the SHA-256, source URL, download date, and README text identifying it as the whole-human pretrain (not perturbation-fine-tuned).
3. Resolve the leakage status (institutional access to Cui et al. 2024 Methods, or CELLxGENE Discover census API query for Replogle 2022 K562 / GSE264667 / Figshare 20029387). Record sources + verdict here.
4. Run `ScGPTRep.check_decode_direction_feasibility(...)` on 5 K562 non-targeting-control cells; record per-cell cosine table and halt verdict.
5. Only then run Task 4 (fitting arms × seeds × k) per the PREREG recipe.

## 2026-09-30 — amendment 2 trigger: residual-mean verification

**Verification run** (preceding PREREG amendment 2):

```
python ad.read_h5ad examples/data/K562_essential_normalized_singlecell_01.h5ad
→ |mean(X_ctrl[:, g])| over 188 target genes:
    min = 3.87e-13, p05 = 4.54e-11, median = 5.70e-10,
    p95 = 2.55e-09, max = 3.83e-09
    n with |mean| < 1e-6: 188 / 188
```

The h5ad we use for the preprint is pre-scaled ctrl-residual data: every per-gene control-cell mean is at machine-precision zero. Under the multiplicative A4 definition `u_z = mean_i [E(x_i · scale(target, 1 − κ)) − E(x_i)]`, linear encoders collapse to 0 on this input (closed-form `−κ · mean(X_ctrl[:, g]) · J[:, g] = 0`). scGPT also cannot ingest residuals.

**Consequence**: all exp1 arms must be built on log1p-normalised raw counts, not on the preprint's residual h5ad. The PCA-on-residuals paper-1 arm (A1 gate) becomes a reference row only; the comparator question (does scGPT help?) is between scGPT→30 and PCA-lognorm, both at d = 30 on identical log1p-normalised input.

Code change (committed with amendment 2):

- `src/anchorop/state_space/base.py::StateSpace.knockdown_scale_difference` gains an `input_space` kwarg (`"log1p"` default, `"linear"` legacy). The log1p path un-logs the target gene with `expm1`, scales by `(1 − κ)`, re-logs with `log1p`, and feeds to `encode`.
- Tests `tests/test_state_space.py`:
  - `test_linear_knockdown_scale_log1p_space` (PCA × FA × κ ∈ {0.5, 0.7, 0.9}) enforces the log1p closed form `mean_i[log1p((1 − κ) · expm1(x_i[g])) − x_i[g]] · J[:, g]` to `atol = 1e-10`.
  - `test_knockdown_scale_zero_on_residuals` enforces that the `"linear"` path returns exactly 0 on zero-centred inputs — the motivation for the amendment.
- Full state_space suite: `pytest -q tests/test_state_space.py` → **25 passed, 1 skipped**.

The "non-trivial on at least 95 % of 188 targets on the real-data log-normalised h5ad" assertion lives in `experiments/exp1_statespace/task2b_lognorm_input_check.py` and runs after the raw-counts h5ad is on disk.

## 2026-09-30 — leakage audit, CELLxGENE census query result

**Query** (run inside `anchor-op-scgpt` conda env, `cellxgene_census 1.18.0`, latest census):

```python
with cxg.open_soma(census_version="latest") as census:
    datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    # 1,852 datasets in latest census.
    for q in ("Replogle", "perturb", "K562", "CRISPR"):
        hits = datasets[matches q in collection_name/dataset_title/dataset_h5ad_path]
```

Results:

| Query | Hits | Notes |
|---|---:|---|
| `Replogle` | **0** | — |
| `K562` | **0** | — |
| `CRISPR` | **0** | — |
| `perturb` | 7 | all mouse-thalamus development datasets (word "perturb" in collection title; not Perturb-seq) |

**Verdict**: Replogle 2022 K562 essential-gene Perturb-seq is NOT in the latest CELLxGENE census. Since scGPT_human was pretrained on CELLxGENE census (confirmed below), the scGPT whole-human pretrain **does not contain Replogle 2022 pretraining data**. Low leakage risk.

User will also pull the Cui et al. 2024 Methods text via BYU access for the explicit per-study enumeration; this CELLxGENE query is the independent check.

## 2026-09-30 — scGPT_human checkpoint on disk; lineage confirmed

Download: `gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y`, then individual-file fallback for `vocab.json`. Local path: `experiments/exp1_statespace/weights/scGPT_human/`, gitignored.

Files on disk (download date 2026-09-30):

| File | Size | SHA-256 |
|---|---:|---|
| `best_model.pt` | 196 MB | `6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922` |
| `args.json` | 1.3 KB | `c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60` |
| `vocab.json` | 1.3 MB | `acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f` |

`args.json` content (relevant fields, verbatim):

- `"data_source": "/scratch/ssd004/datasets/cellxgene/scb_strict/human"`
- `"save_dir": "/scratch/ssd004/datasets/cellxgene/save/cellxgene_census_human-May23-08-36-2023"`
- `"training_tasks": "both"`
- `"MVC": true` (Masked Value Completion; self-supervised)
- `"USE_GENERATIVE_TRAINING": true`
- `"input_style": "binned", "input_emb_style": "continuous"`
- `"n_bins": 51`
- `"nlayers": 12, "nheads": 8, "embsize": 512, "d_hid": 512`
- No `fine_tune`, `pert`, `perturbation`, or `GEARS` keywords.

**Lineage verdict**: this is the whole-human pretrain checkpoint, trained on the May 2023 CELLxGENE census human snapshot with self-supervised masked-value and generative objectives only. **Not fine-tuned on perturbation data**. Native encoder dim = 512; a PCA head to `d = 30` fits on control-cell embeddings per the exp1 PREREG.

Note on A4: `input_style = "binned"` combined with `input_emb_style = "continuous"` suggests the embedding path is continuous after binning. The autograd Jacobian path is reported as a sensitivity check once the loader is wired (A4 reporting only; not required for arm admission per the A4 revision).

## Leakage check — pending

Grep the scGPT pretraining-corpus manifest (CellxGene + the scGPT README) for:

- "Replogle" (2022, 2023)
- "K562" + "Perturb-seq"
- GEO accession GSE264667 (Replogle 2022 Cell)
- Figshare Plus deposit 20029387 (Replogle 2022 K562 essential h5ad we use)
- "perturb-seq" / "perturbation" datasets listed in the scGPT pretraining description

Source URLs and the verdict will be committed here before any encoder fitting runs. Checkpoint lineage (whether it was perturbation-fine-tuned) is confirmed from the scGPT release notes for the specific SHA-256 downloaded.
