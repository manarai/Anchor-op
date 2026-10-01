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

## PREREG amendment 2 committed — 2026-09-30

Commit hash: **`0af210dc52b4f6442de34ce0f305b0dcb6340963`** on branch `experiments/exp1-statespace`.

Preregistered artefacts in this commit:
- `src/anchorop/state_space/base.py` — `knockdown_scale_difference(input_space='log1p'|'linear')`.
- `tests/test_state_space.py` — log1p consistency + residual-zero tests (25 passed, 1 skipped).
- `experiments/exp1_statespace/EXPERIMENT_LOG.md` — residual-mean verification, CELLxGENE census audit, scGPT_human checkpoint provenance.
- `experiments/exp1_statespace/PREREG_amendment2.md` — the amendment itself.

Deferred follow-up (pending raw-counts download):
- `experiments/exp1_statespace/task2b_lognorm_input_check.py` — barcode match + log1p normalisation + non-triviality assertion on 188 target genes.
- `experiments/exp1_statespace/A2_pca_lognorm_gate.py` — second sanity gate: PCA-lognorm nested CV under the amendment-2 recipe.

## 2026-09-30 — task2b + A2 results + scGPT install blocker

- Raw-counts download complete: `K562_essential_raw_singlecell_01.h5ad` (9.9 GB on disk, 310,385 × 8,563). Same obs columns and var_names as the preprint's residual h5ad.
- task2b non-triviality: 188/188 targets have ‖u_z‖ > 1e-6 at κ = 0.7 under log1p. Per-gene control-cell |mean| p05 = 0.11, median = 0.53, max = 4.07. **PASS**.
- A2 PCA-lognorm gate: real ρ = 0.9208 ± 0.0451 (A1 was 0.9658), matched-linear ρ = 0.1320 ± 0.0036 (A1 was 0.1818). Both arms carry the paper-1 operator-level failure story (gap real − matched = 0.79 vs A1's 0.78). The log1p basis is slightly stronger / picks lower ranks (median rank 2 vs A1's 3). Written up in `TASK2b_lognorm_input_check.md` for user review before Task 4.
- **scGPT runtime BLOCKED** by a `torchtext` ABI mismatch against torch 2.13. `import scgpt` fails inside `scgpt.tokenizer.gene_tokenizer` on `torchtext._extension._load_lib("libtorchtext")`. The scGPT A4 feasibility check (`check_decode_direction_feasibility`) cannot run until this is resolved. Options for the user listed in `TASK2b_lognorm_input_check.md` §scGPT runtime.
- Task 4 remains on hold.

## 2026-09-30 — A2b re-anchoring and scGPT state-dict blocker

### A2b: σ, α_S re-anchored on the log1p PCA-30 basis

Script `experiments/exp1_statespace/A2b_reanchored_sigma_alpha.py`.
Output `A2b_reanchored_sigma_alpha.json`.

- **σ_log1p = 0.0820** (vs preprint K562 σ = 0.240 on the residual basis), via within-guide split-half bootstrap on the log1p PCA-30 basis, median over 188 targets at ≥ 20 cells. The log1p basis has lower per-entry variance than the residual basis.
- **α_S_log1p = 214.99** (vs preprint K562 α_S = 369 on the residual basis), by matching median column norm of `S_true` to the observed median column norm of `S` on the log1p basis. Observed median `‖S_col‖` = 4.32; baseline (α_S = 1) median = 0.0201.
- **Real target-held-out ρ = 0.9208 ± 0.0451** (unchanged from A2 — same recipe, same data).
- **Matched linear-truth ρ with re-anchored σ, α_S = 0.0773 ± 0.0019** (vs 0.1320 with A2's mis-anchored σ = 0.240, α_S = 369; vs 0.1818 on the A1 residual basis).

Reference table (ignoring the scGPT→30 arm, which is still blocked):

| Setting | real ρ | matched-linear ρ | gap | σ | α_S |
|---|---:|---:|---:|---:|---:|
| A1 — residual-space PCA-30 (paper 1) | 0.9658 ± 0.0241 | 0.1818 | 0.78 | 0.240 | 369 |
| A2 — log1p PCA-30, wrong anchors | 0.9208 ± 0.0451 | 0.1320 ± 0.0036 | 0.79 | 0.240 | 369 |
| **A2b — log1p PCA-30, re-anchored** | **0.9208 ± 0.0451** | **0.0773 ± 0.0019** | **0.84** | **0.0820** | **214.99** |

The paper-1 operator-level failure story not only reproduces on the log1p basis — the re-anchored matched-linear-truth drops to ρ = 0.08, giving a *larger* gap than A1 (0.84 vs 0.78). The log1p basis is better-conditioned for the matched-SNR control. A2b supersedes A2's mis-anchored matched-linear value as the apples-to-apples reference for the Task 4 PCA-lognorm row. **A2b becomes the PCA-30 reference row for Task 4.**

### scGPT runtime — torchtext shim installed; state-dict format mismatch pending

The `torchtext` ABI blocker is resolved by `src/anchorop/state_space/_torchtext_shim.py`, which installs a plain-`dict`-backed `_DictVocab` and `vocab` function in `sys.modules["torchtext"]` before `import scgpt`. `ScGPTRep` now imports the shim on module load. Verified: `import scgpt` succeeds in `anchor-op-scgpt` on CPU macOS.

Loading `scGPT_human/best_model.pt` into a `TransformerModel` instantiated per `args.json` reveals a **state-dict layout mismatch**:

- 34 missing keys of the form `transformer_encoder.layers.N.self_attn.in_proj_{weight,bias}` (the standard PyTorch `nn.MultiheadAttention` fused projection).
- 25 unexpected keys of the form `transformer_encoder.layers.N.self_attn.Wqkv.{weight,bias}` + `flag_encoder.weight` (the `flash_attn` fused-attention layout).

The checkpoint was trained with `use_fast_transformer: true` (per `args.json`), which uses `flash_attn`. `flash_attn` is a CUDA extension and does not install on macOS CPU, so the standard-torch forward path is the only option here, which uses the `in_proj_*` layout.

A state-dict conversion (`Wqkv.weight` → `in_proj_weight`, `Wqkv.bias` → `in_proj_bias`, with proper Q/K/V slicing on dim 0; and `flag_encoder.weight` either folded into another embedding or dropped) is **tractable but non-trivial** (~ 50–100 LOC, needs testing). Rather than ship an unverified conversion, I stop here per the "stop and tell me" rule on install failures.

Options for the user:

- (D) Write the state-dict conversion in this branch, test on 5 cells, commit. Open to attempting if authorised.
- (E) Run the scGPT A4 feasibility check on the user's cluster GPU where `flash_attn` is available (would also be the natural venue for Task 4 fitting).
- (F) Point me at a published scGPT inference script that already handles this conversion.

**Task 4 remains on hold.** The scGPT A4 feasibility check remains on hold pending resolution of (D/E/F). The PCA-lognorm and FA-lognorm arms of Task 4 can run on CPU here under the A2b-anchored recipe once the user authorises.

## 2026-09-30 — scGPT state-dict conversion + A4 feasibility: HALT

### State-dict conversion (option D)

`src/anchorop/state_space/scgpt.py::_load_scgpt_encoder` now instantiates `scgpt.model.TransformerModel` with `use_fast_transformer=False` and converts the flash_attn checkpoint on the fly:

- rename `.self_attn.Wqkv.weight` → `.self_attn.in_proj_weight` (shape (1536, 512) in both)
- rename `.self_attn.Wqkv.bias` → `.self_attn.in_proj_bias` (shape (1536,))
- drop `flag_encoder.weight` (flash_attn packed-sequence aux; unused on standard torch path)

After conversion: `load_state_dict(..., strict=False)` reports **10 missing, 0 unexpected**. The 10 missing keys are all `cls_decoder._decoder.*` (the classifier head not used in the frozen-pretrain + CLS=False configuration). Attention, embeddings, feedforward, norms, MVC decoder, flag-encoder-removed path all load cleanly.

`_scgpt_forward` / `_scgpt_continuous_forward` tokenize each cell as the top-`max_seq_len=1200` genes by |expression|, map gene symbols to vocab indices via the fitted vocab, pass through `model(src, values, src_key_padding_mask, CLS=False, CCE=False, MVC=False, ECS=False)`, and return `output["cell_emb"]`.

### A4 feasibility result

Script: `experiments/exp1_statespace/A4_scgpt_feasibility.py`.
Output: `A4_scgpt_feasibility.json`.

Setup: 200 K562 NT control cells, log1p-normalised, fitted into `ScGPTRep(d_out=30)`. PCA head retains 97.8 % of the scGPT native 512-dim variance in the top-30 components. Target gene: `ENSG00000004897` (CDC27, the first measurement-bundle target). `n_subset = 50` (PREREG default). κ ∈ {0.5, 0.7, 0.9}. Cosine threshold 0.9.

Timings: fit = 204 s (200 cells on CPU), feasibility = 530 s (12 encode calls on 50-cell batches). Total ~ 12 min.

Verdict: **HALT**.

| Clause | Value |
|---|---|
| (i) non-trivial | **PASS** (‖u_z‖ = 0.0261 > encoder noise ‖·‖ = 0.0) |
| (ii) stability — within subset A, across κ | PASS: 0.5↔0.7 = **0.986**, 0.5↔0.9 = **0.986**, 0.7↔0.9 = **1.000** |
| (ii) stability — across two disjoint 50-cell subsets, same κ | **FAIL**: κ=0.5 = **−0.740**, κ=0.7 = **−0.760**, κ=0.9 = **−0.767** |
| halt_reason | min cosine = −0.767 < 0.9 |

**The subset cosines are antiparallel (~ −0.77), not random (~ 0).** Within one random 50-cell subset of controls, scaling CDC27's log1p expression by (1 − κ) for κ ∈ {0.5, 0.7, 0.9} produces an embedding shift whose direction is essentially unity-cosine stable across κ (the perturbation is magnitude-invariant). But between two disjoint 50-cell subsets of the same control population, the embedding shift points in **opposite** directions.

This is not within the (ii) stability band under A4, so the scGPT→30 arm is halted per the preregistered rule.

**Interpretation (for user review, no analysis action taken)**: the antiparallel direction across subsets suggests the per-cell gene-ranking tokenization that scGPT uses (top-`max_seq_len=1200` genes by expression) interacts with the knockdown-scale perturbation differently in different random subsets of controls — not simply as "noise on an underlying signal" but as two distinct geometries with opposite polarity. Possible causes to explore in a follow-up (NOT run here):

- Fit the PCA head on more controls and check whether PCA-axis sign ambiguity is the source (per-component sign is arbitrary; a different fit could flip signs).
- Use a larger `n_subset` (say 200 or 400) to see if the antiparallelism is a small-sample artefact.
- Try a less highly-expressed target (CDC27 is a cell-cycle regulator, highly expressed in K562).
- Test on an alternative base cell embedding (e.g., mean-pool of transformer output rather than scGPT's internal `avg-pool`).

Per A4 as revised in PREREG amendment 2: **no workaround**. The scGPT→30 arm cannot participate in Task 4 with its knockdown-scale perturbation input definition on this checkpoint and this cell set.

Task 4 remains on hold. PCA-lognorm and FA-lognorm arms are the two linear arms still eligible under A2b's re-anchored σ and α_S; the user decides whether to proceed with only those arms, re-open scGPT with a workaround that steps outside A4, or close out the experiment with the halt as a reported result.

## Leakage check — pending

Grep the scGPT pretraining-corpus manifest (CellxGene + the scGPT README) for:

- "Replogle" (2022, 2023)
- "K562" + "Perturb-seq"
- GEO accession GSE264667 (Replogle 2022 Cell)
- Figshare Plus deposit 20029387 (Replogle 2022 K562 essential h5ad we use)
- "perturb-seq" / "perturbation" datasets listed in the scGPT pretraining description

Source URLs and the verdict will be committed here before any encoder fitting runs. Checkpoint lineage (whether it was perturbation-fine-tuned) is confirmed from the scGPT release notes for the specific SHA-256 downloaded.
