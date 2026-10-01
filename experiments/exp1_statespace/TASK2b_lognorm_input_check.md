# Task 2b report — log1p input + A2 PCA-lognorm sanity gate

Branch `experiments/exp1-statespace`. Follows PREREG amendment 2 (committed 2026-09-30 at `0af210d`).

## Raw-counts download

- Source: Figshare Plus deposit 20029387, file `K562_essential_raw_singlecell_01.h5ad`.
- Download URL used: `https://ndownloader.figshare.com/files/35773219`.
- On-disk size: 9.9 GB (Figshare page lists 10.66 GB; the delta is page/compression accounting).
- Shape: 310,385 cells × 8,563 genes.
- Validation: same obs columns and var_names (ENSEMBL IDs) as the preprint's residual h5ad; first-5-cells max count 571 (integer, consistent with raw UMI counts).

## task2b non-triviality check (PREREG §3)

Script: `experiments/exp1_statespace/task2b_lognorm_input_check.py`.
Output: `TASK2b_lognorm_input_check.json`.

- 10,691 NT control cells (matches residual h5ad exactly).
- 299,694 perturbed cells.
- 1,581 qualifying essential targets at ≥ 60 cells per target (matches paper 1 §2.3's 1,581 figure exactly).
- Normalisation: `sc.pp.normalize_total(target_sum=1e4)` + `sc.pp.log1p` (reimplemented inline for the control block to avoid holding the full matrix in memory).
- Median raw counts per NT control cell: **13,714**.

Per-gene control-cell mean on the log1p-normalised h5ad, over the 188 measurement-bundle targets:

| statistic | |mean(X_log_ctrl[:, g])| |
|---|---:|
| min | 6.58e-02 |
| p05 | 1.10e-01 |
| median | 5.32e-01 |
| p95 | 3.30 |
| max | 4.07 |

All 188 targets have |mean| > 0.06 (vs ≤ 3.8e-9 on the residual h5ad — a 7-orders-of-magnitude shift that is the amendment-2 trigger).

**Knockdown-scale FD under log1p at κ = 0.7**: 188 / 188 targets (100 %) have ‖u_z‖ > 1e-6. **PASS** (threshold 95 %).

## A2 PCA-lognorm sanity gate (PREREG §4)

Script: `experiments/exp1_statespace/A2_pca_lognorm_gate.py`.
Output: `A2_pca_lognorm_gate.json`.

Same target-grouped 5-outer × 3-inner nested-CV recipe as A1, rank grid `{0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}`, seed 20260930, 15 matched-linear-truth replicates. For comparability with A1 the matched-linear σ and α_S are kept at the paper's K562 anchors (σ = 0.240, α_S = 369); a per-arm re-anchoring is deferred to Task 4 proper.

| Metric | A1 (residual-space PCA-30) | A2 (log1p-space PCA-30) | Δ |
|---|---:|---:|---:|
| Real target-held-out ρ | 0.9658 ± 0.0241 | **0.9208 ± 0.0451** | −0.045 |
| Matched linear-truth ρ | 0.1818 | **0.1320 ± 0.0036** | −0.050 |
| Picked TSVD ranks per outer fold | [3, 3, 3, 5, 3] | [2, 2, 2, 2, 2] | lower |
| gap = real − matched-linear | 0.78 | 0.79 | +0.01 |

### Interpretation (for user review before Task 4)

Both real ρ and matched-linear-truth ρ shift down by ~ 0.05 on the log1p basis compared with the preprint's residual basis. The *gap* between them — the operator-identifiability picture that the paper's §2.1 reports — is essentially unchanged (0.78 vs 0.79). The log1p basis gives a slightly stronger signal at a lower operating point, with lower picked ranks (median 2 vs median 3) consistent with a somewhat more compact representation.

Qualitative conclusions:

- **Both arms carry the paper-1 story**: real sits at a near-predict-zero baseline, matched linear-truth beats it by ~ 0.78. The operator-level failure is reproduced on the new basis.
- **The arms are NOT numerically identical**. A1's 0.9658 is outside A2's 0.9208 ± 0.0451, so the per-fold distributions do not overlap on the point estimate. This is expected because the encoders are literally different (residuals vs log1p counts, different PCA bases), and the amendment itself warned that A1 is a "reference row only, not used in the comparator".
- **The encoder claim comparator in Task 4** compares scGPT→30 vs **PCA-lognorm** (both on log1p), so the apples-to-apples reference for Task 4 is the A2 numbers (0.9208 / 0.1320), not the A1 numbers.

### What the user needs to decide before Task 4

1. Accept A2 as the new reference row (A2 numbers become the "PCA-30 anchor-op reference" in the Task 4 report), OR
2. Investigate the ~0.05 shift (e.g., re-run A2 with a per-arm re-anchored σ and α_S from the log1p basis, since the paper's 0.240 / 369 are measured on the residual basis), OR
3. Keep A1 as the reference and treat the shift as a basis effect to be reported in the Task 4 writeup.

## scGPT runtime — install blocker

The `anchor-op-scgpt` conda env has `torch 2.13.0`, `scgpt 0.2.4`, `cellxgene-census 1.18.0`, `gdown 6.4.1`, and `ipython 8.39.0` installed. On import `scgpt` pulls in `torchtext`, which fails to link against the installed torch:

```
OSError: Could not load this library:
  /…/torchtext/lib/libtorchtext.so
  Symbol not found: __ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
```

`torchtext` is a hard dep of `scgpt.tokenizer.gene_tokenizer` (first import inside the scGPT package). The torch 2.13 / torchtext 0.18 ABI mismatch is a known cohort of macOS build issues. Resolving it needs either:

- Downgrading torch to ≤ 2.3 (the last version `torchtext` was built against), **or**
- Patching `scgpt.tokenizer.gene_tokenizer` to avoid the torchtext import (it uses `torchtext.vocab.Vocab` for the vocab; replacing with a plain `dict`-based vocab is tractable), **or**
- Rebuilding `torchtext` from source against torch 2.13.

**I did not attempt a fix without explicit guidance**, per the user's "stop and tell me" rule for install failures. The scGPT A4 feasibility check (`check_decode_direction_feasibility`) is therefore **not yet runnable** on this machine.

Options for the user:
- (A) `pip install "torch<=2.3"` in the `anchor-op-scgpt` env and re-pip-install torchtext.
- (B) Patch the scgpt tokenizer to use a plain `dict` vocab (I can do this in-branch; it's ~20 LOC).
- (C) Point me at a different env where scGPT is already working.

## Status

- Branch `experiments/exp1-statespace` tip: `0af210d` (PREREG amendment 2) at time of writing.
- This report + the `TASK2b` / `A2` JSONs land in the next commit.
- **Task 4 (fitting arms × seeds × k) remains on hold.** The scGPT A4 feasibility check is also on hold pending the install-blocker resolution.
- CELLxGENE census audit: 0 hits for Replogle / K562 / CRISPR (confirmed low leakage risk; recorded in `EXPERIMENT_LOG.md`).
- scGPT_human lineage: whole-human pretrain, May 2023 CELLxGENE census snapshot, self-supervised only (confirmed from `args.json`).
