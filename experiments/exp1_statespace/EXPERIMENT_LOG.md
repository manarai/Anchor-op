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

## Leakage check — pending

Grep the scGPT pretraining-corpus manifest (CellxGene + the scGPT README) for:

- "Replogle" (2022, 2023)
- "K562" + "Perturb-seq"
- GEO accession GSE264667 (Replogle 2022 Cell)
- Figshare Plus deposit 20029387 (Replogle 2022 K562 essential h5ad we use)
- "perturb-seq" / "perturbation" datasets listed in the scGPT pretraining description

Source URLs and the verdict will be committed here before any encoder fitting runs. Checkpoint lineage (whether it was perturbation-fine-tuned) is confirmed from the scGPT release notes for the specific SHA-256 downloaded.
