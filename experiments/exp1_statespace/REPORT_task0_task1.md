# Report — Experiment 1 Tasks 0 and 1

Branch: `experiments/exp1-statespace`, forked from `main @ v0.3.2` (`37e12bb`).

## Task 0 — StateSpace abstraction and three linear implementations

Delivered as `src/anchorop/state_space/`:

| Implementation | Status | Jacobian | Notes |
|---|---|---|---|
| `PCARep(d)` | **done** | constant, `components_` (dim × G) | sklearn `PCA` fit on controls; encode matches `sklearn.PCA.transform` up to column sign; A3 primary non-encoder arm |
| `FARep(d)` | **done** | constant, `pinv(components_).T` (dim × G) | sklearn `FactorAnalysis` fit on controls |
| `MultiomeRep(d)` | **stub** | — | every call raises `NotImplementedError` until a confirmed multiome Perturb-seq dataset |
| `ScGPTRep(d_out)` | **scaffolded** | autograd via continuous-input path (A4) | lazy load of `scgpt` + checkpoint; `ScGPTCheckpoint` dataclass records name / path / SHA-256 / native_dim / continuous_input_supported; three concrete shims await the checkpoint download |

Shape conventions enforced across all arms:

- `encode(X)` returns `(n_cells, dim)`.
- `jacobian(X)` returns `(n_cells, dim, n_genes)`; for linear encoders the per-cell axis is a broadcast of the constant matrix.
- `decode_direction(u_gene)` returns `(dim,)` and must equal `jacobian[0] @ u_gene` for linear encoders — enforced by test.

## Task 1 — ScGPT Jacobian feasibility check (A4), scaffolded

The amendment A4 of the preregistration specifies that scGPT's Jacobian feasibility must be the FIRST scGPT check, before any encoder fit. The method `ScGPTRep.check_jacobian_feasibility(X_control, n_cells=5, step_sizes=(1e-3, 1e-2, 1e-1), tol_cosine=0.9)` returns a report dict with keys `continuous_input_supported`, `per_cell_cosine`, `stable`, `halt_reason`, and `checkpoint`. The halt rule is strict: any (cell × step) cosine below `tol_cosine` sets `halt_reason` and the arm is halted (no workaround per A4). If the checkpoint has no continuous-input path, the method returns a halt record immediately — autograd through binning is not a valid Jacobian.

Status: the scaffold is complete and tested in placeholder form. The three concrete scGPT shims (`_load_scgpt_encoder`, `_scgpt_forward`, `_scgpt_continuous_forward`) raise `NotImplementedError` and must be wired in once the `scGPT_human` checkpoint is downloaded.

## Test results

```
$ pytest -q
.......................................................................s [100%]
SKIPPED [1] tests/test_state_space.py:133: scGPT checkpoint not present. Set
$SCGPT_CKPT to the path of a pretrained scGPT_human checkpoint to run the A4
Jacobian feasibility test. Fine on CPU; 5 cells × 3 step sizes.
71 passed, 1 skipped in 1.66s
```

- 59 pre-existing anchor-op tests continue to pass.
- 12 new StateSpace tests pass.
- 1 scGPT A4 feasibility test skipped, as designed, until the checkpoint is on disk.

## Blockers before Task 2

- scGPT installation + `scGPT_human` checkpoint download. Record checkpoint `name`, SHA-256, and `continuous_input_supported` in `EXPERIMENT_LOG.md`. Confirm the checkpoint was NEVER fine-tuned on perturbation data.
- Replogle K562 essential h5ad path: use the existing `examples/data/K562_essential_normalized_singlecell_01.h5ad` (Figshare 20029387), with the anchor-op paper's pipeline and `≥60` cells filter (not `≥100` as originally written; per user amendment).

Task 2 (data + leakage) and the A1 reproduction gate run next on this branch once scGPT is available on disk. Task 3 (PREREG) is committed and pushed *before* any fitting. No Task 4 (fitting) until compute is confirmed.
