# exp1b-statespace-benchmark — EXPERIMENT_LOG.md

Branch: `experiments/exp1b-statespace-benchmark` (forked from `experiments/exp1-statespace`).
Date opened: 2026-09-30.
**Not merged to `main`. Nothing from this experiment enters the anchor-op journal manuscript (`v0.3.4` on main).**

## Scope

Benchmark six state-space arms on target-held-out operator prediction (both inverse and forward), on all three Perturb-seq screens (K562, RPE1, Jost), using the same filters/folds/seeds as paper 1. Primary question: does any arm improve on the PCA(d=30) and FA(d=30, QR) reference arms? Secondary: does this benchmark's embedding ranking correlate with scJDO's embedding ranking.

Prerequisite (satisfied 2026-09-30, v0.3.4 on main):
- PCA re-run through the same code path as FA (`reproduction/55_fa_table1_recheck.py`).
- FA loadings QR-orthonormalized.
- Fold SDs reported for all nested-CV ρ cells of Supp Table S3.

## 2026-09-30 — B0: scJDO embedding-panel extract

Scope: pull the embedding-comparison panel from the pinned scJDO dependency, with file/line refs, so we can compute the secondary Spearman rank correlation between this benchmark's ranking and scJDO's. No assumptions; everything below is read from the repo.

### Pinned dependency

- Repo on disk: `/Users/terooatt/Documents/scJDO`
- scJDO 0.3.0, editable install into `scJDO_simulation/.venv`
- Pinned commit: `b022add4` (recorded at `~/Documents/scJDO_simulation/CLAUDE.md:12`)
- Public submodules: `pp`, `tl`, `pl`, `io`, `models`, `transport`, `simulate`, `archetypes`
- Note: `scjdo.operator` does not exist (`~/Documents/scJDO_simulation/CLAUDE.md:14`, repeated line 23)

### Panel source

- Notebook: `~/Documents/scJDO/examples/08_embedding_benchmark.ipynb`
- Companion PDF: `Final_check/02_figures/panels_and_diagnostics/fig2c_embedding_benchmark_8method.pdf`
- Also referenced by `analysis/logs/fig2c_embedding_benchmark.log`
- Output table lives at `results/benchmark/benchmark_scores.csv` (per the ipynb's cell at line 1033).

### Panel composition (notebook line numbers)

Eight methods, run order declared at lines 573–585 (`EMBED_FNS` registry):

| # | Method | Call site | Loadings | Seeds | Determ.? |
|---|---|---|---|---:|:-:|
| 1 | PCA (sklearn) | line 477 | `varm["PCs"]` | 1 | yes |
| 2 | FactorAnalysis (sklearn) | line 484–486 | `components_.T` (NOT orthonormalized) | 2 | no |
| 3 | ICA = FastICA (sklearn) | line 492–494 | `mixing_` | 2 | no |
| 4 | TruncatedSVD (sklearn) | line 500–502 | `components_.T` | 2 | yes (per sklearn's `random_state`) |
| 5 | DiffMap | line 509–514 | ridge-regression on W (skip DC0) | 1 | yes |
| 6 | LDVAE = `scvi.model.LinearSCVI(_ad, n_latent=N_LATENT)` | line 527 | `model.get_loadings()` | 2 | no |
| 7 | scVI = `scvi.model.SCVI(_ad, n_latent=N_LATENT)` | line 550 | ridge-regression on W | 2 | no |
| 8 | PLS = `PLSRegression(n_components=N_LATENT)` | line 568–571 | `model.x_loadings_` (supervised on branch labels) | 1 | yes |

- `N_LATENT = 20` for all methods (line 78). **This differs from Part B's `d = 30`.**
- `N_EPOCHS = 500` for stochastic methods; `N_SEEDS = 2`.

### Dataset

- Palantir bone-marrow hematopoiesis sample (`marrow_sample_scseq_counts.h5ad`), ipynb line 11.
- Developmental trajectory with 3 lineage branches, Palantir pseudotime + branch masks shared across all 8 methods (line 378).
- HVG = 1500 genes selected once; all methods see the same HVG matrix (lines 274, 305–307).
- Full-dataset PCA (50 PCs) and DiffMaps computed for Palantir only (line 316).

### Scoring

Five 0–1 metrics (higher is better), each a developmental-trajectory diagnostic:
- `marker` — enrichment of lineage-marker genes in top-loaded features.
- `specificity` — branch-level specificity of latent coordinates.
- `stability` — seed-to-seed variance (only informative for stochastic methods).
- `timing` — temporal separation between bifurcations in pseudotime.
- `r2` — linear reconstruction R² of the HVG matrix from the latent.

Composite = 0.30·marker + 0.20·specificity + 0.20·stability + 0.15·timing + 0.15·r2 (ipynb line 247).

### Final ranking (notebook output at lines 1033–1041)

```
Rank  Method          Composite  Marker  Specificity  Stability  Timing   r2
1     PLS             0.670      0.246   0.991        1.000      0.317   0.999
2     FactorAnalysis  0.656      0.198   0.963        0.768      0.671   0.999
3     scVI            0.597      0.048   0.878        0.969      0.426   0.995
4     LDVAE           0.586      0.000   0.991        0.975      0.285   0.997
5     TruncatedSVD    0.582      0.095   0.836        1.000      0.245   0.995
6     ICA             0.580      0.048   0.809        0.911      0.480   0.999
7     PCA             0.553      0.048   0.767        1.000      0.245   0.993
8     DiffMap         0.397      0.000   0.000        1.000      0.313   1.000
```

### Overlap with exp1b's six arms

- exp1b arms (planned, d = 30):
  - PCA (log1p)
  - FA (log1p, QR-orthonormalized)
  - LDVAE encoder (posterior-mean latent)
  - LDVAE loadings (decoder loadings, QR-orthonormalized, used as a linear basis)
  - scVI (posterior-mean latent, no sampling)
  - scGPT_human with fixed deterministic ingestion
  - scGPT with random weights, same ingestion (pretraining control)
- scJDO panel: PCA, FA, ICA, TruncatedSVD, DiffMap, LDVAE, scVI, PLS.
- **Overlap = PCA, FA, LDVAE, scVI (4 methods).**
- scGPT (both variants) is exp1b-only. PLS, ICA, TruncatedSVD, DiffMap are scJDO-only.

### Caveats carried into the PREREG

1. **Different task.** scJDO scores developmental-trajectory structure. exp1b scores target-held-out operator prediction. The Spearman is a cross-task transfer question, not an identity test.
2. **Different d.** scJDO N_LATENT = 20; exp1b d = 30.
3. **Different dataset.** scJDO: bone-marrow hematopoiesis. exp1b: Perturb-seq K562 + RPE1 + Jost.
4. **Different FA convention.** scJDO's FA uses raw `components_.T` (NOT orthonormalized). exp1b's FA uses QR-orthonormalized loadings so the FA/PCA scale matches (same correction that landed in anchor-op v0.3.4 on main). The ranks of the two FA runs are still comparable but their numerical values are not.
5. **LDVAE split.** scJDO treats LDVAE as a single arm (encoder-based scoring). exp1b splits LDVAE into 3a (encoder posterior mean) and 3b (decoder loadings as a linear basis). For the Spearman mapping, LDVAE in scJDO's panel ↔ LDVAE-encoder in exp1b.
6. **Overlap n = 4** is small for a Spearman; the correlation will be reported with a confidence interval and treated as suggestive, not inferential.
