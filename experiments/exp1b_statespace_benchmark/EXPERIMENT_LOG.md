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

## 2026-09-30 — B1: six-arm classes committed (code only, no fitting)

`experiments/exp1b_statespace_benchmark/arms.py` (commit `510771b`). Seven classes:
- PCALog1pArm, FAQRLog1pArm — linear bases on log1p control counts; QR orthonormalization.
- LDVAEEncoderArm — scvi.model.LinearSCVI(n_latent=30), posterior-mean latent, raw counts.
- LDVAELoadingsArm — reuses the trained LDVAE's decoder loadings, QR-orthonormalized, used as a linear basis on log1p; `.fit(ldvae_arm)` + `.set_mean(X_log1p_ctrl)`.
- SCVIArm — scvi.model.SCVI(n_latent=30), posterior-mean latent, raw counts.
- ScGPTFixedIngestionArm — scGPT_human (or random-weights twin), with the exp1 A4 diagnosis fixes wired: np.argsort(kind='stable') + target force-included at token position 0 for both ctrl and kd. PCA head on controls.

Arm ABC carries `input_space` ∈ {log1p, count}; `knockdown_u_z` applies the right scaling.

## 2026-09-30 — B2: feasibility gate harness committed

`experiments/exp1b_statespace_benchmark/feasibility.py` (commit `29879cc`). Five clauses:
- (a) determinism — identical input twice gives max |Δ| < 1e-6.
- (b) negative control — a gene with zero counts in all controls gives ‖u_z‖ < 1e-6.
- (c) non-triviality — ‖u_z‖ at κ=0.7 above the cell-sampling noise floor across 20 random disjoint 50-cell subset pairs.
- (d) stability — mean subset cos > 0.9; cross-seed cos reported for VAE arms.
- (e) dose-grading — ‖u_z‖ increases with κ ∈ {0.5, 0.7, 0.9}.
Targets: q10, q30, q50, q70, q90 (control log1p-mean quantiles) + CDC27.

## 2026-09-30 — B3: PREREG_exp1b.md committed before any fit

`experiments/exp1b_statespace_benchmark/PREREG_exp1b.md` (commit `29879cc`). Reference arm = FA(QR) (scJDO default; Part A corrections showed PCA ≡ FA(QR) within fold SDs on all three screens). Pass criterion per §7; ladder comparisons per §8; secondary Spearman per §9; leakage per §11; expected outcome per §12; stop rule per §13.

## 2026-09-30 — B4 compute estimate (reporting + STOP before full benchmark)

Local machine capabilities:
- scvi-tools 1.5.0.post1, torch 2.10.0, Apple MPS available (CUDA not available).

### Feasibility gate (B4 phase 1, local)

Per (arm × screen):
- **Linear arms** (PCA, FA(QR), LDVAE-loadings): fit ≤ 30 s; gate ≤ 1 min. ~5 min each.
- **VAE arms** (LDVAE-encoder, scVI): train 100 epochs on NT controls (feasibility; full benchmark uses 400) → ~5–15 min per screen per training on MPS. Gate adds ~2 min. ~15 min per arm per screen; ×3 training seeds for the (d) cross-seed clause.
- **scGPT arms** (fixed-ingestion, random-weights): checkpoint load ~5 min; PCA-head fit on controls ~5 min; gate = 6 targets × 3 κ × (2 encodes for the primary u_z + 2 × 20 encodes for the between-subset pairs) ≈ 240 encodes at 50 cells each ≈ **4 h per scGPT arm per screen on local CPU** (MPS not implemented in `ScGPTRep`). Mitigations possible: `max_seq_len = 512` (half the budget), 10 subset pairs instead of 20, 50-cell subsets reduced to 25. Each would cut scGPT feasibility to ~1 h per arm per screen.

**Feasibility total (as spec'd, local):**
- Linear × 7 (3 PCA/FA/LDVAE-load × 3 screens): ~0.5 h
- VAE × 2 arms × 3 screens × 3 seeds: ~5 h on MPS
- scGPT × 2 arms × 3 screens: **~24 h on local CPU** (dominant cost)
- Grand total: **~30 h local**.

### Full benchmark (B4 phase 2, GATED on user confirmation)

Per (arm × screen × seed):
- Linear arms: nested CV + matched-linear sims < 5 min each.
- VAE arms: 400-epoch training (3× feasibility epochs) + nested CV. ~30 min per training on MPS, ~2 min nested CV. × 3 seeds × 3 screens × 2 VAE arms = **~9 h on MPS**.
- scGPT: nested CV + matched-linear sims on frozen embeddings. The expensive part is building the embeddings once per screen (already cached from feasibility), after which per-fold is cheap. ~2 h extra per screen per scGPT arm after feasibility caches.

**Full benchmark total (local):**
- Linear nested CV: ~0.5 h
- VAE: ~9 h on MPS
- scGPT: ~12 h additional on local CPU after feasibility caches
- Grand total: **~22 h on local**, dominated by scGPT.

**On a single CUDA GPU (A100-class):**
- VAE: ~1 h total (parallelisable across seeds; scvi-tools has mixed MPS/CUDA perf but CUDA is the mature path)
- scGPT: ~2 h total (batched embedding + nested CV)
- Full benchmark: **~3 h on cluster GPU**.

### STOP (per PREREG § 14)

Awaiting user confirmation on compute venue. Three natural options:

1. **Local only, as-spec'd.** ~30 h feasibility + ~22 h full benchmark = ~2 days. Feasible but blocks the laptop.
2. **Local with scGPT reductions.** Use `max_seq_len=512` + 10 subset pairs for the gate; keep κ grid and target quantiles. Cuts feasibility to ~6 h, full benchmark unchanged (~22 h) unless the same reduction applies there (it would make the forward-task comparisons weaker). Report both.
3. **Cluster CUDA GPU for everything.** Est. 3–4 h total. Requires scp'ing the h5ads + scGPT checkpoint to the cluster node (K562 10 GB + RPE1 8 GB + Jost 500 MB + scGPT 2 GB).

Not launching anything until the user picks.

`experiments/exp1b_statespace_benchmark/run_feasibility.py` is in place (commit TBD) with `load_screen(...)` left as a stub; populate before launch (mirror `reproduction/55_fa_table1_recheck.py`'s HVG + force-included recipe) and set `ARMS`/`SCREENS` to the user-approved subset.
