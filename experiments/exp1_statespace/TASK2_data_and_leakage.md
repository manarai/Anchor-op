# Task 2 — Data and leakage report

Branch: `experiments/exp1-statespace`. Written 2026-09-30, before the Task 3 PREREG commit.

## Data

**Dataset**: Replogle et al. 2022, *Cell* 185:2559 — K562 essential-gene Perturb-seq. Source: Figshare Plus deposit 20029387 (`https://plus.figshare.com/articles/dataset/_strong_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_strong_/20029387`). No GEO fetch; we use the same `examples/data/K562_essential_normalized_singlecell_01.h5ad` the preprint uses (10 GB; 310,385 cells × 8,563 genes).

**Pipeline**: anchor-op paper 1's pipeline from `reproduction/03_fig3_k562_essential.py`, with the preregistered **≥ 60 cells per target** filter (NOT ≥ 100 cells; corrected per user instruction).

**Measurement bundle used by the A1 gate**:

- `results/k562_essential_measurement.pkl`
- `S` ∈ ℝ^{30 × 188}, `U` ∈ ℝ^{30 × 188}
- 188 retained guides, 188 targets (one guide per target after Replogle's released aggregation)
- Program basis fit on non-targeting controls at `d = 30`
- Noise anchor σ = 0.240 (from `F1_K562_essential.json`)
- Matched α̂_S = 369.0 (from `F1_F4_summary.json`)

The ≥ 60 cells filter will be applied again when the state-space arms rebuild per-knockdown shifts from control cells — the h5ad has 1,581 qualifying essential-gene targets at ≥ 60 cells (used in paper 1 §2.3), and the 188 of those that pass the paper's full retention pipeline are the baseline for the A1 gate and the comparator.

## A1 reproduction gate (preregistered)

Script: `experiments/exp1_statespace/A1_reproduction_gate.py`
Output: `experiments/exp1_statespace/A1_reproduction_gate.json`

Target-grouped 5-outer × 3-inner nested CV, rank grid `{0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}`, seed 20260930, 15 matched-linear-truth replicates at σ = 0.240 and α_S = 369.0. These match MANUSCRIPT.md §4.5 of the preprint exactly.

**Result**: **PASS**.

- Real K562 target-held-out ρ = **0.9658 ± 0.0241** (reference 0.96; tol ±0.03 ✓)
- Matched linear-truth ρ = **0.1818** (reference 0.18; tol ±0.03 ✓)
- Picked ranks per outer fold: `[3, 3, 3, 5, 3]`

The branch's code reproduces Table 1 of the preprint for the PCA-30 arm. Encoder arms (FA-30, scGPT→30) proceed on this foundation.

## Pretraining leakage audit — scGPT_human

**Checkpoint provenance** (to record once downloaded):

- Source URL: `https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing` (the "whole-human" / "scGPT_human" model, from the scGPT GitHub Model Zoo README).
- Download date: *to be filled when the file is on disk*.
- SHA-256 of the downloaded weights file: *to be filled; stored in `EXPERIMENT_LOG.md` alongside* `ScGPTCheckpoint.sha256`.
- Local path: `experiments/exp1_statespace/weights/scGPT_human/` (gitignored).

**Public pretraining-corpus information available (2026-09-30)**:

- scGPT GitHub README (Model Zoo table) states: *"Pretrained on 33 million normal human cells"* for the whole-human checkpoint (no further study-level enumeration on this page).
- scGPT ReadTheDocs introduction and API reference pages: no study-level corpus list.
- Cui et al. 2024, *Nature Methods* 22(8):1657 (DOI 10.1038/s41592-025-02772-6 — the published scGPT paper): paywalled, Methods not retrievable via public WebFetch. The exact per-study pretraining-corpus list is in the paper's Methods and in the CELLxGENE Discover census snapshot referenced there.

**Leakage search — specific accessions queried**:

| Accession / name | Source searched | Result |
|---|---|---|
| GEO GSE264667 (Replogle 2022 K562 essential, canonical GEO) | scGPT GitHub README, scGPT docs introduction, Nature Methods landing page (paywall) | not found in any public page I could fetch; status *unresolved* |
| Figshare Plus deposit 20029387 (the h5ad we use) | scGPT GitHub README, scGPT docs | not found; status *unresolved* |
| "Replogle" (any year) | scGPT GitHub README, scGPT docs | not found; status *unresolved* |
| "Perturb-seq" or "K562 Perturb-seq" | scGPT GitHub README, scGPT docs | the Model Zoo mentions a *"continual pretrained"* variant *"for zero-shot cell embedding related tasks"* but does not call out Perturb-seq in the whole-human pretrain |

**Verdict (preliminary, to be strengthened before Task 4)**: the exact pretraining-corpus listing for `scGPT_human` is not publicly enumerated in the pages I could fetch. The canonical sources are (a) Cui et al. 2024 Methods and (b) the CELLxGENE Discover census snapshot. The Task 4 scGPT arm is **halted** until one of the following is on record:

- Methods text from Cui et al. 2024 explicitly confirming whether Replogle 2022 / GSE264667 / Figshare 20029387 are in the pretraining corpus (via institutional access to the paper), **OR**
- A direct query of the CELLxGENE Discover census API for `collection.name = "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq"` or dataset ID matching Replogle 2022 K562 essential, showing presence or absence.

If leakage is confirmed, the scGPT arm is reported as **"possibly leaked"** throughout and a second encoder not trained on Replogle 2022 is added as a non-leaky comparator (per the original Task 2 instruction).

**Checkpoint lineage** (whole-human vs. perturbation-fine-tuned) will be verified from the Model Zoo README text next to the download link and from the file name / version string once the weights are on disk. The scGPT README's Model Zoo lists multiple variants (`whole-human`, `continual pretrained`, `heart`, `kidney`, etc.); the one we use is explicitly the whole-human / foundational pretrain. If the file I download contains "pert" in the name, I halt and switch to the base variant.
