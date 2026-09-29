# Cover letter — NAR Genomics and Bioinformatics (Standard Paper)

*Draft: 2026-09-28*

Dear Editors,

We submit *Additive-input encoding fails target-held-out prediction on current Perturb-seq screens* for consideration as a Standard Paper at NAR Genomics and Bioinformatics.

**Central finding.** On three current Perturb-seq screens (Replogle K562 essential, Replogle RPE1 essential, Jost 2020) the operator-level target-held-out prediction of the additive-input linear encoding class fails on all three screens under nested cross-validation, while a matched-signal-to-noise linear-truth positive control on the identical inputs, noise, and estimator succeeds. On the forward prediction task, calibrated by the required predict-training-mean baseline and evaluated under a preregistered comparator panel of three linear encodings — a fixed program-space encoding `u_g = −κ_g Wᵀδ_g`, a control-covariance footprint encoding, and a training-fold learned linear encoding in the style of the recent Perturb-seq-prediction benchmark [Ahlmann-Eltze et al. 2025] — the per-screen picture is mixed: on K562 essential all three encodings beat the training-mean baseline by ~ 2 outer-fold SDs, while on RPE1 essential and Jost 2020 none of the three beats the baseline. The K562 forward win is real and preregistered but modest: the fitted encodings close only ~ 7 % of the training-mean-to-linear-truth gap that a well-posed linear operator would deliver at matched SNR. The full outcome is preregistered, calibrated, and reported per screen; the title's narrowing to *operator-level* isolates the failure claim to the inverse-direction reading in §2.1–§2.5.

**Why it matters for Perturb-seq network inference.** Pooled Perturb-seq analyses are increasingly used to learn regulatory operators from single-cell responses to targeted perturbations. The most direct linear formulation projects the target-gene identity through a control-derived program basis and inverts the resulting design; it is closed-form and reads as a network. Our result shows that on the operator-level reading — the inverse task, whose fitted `A ≈ −U·S⁺` is what would be read out as a regulatory network — the additive-input linear encoding class fails on all three tested screens against a matched-SNR linear-truth positive control, regardless of whether the encoding is fixed by design (one-hot or footprint) or learned within training folds only. Downstream operator-level inferences from these screens should be read against that calibration. On the forward *prediction* task, the K562 result shows that under a proper training-mean baseline the same linear class can beat noise, but only by a small fraction of what a well-posed linear operator would deliver — a distinction worth surfacing before over-reading the K562 forward-task success as an operator-level result.

**Three reusable contributions.**

1. **Target-grouped nested cross-validation on `J·P_X = −U·S⁺`.** A fold-honest evaluation with a rank-0 baseline in the inner selection grid, and a required predict-training-mean baseline on the forward task. About half of each screen's response energy is a shared program-space mode, so beating predict-zero is trivial; the training-mean baseline is what makes any encoding claim calibrated.

2. **Matched-SNR generative controls.** A linear ground truth simulated at each screen's own `U`, `κ`, and per-entry noise, evaluated under the same nested-CV recipe. This makes a real held-out ρ interpretable: if the matched control cannot beat the same baseline on this design, the estimator is the bottleneck, not the model. In this paper the matched control reaches ρ = 0.18 (K562), 0.53 (RPE1), 0.72 (Jost) while the real data sits at the baseline. The controls run across five ensembles (dense, sparse-10 / 2%, rank-5, block-modular) and confirm the result is not specific to the dense-ground-truth family.

3. **An identifiability and efficiency audit.** The identifiability gate for the linear operator now requires both `rank(U) = d` and retained `rank(S) = d` at the preregistered tolerance, fixing a silent numerical-rank inflation on collinear guide libraries. The knockdown-efficiency router routes count data to `mean_ratio` and pre-scaled residual data to a `detection_rate` proxy score explicitly labelled as a signed distributional shift rather than a fractional-knockdown estimate.

**Open-source and reproducibility.** anchor-op is available at <https://github.com/manarai/anchor-op> under the MIT license. A pinned conda `environment.yml` (Python 3.11), a 59-test pytest suite, per-figure regeneration scripts under `reproduction/`, and every JSON that a numeric claim in the paper traces to are shipped with the release. Every number in the Results section that is not derived in-text has an entry in `results/recheck/number_provenance.csv` giving the source JSON, the reproduction script that produced it, and the rounding rule where applicable. All preregistrations and amendments are included, and analyses are labelled preregistered vs post hoc.

- **Zenodo DOI (release archive):** *placeholder — will be inserted upon v0.3.1 tag archival on Zenodo.*
- **bioRxiv DOI (preprint):** *placeholder — will be inserted upon repost of the v0.3.1 revision on bioRxiv.*

We do not claim methodological superiority over any other Perturb-seq analysis. We report a negative held-out-prediction result on one encoding class under a specific evaluation recipe, calibrated by matched-SNR positive controls. The result is a diagnostic result about the state of the additive-input linear class on these three screens.

We suggest reviewers with expertise in single-cell perturbation analysis, network inference from Perturb-seq, and cross-validation methodology in machine learning of biological data.

The manuscript has not been submitted elsewhere. All authors have approved the submission and declare no competing interests. We thank the editors for their consideration.

Sincerely,

Dalton Kutzen, Kyller Fullmer, Tommy W. Terooatea (corresponding author)
Brigham Young University, Provo, UT, USA

*Corresponding author: [Tommy W. Terooatea — affiliation email TBD]*
