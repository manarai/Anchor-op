"""F7: target-selection sensitivity — rerun F3/F4 on random-200 and count-quantile-200.

The notebooks 01b/01c pick 200 targets by ``value_counts().head(200)`` — i.e.
the targets with the *most* cells. In essential-gene screens, high cell count
correlates with mild fitness defect, biasing the sample toward mild
knockdowns. Reviewer note (2026-09-27): the h5ads carry no explicit fitness
score, so cells-per-target is the practical fitness proxy — and it is the same
quantity the original selection biased on. This script uses cell-count
quantiles for the stratification and treats random-200 as the main
comparison.

Selections:
  1. published:            top-200-by-cell-count (current behavior).
  2. random_200 (MAIN):    random sample of 200 targets with >= 60 cells.
  3. count_quantile_200:   40 targets from each of 5 cell-count quantiles
                            (0–20%, 20–40%, …, 80–100%) among qualifying
                            targets. Explicitly a cell-count stratification,
                            since no explicit fitness score is on the h5ad.

For each selection we report:
  - held-out prediction rho (F3),
  - top-1 singular energy fraction, between-target mean cos, and |cos(s, u)|
    median (F4).

Runtime is dominated by the h5ad load (~10 GB). ~10–25 min per dataset.
Requires the Replogle essential-screen h5ads under examples/data/.
Not automatic in tests; add to reproduction/run_all.sh once the anchor lab has
verified the pipeline on their machine.

Usage:
    python reproduction/35_recheck_F7_selection_sensitivity.py --dataset K562
    python reproduction/35_recheck_F7_selection_sensitivity.py --dataset RPE1
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import anchorop as ao

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

CFG = {
    "K562": {"h5ad": "K562_essential_normalized_singlecell_01.h5ad", "label": "K562_essential"},
    "RPE1": {"h5ad": "rpe1_normalized_singlecell_01.h5ad", "label": "RPE1_essential"},
}
MIN_CELLS_PER_TARGET = 60
MIN_CELLS_PER_GUIDE = 30
MIN_KD_EFF = 0.05
D = 30
N_HVG = 3000
N_TARGETS_KEEP = 200
SEED = 20260927


def _select(target_counts, mode: str, seed: int):
    qualifying = target_counts[target_counts >= MIN_CELLS_PER_TARGET].index.tolist()
    rng = np.random.default_rng(seed)
    if mode == "published":
        return qualifying[:N_TARGETS_KEEP]
    if mode == "random_200":
        return list(rng.choice(qualifying, size=min(N_TARGETS_KEEP, len(qualifying)), replace=False))
    if mode == "count_quantile_200":
        counts = target_counts.loc[qualifying]
        # Cells-per-target as fitness proxy — the h5ad carries no explicit
        # fitness score, so this is the same quantity the original selection
        # biased on. Low count ≈ strong fitness defect (fewer cells survived);
        # high count ≈ mild. 5 quantile bands, 40 targets each.
        quantiles = np.quantile(counts.values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        picks = []
        for i in range(5):
            lo, hi = quantiles[i], quantiles[i + 1]
            band = counts[(counts >= lo) & (counts <= hi)].index.tolist()
            picks.extend(rng.choice(band, size=min(40, len(band)), replace=False))
        return picks
    raise ValueError(mode)


def _fit_and_diagnose(adata, targets_to_keep):
    import scanpy as sc
    from sklearn.decomposition import PCA
    keep = adata.obs["target_gene"].isin(targets_to_keep) | (adata.obs["target_gene"] == "")
    a = adata[keep].copy()
    a.obs.loc[a.obs["target_gene"] != "", "guide"] = "guide_" + a.obs.loc[a.obs["target_gene"] != "", "target_gene"].astype(str)
    # finite-gene filter (RPE1 has a few NaN cols)
    sample = np.asarray(a.X[:min(20000, a.n_obs), :].toarray() if hasattr(a.X, "toarray") else a.X[:min(20000, a.n_obs), :])
    finite = np.isfinite(sample).all(axis=0)
    if not finite.all():
        a = a[:, finite].copy()
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat_v3_paper", subset=True)
    ctrl = (a.obs["target_gene"] == "").to_numpy()
    Xc = np.asarray(a[ctrl].X.toarray() if hasattr(a[ctrl].X, "toarray") else a[ctrl].X, dtype=np.float32)
    pca = PCA(n_components=D, random_state=SEED).fit(Xc)
    loadings = pca.components_.T.astype(np.float32)
    basis = ao.make_program_basis(loadings, a.var_names, method="pca_external",
                                  control_count=int(ctrl.sum()), normalize=False)
    meas = ao.measure_operator(
        a, basis,
        guide_key="guide", target_key="target_gene", control_label="non-targeting",
        min_cells_per_guide=MIN_CELLS_PER_GUIDE, min_knockdown_efficiency=MIN_KD_EFF,
        reg="tsvd", reg_param="path", rank_tol=1e-2, state_label="F7_selection_test",
    )
    S, U = meas.S, meas.U
    hop = ao.held_out_prediction_check(meas, n_folds=5, seed=SEED)
    linc = ao.linearity_check(meas)
    # F4 shared-mode + cos(s,u)
    _, sv, _ = np.linalg.svd(S, full_matrices=False)
    top_frac = float((sv[0] ** 2) / (sv ** 2).sum())
    from collections import defaultdict
    gt = getattr(meas.report, "guide_targets", None) or {n: n.replace("guide_", "") for n in meas.report.retained_guides}
    grouped = defaultdict(list)
    for j, gn in enumerate(meas.report.retained_guides):
        grouped[gt.get(gn, gn)].append(j)
    tnames = sorted(grouped)
    S_tgt = np.stack([S[:, grouped[t]].mean(axis=1) for t in tnames], axis=1)
    Sn = S_tgt / np.maximum(np.linalg.norm(S_tgt, axis=0, keepdims=True), 1e-30)
    C = Sn.T @ Sn
    iu = np.triu_indices_from(C, k=1)
    cos_su = [abs(float(S[:, j] @ U[:, j]) / max(np.linalg.norm(S[:, j]) * np.linalg.norm(U[:, j]), 1e-30))
              for j in range(S.shape[1])]
    return {
        "n_targets_kept": int(len(targets_to_keep)),
        "n_guides_retained": int(len(meas.report.retained_guides)),
        "rho_pooled": float(hop.rho_pooled),
        "rel_diff": float(linc.relative_difference),
        "top_sv_energy_fraction": top_frac,
        "between_target_mean_cos": float(np.mean(C[iu])),
        "median_abs_cos_s_u": float(np.median(cos_su)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CFG), required=True)
    args = ap.parse_args()
    cfg = CFG[args.dataset]
    path = DATA / cfg["h5ad"]
    if not path.exists():
        raise SystemExit(f"h5ad not found: {path}")
    print(f"loading {path} ...")
    adata = ao.load_replogle_h5ad(str(path))
    target_counts = adata.obs.loc[adata.obs["target_gene"] != "", "target_gene"].value_counts()
    print(f"total qualifying targets: {int((target_counts >= MIN_CELLS_PER_TARGET).sum())}")
    out = {"seed": SEED, "dataset": args.dataset, "n_targets_keep": N_TARGETS_KEEP, "modes": {}}
    for mode in ("published", "random_200", "count_quantile_200"):
        keep = _select(target_counts, mode, seed=SEED)
        print(f"\n### mode={mode} n_targets={len(keep)}")
        stats = _fit_and_diagnose(adata, keep)
        print(stats)
        out["modes"][mode] = stats
    dest = OUT / f"F7_{args.dataset}.json"
    dest.write_text(json.dumps(out, indent=2))
    print("saved:", dest)


if __name__ == "__main__":
    main()
