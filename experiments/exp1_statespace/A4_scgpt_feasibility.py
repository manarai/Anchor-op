"""Run the scGPT A4 feasibility check on real K562 NT control cells.

Preregistered in A4 (revised 2026-09-30) and clarified in PREREG
amendment 2. Loads the scGPT_human whole-pretrain checkpoint via the
state-dict conversion in ``anchorop.state_space.scgpt``, fits a
PCA-30 head on 200 non-targeting-control cells, then runs the
knockdown-scale FD on 50-cell disjoint subsets at κ ∈ {0.5, 0.7,
0.9} for a representative measurement-bundle target gene.

Verdict → ``experiments/exp1_statespace/A4_scgpt_feasibility.json``.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ["SCGPT_CKPT"] = str(
    REPO / "experiments" / "exp1_statespace" / "weights" / "scGPT_human" / "best_model.pt")

OUT = Path(__file__).parent / "A4_scgpt_feasibility.json"
RAW_H5AD = REPO / "examples" / "data" / "K562_essential_raw_singlecell_01.h5ad"
PKL = REPO / "results" / "k562_essential_measurement.pkl"

N_CONTROL = 200
N_SUBSET = 50
KAPPAS = (0.5, 0.7, 0.9)
COS_THRESHOLD = 0.9
RANDOM_STATE = 0
TARGET_SUM = 1e4


def main():
    import anndata as ad
    import scipy.sparse as sp

    from anchorop.state_space import ScGPTRep

    print(f"[A4] loading {RAW_H5AD.name} …", flush=True)
    a = ad.read_h5ad(RAW_H5AD, backed="r")

    ctrl_idx = np.where(
        (a.obs["gene_id"] == "non-targeting").to_numpy())[0][:N_CONTROL]
    Xc = a.X[ctrl_idx]
    if sp.issparse(Xc):
        Xc = Xc.toarray()
    Xc = np.asarray(Xc, dtype=np.float64)
    sums = Xc.sum(axis=1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    X_log = np.log1p(Xc * (TARGET_SUM / sums))
    gene_names = list(a.var["gene_name"])
    print(f"[A4] log1p X shape {X_log.shape}", flush=True)

    print(f"[A4] fitting ScGPTRep(30) on {N_CONTROL} NT control cells …",
          flush=True)
    t0 = time.time()
    rep = ScGPTRep(d_out=30, checkpoint_path=os.environ["SCGPT_CKPT"])
    rep.fit(X_log, gene_names=gene_names)
    print(f"[A4] fit: {time.time()-t0:.1f}s   "
          f"native_dim={rep._ckpt.native_dim}   "
          f"PCA var retained: "
          f"{float(np.sum(rep._pca_head.explained_variance_ratio_)):.3f}",
          flush=True)

    # Target gene = first measurement-bundle target (first ENSEMBL).
    bundle = pickle.load(open(PKL, "rb"))
    retained = list(bundle["measurement"].report.retained_guides)
    targets = sorted({
        str(g)[6:] if str(g).startswith("guide_") else str(g).split("_")[0]
        for g in retained
    })
    var_names = list(a.var_names)
    gene_index_ens = {g: i for i, g in enumerate(var_names)}
    t_ens = targets[0]
    t_idx = gene_index_ens[t_ens]
    print(f"[A4] feasibility check: target {t_ens} ({gene_names[t_idx]}), "
          f"n_subset={N_SUBSET}, κ ∈ {KAPPAS}", flush=True)

    t0 = time.time()
    report = rep.check_decode_direction_feasibility(
        X_control=X_log, target_gene_idx=t_idx,
        kappas=KAPPAS, n_subset=N_SUBSET,
        cos_threshold=COS_THRESHOLD, random_state=RANDOM_STATE,
    )
    print(f"[A4] elapsed: {time.time()-t0:.1f}s", flush=True)

    print(f"[A4] non_trivial={report['non_trivial']}", flush=True)
    print(f"[A4] knockdown_norm={report['knockdown_norm']:.5g}, "
          f"noise_norm={report['noise_norm']:.5g}", flush=True)
    print(f"[A4] kappa_cosines (within subset A):", flush=True)
    for k, v in report["kappa_cosines"].items():
        print(f"[A4]   {k}: {v:.4f}", flush=True)
    print(f"[A4] subset_cosines (A vs B at each κ):", flush=True)
    for k, v in report["subset_cosines"].items():
        print(f"[A4]   κ={k}: {v:.4f}", flush=True)
    print(f"[A4] stable={report['stable']}", flush=True)
    print(f"[A4] halt_reason={report['halt_reason']}", flush=True)

    rep_js = {k: v for k, v in report.items() if k != "checkpoint"}
    rep_js["checkpoint"] = dict(
        name=report["checkpoint"].name,
        sha256=report["checkpoint"].sha256,
        native_dim=report["checkpoint"].native_dim,
        continuous_input_supported=report["checkpoint"].continuous_input_supported,
    )
    rep_js["target_gene_ensembl"] = t_ens
    rep_js["target_gene_symbol"] = gene_names[t_idx]
    rep_js["n_control_cells"] = int(X_log.shape[0])
    rep_js["n_subset"] = N_SUBSET
    rep_js["cos_threshold"] = COS_THRESHOLD
    rep_js["kappas"] = list(KAPPAS)
    OUT.write_text(json.dumps(rep_js, indent=2, default=str))
    print(f"[A4] saved: {OUT}", flush=True)
    if report["halt_reason"]:
        print(f"[A4] VERDICT: HALT — {report['halt_reason']}", flush=True)
        sys.exit(2)
    print(f"[A4] VERDICT: PASS", flush=True)


if __name__ == "__main__":
    main()
