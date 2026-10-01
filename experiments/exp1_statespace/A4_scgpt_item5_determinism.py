"""POST-HOC DIAGNOSIS item 5 — determinism + negative-control + noise re-expression.

Closeout step after A4 halt and items 1–4. Not a re-enabling of
the scGPT arm.

(a) Determinism: encode the same 200 control cells twice with
    identical input. Max ``|Δ|`` in the 512-d native embedding and
    the 30-d PCA head. Confirm ``model.eval()`` and all dropouts
    inactive. Report how ``np.argsort`` ties are broken.

(b) Negative control (CENPJ): CENPJ (q10 target) is never in the
    top-``max_seq_len`` token set per item 1. Its in-silico
    knockdown should give ``u_z = 0`` exactly. Report per-cell
    ``‖E(x_kd) − E(x)‖`` and the maximum / median across 50 cells.
    If non-zero, trace through normalization → binning → top-k
    selection, cell by cell, identifying which step propagates the
    change.

(c) Re-express the item 2 noise floor in light of (a)–(b): is the
    0.93 ratio dominated by cell sampling, by pipeline
    non-determinism, or by both?

Output: ``A4_scgpt_item5_determinism.json``.
"""
from __future__ import annotations

import json
import os
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

OUT = Path(__file__).parent / "A4_scgpt_item5_determinism.json"
RAW_H5AD = REPO / "examples" / "data" / "K562_essential_raw_singlecell_01.h5ad"

N_CONTROL = 200
N_NEG_CTRL = 50
TARGET_SUM = 1e4
CENPJ_ENSEMBL = "ENSG00000151849"
KAPPA = 0.7


def main():
    import anndata as ad
    import scipy.sparse as sp
    import torch

    from anchorop.state_space import ScGPTRep

    print("[ITEM5] loading h5ad …", flush=True)
    a = ad.read_h5ad(RAW_H5AD, backed="r")
    gene_names = list(a.var["gene_name"])
    var_names = list(a.var_names)
    gene_index_ens = {g: i for i, g in enumerate(var_names)}

    ctrl_idx = np.where(
        (a.obs["gene_id"] == "non-targeting").to_numpy())[0][:N_CONTROL]
    Xc = a.X[ctrl_idx]
    if sp.issparse(Xc):
        Xc = Xc.toarray()
    Xc = np.asarray(Xc, dtype=np.float64)
    sums = Xc.sum(axis=1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    X_log = np.log1p(Xc * (TARGET_SUM / sums))
    print(f"[ITEM5] X_log shape {X_log.shape}", flush=True)

    print(f"[ITEM5] fitting ScGPTRep(30) on {N_CONTROL} NT control cells …",
          flush=True)
    t0 = time.time()
    rep = ScGPTRep(d_out=30, checkpoint_path=os.environ["SCGPT_CKPT"])
    rep.fit(X_log, gene_names=gene_names)
    print(f"[ITEM5]   fit: {time.time()-t0:.1f}s", flush=True)

    model = rep._encoder["model"]
    print(f"[ITEM5]   model.training = {model.training} (expect False)",
          flush=True)
    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    print(f"[ITEM5]   #Dropout modules = {len(dropout_modules)}; "
          f"first few p = "
          f"{[round(d.p, 3) for d in dropout_modules[:5]]}",
          flush=True)

    # ─── (a) determinism check ────────────────────────────────────────
    print(f"[ITEM5] (a) determinism — encode the same {N_CONTROL} cells "
          "twice with identical input …", flush=True)
    t0 = time.time()
    Z_native_1 = rep._embed_with_scgpt(X_log)
    print(f"[ITEM5]   encode 1: {time.time()-t0:.1f}s, "
          f"Z shape {Z_native_1.shape}", flush=True)
    t0 = time.time()
    Z_native_2 = rep._embed_with_scgpt(X_log)
    print(f"[ITEM5]   encode 2: {time.time()-t0:.1f}s", flush=True)
    max_native = float(np.max(np.abs(Z_native_1 - Z_native_2)))
    median_native = float(np.median(np.abs(Z_native_1 - Z_native_2)))
    print(f"[ITEM5]   max |Δ| in 512-d native: {max_native:.6g}", flush=True)
    print(f"[ITEM5]   median |Δ| in 512-d native: {median_native:.6g}",
          flush=True)

    Z_head_1 = rep._pca_head.transform(Z_native_1)
    Z_head_2 = rep._pca_head.transform(Z_native_2)
    max_head = float(np.max(np.abs(Z_head_1 - Z_head_2)))
    median_head = float(np.median(np.abs(Z_head_1 - Z_head_2)))
    print(f"[ITEM5]   max |Δ| in 30-d head: {max_head:.6g}", flush=True)
    print(f"[ITEM5]   median |Δ| in 30-d head: {median_head:.6g}",
          flush=True)

    # argsort tie-breaking: numpy's default is 'quicksort', which is
    # not stable. We use default argsort in _tokenize_cells. Document
    # exactly what that means.
    tie_test = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    tie_order = np.argsort(-np.abs(tie_test))
    print(f"[ITEM5]   np.argsort tie-breaking test: all-equal input "
          f"[0.5, 0.5, 0.5, 0.5, 0.5] → argsort = {tie_order.tolist()}; "
          f"numpy default kind = 'quicksort', NOT stable.", flush=True)

    item5a = {
        "model_training_flag": bool(model.training),
        "n_dropout_modules": len(dropout_modules),
        "first_dropout_p": [float(d.p) for d in dropout_modules[:5]],
        "max_abs_delta_native": max_native,
        "median_abs_delta_native": median_native,
        "max_abs_delta_head": max_head,
        "median_abs_delta_head": median_head,
        "argsort_tie_breaking": {
            "numpy_default_kind": "quicksort",
            "stable": False,
            "note": (
                "np.argsort(-np.abs(x))[:k] with default kind='quicksort' "
                "is not stable. For a given cell, if the k-th and (k+1)-th "
                "absolute values are equal, the token set is implementation-"
                "defined. Across two IDENTICAL input runs the result is "
                "deterministic within a single process, but minor "
                "perturbations to a single gene can non-deterministically "
                "rearrange ties among OTHER genes."),
        },
    }

    # ─── (b) negative control: CENPJ ──────────────────────────────────
    print(f"[ITEM5] (b) negative control — CENPJ ({CENPJ_ENSEMBL}) is never "
          "in top-max_seq_len; u_z should be 0 for every control cell …",
          flush=True)
    cenpj_idx = gene_index_ens[CENPJ_ENSEMBL]
    cells = X_log[:N_NEG_CTRL]
    per_cell_diffs = []
    tokenization_changes = []
    binning_changes = []  # scGPT does per-cell binning inside the model;
                           # here we track whether top-k input values differ.
    perturbed_rows = []
    for i, row in enumerate(cells):
        row_perturbed = row.copy()
        # knockdown-scale: scale CENPJ in count space, re-log
        y = np.expm1(row_perturbed[cenpj_idx])
        row_perturbed[cenpj_idx] = np.log1p(max(y, 0.0) * (1 - KAPPA))
        perturbed_rows.append(row_perturbed)
    perturbed = np.stack(perturbed_rows)

    print(f"[ITEM5]   encoding {N_NEG_CTRL} control cells (unperturbed) …",
          flush=True)
    Z_ctrl = rep.encode(cells)
    print(f"[ITEM5]   encoding {N_NEG_CTRL} control cells (CENPJ knockdown) …",
          flush=True)
    Z_pert = rep.encode(perturbed)

    diffs = np.linalg.norm(Z_ctrl - Z_pert, axis=1)
    print(f"[ITEM5]   per-cell ‖E(x_kd) − E(x)‖  "
          f"min={diffs.min():.6g}, median={np.median(diffs):.6g}, "
          f"max={diffs.max():.6g}", flush=True)
    print(f"[ITEM5]   #cells with ‖Δ‖ > 1e-6: "
          f"{int(np.sum(diffs > 1e-6))} / {N_NEG_CTRL}", flush=True)

    # Trace propagation step by step for the first cell with a non-zero Δ.
    if np.any(diffs > 1e-6):
        bad_i = int(np.argmax(diffs))
        row_c = cells[bad_i]
        row_p = perturbed_rows[bad_i]
        # Normalization step: already done; both go through log1p + already
        # normalized. The per-gene diff:
        row_diff = row_p - row_c
        n_nonzero_diff = int(np.sum(np.abs(row_diff) > 1e-15))
        print(f"[ITEM5]   trace cell {bad_i}: #genes with log1p-value "
              f"change = {n_nonzero_diff} (expected 1 — CENPJ only)",
              flush=True)
        # top-k selection: do the two inputs tokenize identically?
        from anchorop.state_space.scgpt import _tokenize_cells
        src_c, val_c, mask_c = _tokenize_cells(
            rep._encoder, row_c[None, :], gene_names)
        src_p, val_p, mask_p = _tokenize_cells(
            rep._encoder, row_p[None, :], gene_names)
        same_src = bool(torch.equal(src_c, src_p))
        same_val = bool(torch.allclose(val_c, val_p, atol=1e-10))
        same_mask = bool(torch.equal(mask_c, mask_p))
        print(f"[ITEM5]   cell {bad_i} top-k token IDs identical: "
              f"{same_src}", flush=True)
        print(f"[ITEM5]   cell {bad_i} top-k values identical "
              f"(atol 1e-10): {same_val}", flush=True)
        print(f"[ITEM5]   cell {bad_i} padding mask identical: "
              f"{same_mask}", flush=True)
        if not same_src:
            diff_pos = int(torch.sum(src_c != src_p).item())
            print(f"[ITEM5]   #token-ID positions that differ: {diff_pos} "
                  "— top-k selection reshuffled due to argsort tie-breaking.",
                  flush=True)
        trace = {
            "cell_index": bad_i,
            "u_z_norm_this_cell": float(diffs[bad_i]),
            "n_gene_log1p_values_changed": n_nonzero_diff,
            "topk_token_ids_identical": same_src,
            "topk_values_identical_atol_1e-10": same_val,
            "topk_padding_mask_identical": same_mask,
            "diff_positions_in_src": int(torch.sum(src_c != src_p).item())
                                      if not same_src else 0,
        }
    else:
        trace = {"all_delta_below_1e-6": True}

    item5b = {
        "target": "CENPJ",
        "target_ensembl": CENPJ_ENSEMBL,
        "kappa": KAPPA,
        "n_cells": N_NEG_CTRL,
        "per_cell_delta_norm_summary": {
            "min": float(diffs.min()),
            "median": float(np.median(diffs)),
            "max": float(diffs.max()),
            "n_above_1e-6": int(np.sum(diffs > 1e-6)),
        },
        "propagation_trace": trace,
    }

    # ─── (c) re-express the item 2 noise floor ────────────────────────
    print(f"[ITEM5] (c) re-expressing the item 2 noise floor …",
          flush=True)
    # Load item 2 result from the earlier diagnosis JSON.
    item2_path = Path(__file__).parent / "A4_scgpt_diagnosis.json"
    if item2_path.exists():
        item2_data = json.loads(item2_path.read_text())["item2_noise_floor_cdc27"]
        print(f"[ITEM5]   item 2 recap: mean ‖u_z‖ = "
              f"{item2_data['mean_u_z_norm']:.4g}, "
              f"between-subset norm = "
              f"{item2_data['mean_between_subset_norm']:.4g}, "
              f"ratio = {item2_data['ratio_u_over_noise_floor']:.3f}",
              flush=True)
    else:
        item2_data = None

    # Compare: pipeline-determinism budget (b.max) vs between-subset norm.
    pipeline_det = diffs.max() if item2_data else None
    if item2_data:
        between_subset = item2_data["mean_between_subset_norm"]
    print(f"[ITEM5]   pipeline-determinism budget (max per-cell CENPJ Δ) = "
          f"{pipeline_det:.4g}", flush=True)
    if item2_data:
        print(f"[ITEM5]   between-subset norm (item 2) = "
              f"{between_subset:.4g}", flush=True)
        print(f"[ITEM5]   determinism / between-subset = "
              f"{pipeline_det/between_subset:.3f} "
              "(fraction of noise explained by pipeline non-determinism "
              "of a single gene change on CENPJ)", flush=True)
    item5c = {
        "note": (
            "CENPJ is a no-op perturbation for scGPT (not in top-k, so "
            "the input to the transformer should be identical). Any "
            "non-zero u_z for CENPJ is PIPELINE non-determinism, not "
            "biology. The pipeline-determinism budget from item 5(b) is "
            "compared to the item-2 cell-sampling noise floor; if the "
            "two are of similar order, the noise floor itself is "
            "dominated by pipeline non-determinism."),
        "pipeline_determinism_budget_cenpj_max": pipeline_det,
        "between_subset_norm_item2": (
            between_subset if item2_data else None),
        "ratio_pipeline_over_between_subset": (
            pipeline_det / between_subset if item2_data else None),
    }

    report = {
        "status": "POST-HOC DIAGNOSIS item 5 — NOT RE-ENABLING THE scGPT ARM",
        "scope_note": (
            "Items 1–5 are diagnosis after the A4 halt verdict. Per the "
            "exp1 PREREG amendment 2 and the user closeout instruction: "
            "no workaround re-enables the scGPT arm."),
        "n_control_cells": N_CONTROL,
        "item5a_determinism": item5a,
        "item5b_negative_control": item5b,
        "item5c_noise_floor_reexpression": item5c,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[ITEM5] saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
