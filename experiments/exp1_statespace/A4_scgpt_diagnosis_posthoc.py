"""POST-HOC DIAGNOSIS of the A4 halt (closeout item). NOT a re-enabling
of the scGPT arm, NOT part of Task 4, NOT a preregistered test.

Four items, in order:

1. **Tokenization check (CDC27 + 5 prespecified targets)**. For each
   target, fraction of control cells where the target is in the
   top-``max_seq_len = 1200`` input tokens before scaling, and the
   fraction where it drops out after scaling at κ ∈ {0.5, 0.7, 0.9}.

2. **Correct noise floor for clause (i)**. The A4 "noise_norm" was
   computed as the encoder's run-to-run noise on *identical input*
   (zero by construction for a deterministic forward pass). The
   proper noise floor for clause (i) is the between-subset
   (cell-sampling) variability of ``u_z``. We compute the mean
   pairwise ``u_z`` norm across 10 random disjoint 25-cell subset
   pairs at κ = 0.7 on CDC27 and report ``‖u_z‖`` relative to it.

3. **Generality on 5 more targets**. Prespecified selection rule
   (stated here before running): pick 5 targets at the 10th, 30th,
   50th, 70th, 90th percentiles of control mean expression (log1p
   space) of the 188 measurement-bundle targets. Run the identical
   A4 feasibility check (same ``n_subset = 50``, same κ ∈ {0.5,
   0.7, 0.9}, same cos threshold 0.9) on each. Report how many
   pass (expected: zero or close to zero, per the halt verdict).

4. **Force-include diagnosis (conditional on #1)**. If #1 confirms
   the target gene drops out of the top-``max_seq_len`` after
   scaling for an appreciable fraction of cells, rerun the A4
   feasibility check on CDC27 with the target gene *force-included*
   in the token set (kept at the first position, value scaled).
   Report whether the response then becomes graded in κ and stable
   across subsets. Diagnosis only — does NOT re-enable the scGPT
   arm per A4.

Output: ``A4_scgpt_diagnosis.json``.
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

OUT = Path(__file__).parent / "A4_scgpt_diagnosis.json"
RAW_H5AD = REPO / "examples" / "data" / "K562_essential_raw_singlecell_01.h5ad"
PKL = REPO / "results" / "k562_essential_measurement.pkl"

N_CONTROL = 200
N_SUBSET = 50
NOISE_N_SUBSET = 25
NOISE_N_PAIRS = 10
KAPPAS = (0.5, 0.7, 0.9)
COS_THRESHOLD = 0.9
RANDOM_STATE = 0
TARGET_SUM = 1e4
MAX_SEQ_LEN = 1200   # scGPT_human args.json
QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)
DROPOUT_FRAC_THRESHOLD = 0.10  # trigger item 4 if drop-out exceeds 10 %


def main():
    import anndata as ad
    import scipy.sparse as sp

    from anchorop.state_space import ScGPTRep

    print("[DIAG] loading h5ad + bundle …", flush=True)
    a = ad.read_h5ad(RAW_H5AD, backed="r")
    gene_names = list(a.var["gene_name"])
    var_names = list(a.var_names)
    gene_index_ens = {g: i for i, g in enumerate(var_names)}
    bundle = pickle.load(open(PKL, "rb"))
    retained = list(bundle["measurement"].report.retained_guides)
    targets = sorted({
        str(g)[6:] if str(g).startswith("guide_") else str(g).split("_")[0]
        for g in retained
    })

    # Load controls + log1p.
    ctrl_idx = np.where((a.obs["gene_id"] == "non-targeting").to_numpy())[0][:N_CONTROL]
    Xc = a.X[ctrl_idx]
    if sp.issparse(Xc):
        Xc = Xc.toarray()
    Xc = np.asarray(Xc, dtype=np.float64)
    sums = Xc.sum(axis=1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    X_log = np.log1p(Xc * (TARGET_SUM / sums))

    # ── Item 3 prespecified: pick 5 targets by control-mean quantile ──
    control_mean = X_log.mean(axis=0)
    target_means = {t: float(control_mean[gene_index_ens[t]])
                    for t in targets if t in gene_index_ens}
    sorted_targets = sorted(target_means.items(), key=lambda kv: kv[1])
    n = len(sorted_targets)
    quantile_targets = []
    for q in QUANTILES:
        i = int(round(q * (n - 1)))
        quantile_targets.append(sorted_targets[i])
    print(f"[DIAG] prespecified by control-mean quantile:", flush=True)
    for q, (t, m) in zip(QUANTILES, quantile_targets):
        print(f"[DIAG]   q{int(q*100):02d}: {t}  (mean = {m:.3f}, "
              f"symbol = {gene_names[gene_index_ens[t]]})", flush=True)
    # CDC27 as the A4 reference target.
    cdc27 = "ENSG00000004897"
    cdc27_idx = gene_index_ens[cdc27]
    print(f"[DIAG] CDC27 ({cdc27}, idx {cdc27_idx}, mean = "
          f"{control_mean[cdc27_idx]:.3f}, symbol = {gene_names[cdc27_idx]})",
          flush=True)

    # ── Item 1: tokenization check ─────────────────────────────────────
    print(f"[DIAG] item 1 — tokenization check (top {MAX_SEQ_LEN} by "
          f"|expression|) …", flush=True)
    all_targets_item1 = [("CDC27", cdc27_idx)] + [
        (f"q{int(q*100):02d}:{gene_names[gene_index_ens[t]]}", gene_index_ens[t])
        for q, (t, _) in zip(QUANTILES, quantile_targets)
    ]
    item1 = {}
    for label, idx in all_targets_item1:
        in_pre = []
        in_post = {k: [] for k in KAPPAS}
        for i in range(X_log.shape[0]):
            abs_row = np.abs(X_log[i])
            top = np.argsort(-abs_row)[:MAX_SEQ_LEN]
            in_pre.append(idx in set(top.tolist()))
            for kappa in KAPPAS:
                perturbed = X_log[i].copy()
                y = np.expm1(perturbed[idx])
                perturbed[idx] = np.log1p(max(y, 0.0) * (1 - kappa))
                abs_row2 = np.abs(perturbed)
                top2 = np.argsort(-abs_row2)[:MAX_SEQ_LEN]
                in_post[kappa].append(idx in set(top2.tolist()))
        item1[label] = {
            "fraction_in_top_1200_pre": float(np.mean(in_pre)),
            "fraction_in_top_1200_post": {str(k): float(np.mean(v)) for k, v in in_post.items()},
            "fraction_dropped_out": {str(k): float(np.mean(
                [p and not q for p, q in zip(in_pre, in_post[k])])) for k in KAPPAS},
        }
        print(f"[DIAG]   {label}: pre={item1[label]['fraction_in_top_1200_pre']:.3f}, "
              f"post {dict(item1[label]['fraction_in_top_1200_post'])}, "
              f"dropped {dict(item1[label]['fraction_dropped_out'])}", flush=True)
    # Decide whether to run item 4.
    cdc27_dropout = item1["CDC27"]["fraction_dropped_out"][str(KAPPAS[1])]
    run_item4 = bool(cdc27_dropout > DROPOUT_FRAC_THRESHOLD)
    print(f"[DIAG] CDC27 drop-out at κ=0.7 = {cdc27_dropout:.3f}; "
          f"item 4 trigger threshold = {DROPOUT_FRAC_THRESHOLD}; "
          f"run_item4 = {run_item4}", flush=True)

    # ── Fit ScGPTRep once; reuse across items 2–4 ──────────────────────
    print(f"[DIAG] fitting ScGPTRep(30) on {N_CONTROL} NT control cells "
          f"(reused across items 2–4) …", flush=True)
    t0 = time.time()
    rep = ScGPTRep(d_out=30, checkpoint_path=os.environ["SCGPT_CKPT"])
    rep.fit(X_log, gene_names=gene_names)
    print(f"[DIAG]   fit: {time.time()-t0:.1f}s", flush=True)

    # ── Item 2: correct noise floor on CDC27 ───────────────────────────
    print(f"[DIAG] item 2 — correct noise floor on CDC27 via "
          f"{NOISE_N_PAIRS} disjoint {NOISE_N_SUBSET}-cell subset pairs "
          f"at κ=0.7 …", flush=True)
    rng = np.random.default_rng(RANDOM_STATE + 100)
    kd_norms = []
    pairwise_norms = []
    for p in range(NOISE_N_PAIRS):
        perm = rng.permutation(X_log.shape[0])
        sa = perm[:NOISE_N_SUBSET]; sb = perm[NOISE_N_SUBSET:2*NOISE_N_SUBSET]
        u_a = rep.knockdown_scale_difference(
            X_log[sa], cdc27_idx, kappa=0.7, input_space="log1p")
        u_b = rep.knockdown_scale_difference(
            X_log[sb], cdc27_idx, kappa=0.7, input_space="log1p")
        kd_norms.append(float((np.linalg.norm(u_a) + np.linalg.norm(u_b)) / 2))
        pairwise_norms.append(float(np.linalg.norm(u_a - u_b) / np.sqrt(2)))
        print(f"[DIAG]   pair {p+1}/{NOISE_N_PAIRS}: ‖u_a‖={np.linalg.norm(u_a):.4g}, "
              f"‖u_b‖={np.linalg.norm(u_b):.4g}, "
              f"‖u_a − u_b‖/√2 = {pairwise_norms[-1]:.4g}", flush=True)
    noise_floor = float(np.mean(pairwise_norms))
    mean_u_norm = float(np.mean(kd_norms))
    item2 = {
        "n_pairs": NOISE_N_PAIRS,
        "n_subset": NOISE_N_SUBSET,
        "mean_u_z_norm": mean_u_norm,
        "mean_between_subset_norm": noise_floor,
        "ratio_u_over_noise_floor": mean_u_norm / max(noise_floor, 1e-30),
        "deterministic_forward_noise": 0.0,
        "note": (
            "A4 clause (i) ran against the deterministic forward-pass "
            "noise, which is 0 by construction for a frozen network on "
            "identical input. The 'proper' noise floor for stability is "
            "the between-subset (cell-sampling) variability reported "
            "here. If the mean ‖u_z‖ is at the same order of magnitude "
            "as the between-subset norm (ratio ~ 1), the knockdown "
            "signal is at noise."),
    }
    print(f"[DIAG]   mean ‖u_z‖ = {mean_u_norm:.4g}, between-subset norm = "
          f"{noise_floor:.4g}, ratio = {item2['ratio_u_over_noise_floor']:.3f}",
          flush=True)

    # ── Item 3: feasibility on 5 more targets ──────────────────────────
    print(f"[DIAG] item 3 — feasibility check on 5 prespecified targets "
          f"(n_subset = {N_SUBSET}, κ ∈ {KAPPAS}) …", flush=True)
    item3 = {}
    for q, (t_ens, mean_val) in zip(QUANTILES, quantile_targets):
        t_idx = gene_index_ens[t_ens]
        label = f"q{int(q*100):02d}:{gene_names[t_idx]}"
        print(f"[DIAG]   target {label} ({t_ens}, mean = {mean_val:.3f}) …", flush=True)
        t0 = time.time()
        r = rep.check_decode_direction_feasibility(
            X_control=X_log, target_gene_idx=t_idx,
            kappas=KAPPAS, n_subset=N_SUBSET,
            cos_threshold=COS_THRESHOLD, random_state=RANDOM_STATE)
        print(f"[DIAG]     elapsed {time.time()-t0:.1f}s, non_trivial="
              f"{r['non_trivial']}, stable={r['stable']}, "
              f"halt_reason={'(none)' if not r['halt_reason'] else r['halt_reason']}",
              flush=True)
        item3[label] = {
            "target_ensembl": t_ens,
            "control_mean_log1p": mean_val,
            "quantile": q,
            "non_trivial": r["non_trivial"],
            "stable": r["stable"],
            "knockdown_norm": r["knockdown_norm"],
            "kappa_cosines": r["kappa_cosines"],
            "subset_cosines": r["subset_cosines"],
            "halt_reason": r["halt_reason"],
        }
    n_pass = sum(1 for v in item3.values() if not v["halt_reason"])
    print(f"[DIAG]   {n_pass} / {len(item3)} targets pass A4 feasibility", flush=True)

    # ── Item 4: force-include diagnosis (CDC27), if triggered ──────────
    item4 = None
    if run_item4:
        print(f"[DIAG] item 4 — force-include diagnosis on CDC27 …", flush=True)
        # Monkey-patch the ScGPT tokenization so CDC27 is at position 0.
        from anchorop.state_space import scgpt as scgpt_module
        orig_tokenize = scgpt_module._tokenize_cells

        def _tokenize_with_force(encoder, X_, var_gene_names):
            import torch
            vocab = encoder["vocab"]
            cfg = encoder["config"]
            max_seq_len = int(cfg.get("max_seq_len", 1200))
            pad_token = cfg.get("pad_token", "<pad>")
            pad_value = float(cfg.get("pad_value", -2.0))
            pad_idx = int(vocab[pad_token])
            gene_ids = np.array([vocab[g] for g in var_gene_names],
                                 dtype=np.int64)
            batch = X_.shape[0]
            src = np.full((batch, max_seq_len), pad_idx, dtype=np.int64)
            values = np.full((batch, max_seq_len), pad_value, dtype=np.float32)
            mask = np.ones((batch, max_seq_len), dtype=bool)
            for i in range(batch):
                row = X_[i]
                # Force CDC27 at position 0.
                abs_row = np.abs(row).copy()
                abs_row[cdc27_idx] = np.inf
                nz = np.argsort(-abs_row)[:max_seq_len]
                src[i, : len(nz)] = gene_ids[nz]
                values[i, : len(nz)] = row[nz].astype(np.float32)
                mask[i, : len(nz)] = False
            return (torch.from_numpy(src), torch.from_numpy(values),
                    torch.from_numpy(mask))

        scgpt_module._tokenize_cells = _tokenize_with_force
        try:
            t0 = time.time()
            r = rep.check_decode_direction_feasibility(
                X_control=X_log, target_gene_idx=cdc27_idx,
                kappas=KAPPAS, n_subset=N_SUBSET,
                cos_threshold=COS_THRESHOLD, random_state=RANDOM_STATE)
            print(f"[DIAG]   elapsed {time.time()-t0:.1f}s, "
                  f"non_trivial={r['non_trivial']}, stable={r['stable']}, "
                  f"halt_reason={'(none)' if not r['halt_reason'] else r['halt_reason']}",
                  flush=True)
            item4 = {
                "target": "CDC27",
                "mode": "force-include target at token position 0",
                "non_trivial": r["non_trivial"],
                "stable": r["stable"],
                "knockdown_norm": r["knockdown_norm"],
                "kappa_cosines": r["kappa_cosines"],
                "subset_cosines": r["subset_cosines"],
                "halt_reason": r["halt_reason"],
                "note": (
                    "Diagnosis only. Does NOT re-enable the scGPT arm "
                    "per A4; A4's halt verdict stands."),
            }
        finally:
            scgpt_module._tokenize_cells = orig_tokenize
    else:
        item4 = {"skipped": True,
                 "reason": f"CDC27 drop-out at κ=0.7 = {cdc27_dropout:.3f} <= "
                           f"{DROPOUT_FRAC_THRESHOLD}; item 4 not triggered."}
        print(f"[DIAG] item 4 skipped: CDC27 drop-out not above trigger "
              f"threshold.", flush=True)

    # ── Write report ──────────────────────────────────────────────────
    report = {
        "status": "POST-HOC DIAGNOSIS — DOES NOT RE-ENABLE THE scGPT ARM",
        "scope_note": (
            "Items 1–4 are diagnosis after the A4 halt verdict was recorded. "
            "Per the exp1 PREREG amendment 2 and the user instruction at "
            "closeout: no workaround re-enables the scGPT arm."),
        "n_control_cells": N_CONTROL,
        "max_seq_len": MAX_SEQ_LEN,
        "kappas": list(KAPPAS),
        "cos_threshold": COS_THRESHOLD,
        "random_state": RANDOM_STATE,
        "item1_tokenization": item1,
        "item2_noise_floor_cdc27": item2,
        "item3_five_targets": item3,
        "item4_force_include": item4,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[DIAG] saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
