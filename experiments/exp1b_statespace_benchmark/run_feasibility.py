"""B4 driver — run the 5-clause feasibility gate on all arms × all screens.

Loads each screen's control-cell matrix via the paper-1 pipeline (same HVG,
force-included targets, filters) and the K562/RPE1 raw-count h5ad for the
count-mode arms (LDVAE, scVI). Fits one instance of each arm per screen, then
runs `feasibility.run_gate(...)` on 6 prespecified targets. Writes
`experiments/exp1b_statespace_benchmark/feasibility_table.json`.

**Do not launch** without first reviewing the compute estimate at the bottom of
PREREG_exp1b.md § 14. Set `ARMS` / `SCREENS` lists below to a subset for a dry
run; a full run (all 7 arms × 3 screens, scGPT included) is ~1-2 days on local
CPU/MPS.
"""
from __future__ import annotations
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/exp1b_statespace_benchmark"))
sys.path.insert(0, str(REPO / "src"))

from arms import (
    PCALog1pArm, FAQRLog1pArm, LDVAEEncoderArm, LDVAELoadingsArm,
    SCVIArm, ScGPTFixedIngestionArm,
)
from feasibility import pick_prespecified_targets, run_gate

OUT = Path(__file__).resolve().parent / "feasibility_table.json"

# ───────── Which arms × which screens for THIS run ─────────
# Default: everything. Shrink before launch to control compute.
SCREENS = ("K562_essential", "RPE1_essential", "Jost_2020")
ARMS = (
    "PCA",         # log1p linear reference
    "FA_QR",       # log1p FA, QR-orthonormalized (reference arm per PREREG)
    "LDVAE_enc",   # LDVAE encoder (count input, posterior mean)
    "LDVAE_load",  # LDVAE loadings, QR'd (log1p linear basis)
    "scVI",        # scVI (count input, posterior mean)
    "scGPT_fixed", # scGPT_human with fixed deterministic ingestion
    "scGPT_rand",  # scGPT with random weights, same ingestion
)

VAE_SEEDS = (0, 1, 2)   # 3 training seeds for the full benchmark; feasibility uses seed 0 + extras for (d) cross-seed
VAE_EPOCHS_FEASIBILITY = 100   # kept small for feasibility; full benchmark uses 400


def load_screen(screen: str):
    """Return (adata_hvg, X_ctrl_log1p, X_ctrl_counts, gene_names, batch_key).

    Reuses the exp1-statespace fit scripts to avoid duplicating pipeline logic.
    This function is a stub — the actual loader must mirror
    reproduction/55_fa_table1_recheck.py's HVG + force-included-target logic
    on the raw-count h5ad (for count-mode arms) and on the log1p-normalized
    matrix (for log1p-mode arms). Not implemented in this B2+B3 commit; the
    full loader lands in B4's run.
    """
    raise NotImplementedError(
        f"load_screen({screen}) stub — populate before launching run_feasibility.py; "
        "mirror reproduction/55_fa_table1_recheck.py's HVG + force-included logic.")


def main():
    results = {}
    for screen in SCREENS:
        print(f"\n{'='*72}\n{screen}\n{'='*72}", flush=True)
        adata, X_log1p, X_counts, gene_names, batch_key = load_screen(screen)
        targets = pick_prespecified_targets(X_log1p, gene_names)
        results[screen] = {"targets": [(n, i) for n, i in targets], "arms": {}}

        for arm_tag in ARMS:
            print(f"\n--- {screen} / {arm_tag} ---", flush=True)
            t0 = time.time()
            if arm_tag == "PCA":
                arm = PCALog1pArm(dim=30).fit(X_log1p)
                extras = None
            elif arm_tag == "FA_QR":
                arm = FAQRLog1pArm(dim=30).fit(X_log1p)
                extras = None
            elif arm_tag == "LDVAE_enc":
                arm = LDVAEEncoderArm(dim=30, train_seed=VAE_SEEDS[0],
                                       n_epochs=VAE_EPOCHS_FEASIBILITY).fit(adata, batch_key=batch_key)
                extras = [LDVAEEncoderArm(dim=30, train_seed=s,
                                            n_epochs=VAE_EPOCHS_FEASIBILITY).fit(adata, batch_key=batch_key)
                           for s in VAE_SEEDS[1:]]
            elif arm_tag == "LDVAE_load":
                # Reuses the trained encoder from LDVAE_enc.
                ldvae = results[screen]["arms"].get("LDVAE_enc", {}).get("_arm")
                if ldvae is None:
                    ldvae = LDVAEEncoderArm(dim=30, train_seed=VAE_SEEDS[0],
                                             n_epochs=VAE_EPOCHS_FEASIBILITY).fit(adata, batch_key=batch_key)
                arm = LDVAELoadingsArm(dim=30).fit(ldvae).set_mean(X_log1p)
                extras = None
            elif arm_tag == "scVI":
                arm = SCVIArm(dim=30, train_seed=VAE_SEEDS[0],
                               n_epochs=VAE_EPOCHS_FEASIBILITY).fit(adata, batch_key=batch_key)
                extras = [SCVIArm(dim=30, train_seed=s,
                                    n_epochs=VAE_EPOCHS_FEASIBILITY).fit(adata, batch_key=batch_key)
                           for s in VAE_SEEDS[1:]]
            elif arm_tag == "scGPT_fixed":
                arm = ScGPTFixedIngestionArm(dim=30).fit(adata)
                extras = None
            elif arm_tag == "scGPT_rand":
                arm = ScGPTFixedIngestionArm(dim=30, random_weights=True).fit(adata)
                extras = None
            else:
                raise ValueError(arm_tag)
            print(f"  fit took {time.time()-t0:.1f}s", flush=True)

            # pick X_ctrl appropriate for the arm's input_space
            X_ctrl = X_log1p if arm.input_space == "log1p" else X_counts

            t1 = time.time()
            gate = run_gate(arm, X_ctrl, gene_names, targets,
                              kappas=(0.5, 0.7, 0.9), n_pairs=20, subset=50,
                              seed=0, extra_seed_arms=extras)
            print(f"  gate took {time.time()-t1:.1f}s", flush=True)
            arm_payload = gate[arm.name]
            # Stash fit arm for LDVAE_load reuse (not JSON serialisable; strip before write).
            arm_payload["_arm"] = arm
            results[screen]["arms"][arm_tag] = arm_payload

        # Strip the live arm objects before serialising this screen.
        for arm_tag, payload in results[screen]["arms"].items():
            payload.pop("_arm", None)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote: {OUT}", flush=True)


if __name__ == "__main__":
    main()
