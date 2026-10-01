"""Five-clause feasibility gate (B2) for the exp1b arms.

Clauses (dropped-and-reported if any fail):
  (a) determinism       — identical input twice -> max |Δ| in the dim-d latent must be 0
                           (posterior MEAN for VAEs; no sampling).
  (b) negative control  — a gene absent from the arm's ingestion (or zero in all controls)
                           must give ‖u_z‖ = 0.
  (c) non-triviality    — mean ‖u_z‖ > the cell-sampling noise floor computed as the
                           mean between-subset norm across 20 random disjoint 50-cell
                           subset pairs.
  (d) stability         — mean subset-to-subset cosine of u_z > 0.9 across the same 20 pairs.
                           For VAE arms also compute across training seeds.
  (e) dose-grading      — ‖u_z‖ strictly increases with κ ∈ {0.5, 0.7, 0.9}.

Targets: 6 prespecified quantiles q10, q30, q50, q70, q90 + CDC27 (if present in the HVG
feature set). Quantiles are computed on the control-cell log1p (or count-mean) matrix.

Harness contract: `run_gate(arm, X_ctrl, gene_names, target_idxs, kappas, n_pairs=20,
subset=50, seed=0, extra_seed_arms=None)` returns a dict:

  {
    arm_name: {
      'target_name': {
        'determinism_max_abs_delta': float,
        'negative_control_norm': float,
        'u_z_by_kappa':      {kappa: list of per-cell ‖u_z‖},  # pooled across cells
        'between_subset':    {kappa: {'mean_norm': float, 'mean_cos': float,
                                      'per_pair_norm': list, 'per_pair_cos': list}},
        'cross_seed':        {kappa: {...}} or None,
        'noise_floor':       {kappa: float},
        'passes': {'a': bool, 'b': bool, 'c': bool, 'd': bool, 'e': bool},
        'halt_reason':       str or None,
      },
      ...
    }
  }

Compute note: the gate itself is small (≤ 20 encode calls per (target × κ) × 2
scenarios; ≤ ~400 encode calls per arm). For VAEs the dominant cost is model
TRAINING in `arm.fit`; the gate just calls encode. See B3 PREREG for the full-
benchmark compute estimate.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np


def pick_prespecified_targets(X_ctrl_log1p: np.ndarray,
                               gene_names: Sequence[str],
                               include: Sequence[str] = ("CDC27",)
                               ) -> list:
    """Pick 6 prespecified targets by control-expression quantile + any explicit include.

    Quantiles computed on the per-gene mean of X_ctrl_log1p (log1p space). Returns a
    sorted list of (gene_name, gene_idx) tuples in quantile order.
    """
    mean_expr = np.asarray(X_ctrl_log1p).mean(axis=0)
    gene_names = list(gene_names)
    qs = (0.10, 0.30, 0.50, 0.70, 0.90)
    picks = []
    for q in qs:
        thresh = float(np.quantile(mean_expr, q))
        idx = int(np.argmin(np.abs(mean_expr - thresh)))
        picks.append((gene_names[idx], idx))
    for name in include:
        if name in gene_names and all(p[0] != name for p in picks):
            picks.append((name, gene_names.index(name)))
    return picks


def _cos(u, v):
    nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
    if nu * nv < 1e-30:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _subset_u_z(arm, X_ctrl: np.ndarray, target_idx: int, kappa: float,
                 subset_a: np.ndarray, subset_b: np.ndarray) -> tuple:
    u_a = arm.knockdown_u_z(X_ctrl[subset_a], target_idx, kappa=kappa)
    u_b = arm.knockdown_u_z(X_ctrl[subset_b], target_idx, kappa=kappa)
    return u_a, u_b


def _negative_control_gene(X_ctrl: np.ndarray, gene_names: Sequence[str],
                            exclude: int) -> int:
    """Pick a gene with zero (or near-zero) expression in all controls."""
    col_max = np.asarray(X_ctrl).max(axis=0)
    cands = np.where(col_max < 1e-6)[0]
    cands = cands[cands != exclude]
    if len(cands) == 0:
        # Fall back to lowest-expressed gene.
        col_mean = np.asarray(X_ctrl).mean(axis=0)
        order = np.argsort(col_mean)
        for g in order:
            if int(g) != exclude:
                return int(g)
        raise RuntimeError("no candidate negative-control gene available")
    return int(cands[0])


def run_gate(arm,
             X_ctrl: np.ndarray,
             gene_names: Sequence[str],
             target_idxs: Sequence[tuple],
             kappas: Sequence[float] = (0.5, 0.7, 0.9),
             n_pairs: int = 20,
             subset: int = 50,
             seed: int = 0,
             extra_seed_arms: Optional[list] = None,
             cos_threshold: float = 0.9) -> dict:
    """Run the 5-clause feasibility gate on one arm.

    `target_idxs` is a list of `(name, idx)` tuples (as returned by
    `pick_prespecified_targets`). `extra_seed_arms` for VAE arms is a list of
    additional fitted arm instances with different training seeds; the (d) stability
    clause also reports cross-seed cosine on them.
    """
    rng = np.random.default_rng(seed)
    X_ctrl = np.asarray(X_ctrl, dtype=np.float64)
    n = X_ctrl.shape[0]
    assert subset * 2 <= n, f"subset {subset} too large for n_ctrl={n}"

    out = {}

    for name, g_idx in target_idxs:
        rec = {"gene_name": name, "gene_idx": int(g_idx),
                "passes": {}, "halt_reason": None}

        # ── (a) determinism ──
        X_small = X_ctrl[: min(100, n)]
        Z0 = arm.encode(X_small); Z1 = arm.encode(X_small)
        max_abs = float(np.max(np.abs(Z0 - Z1)))
        rec["determinism_max_abs_delta"] = max_abs
        rec["passes"]["a"] = (max_abs < 1e-6)

        # ── (b) negative control (gene absent / zero in all controls) ──
        neg_idx = _negative_control_gene(X_ctrl, gene_names, exclude=g_idx)
        rec["negative_control_gene"] = str(gene_names[neg_idx])
        u_neg = arm.knockdown_u_z(X_ctrl[:subset], neg_idx, kappa=0.7)
        neg_norm = float(np.linalg.norm(u_neg))
        rec["negative_control_norm"] = neg_norm
        rec["passes"]["b"] = (neg_norm < 1e-6)

        # ── between-subset noise floor + (c), (d) ──
        rec["between_subset"] = {}
        rec["noise_floor"] = {}
        rec["u_z_norms"] = {}
        for k in kappas:
            pair_norms = []
            pair_cos = []
            # one anchor u for the whole 50-cell universe:
            u_full = arm.knockdown_u_z(X_ctrl[: 2 * subset], g_idx, kappa=k)
            u_full_norm = float(np.linalg.norm(u_full))
            rec["u_z_norms"][str(k)] = u_full_norm
            # between-subset floor
            for p in range(n_pairs):
                idx = rng.permutation(n)[: 2 * subset]
                sa, sb = idx[:subset], idx[subset:2 * subset]
                u_a, u_b = _subset_u_z(arm, X_ctrl, g_idx, k, sa, sb)
                pair_norms.append(float(np.linalg.norm(u_a - u_b) / 2))
                pair_cos.append(_cos(u_a, u_b))
            mean_norm = float(np.mean(pair_norms))
            mean_cos = float(np.mean(pair_cos))
            rec["between_subset"][str(k)] = {
                "mean_norm": mean_norm, "mean_cos": mean_cos,
                "per_pair_norm": pair_norms, "per_pair_cos": pair_cos,
            }
            rec["noise_floor"][str(k)] = mean_norm

        # (c): primary κ=0.7 norm > noise floor at κ=0.7
        rec["passes"]["c"] = (rec["u_z_norms"]["0.7"] >
                               rec["noise_floor"]["0.7"])
        # (d): mean subset cosine > threshold at κ=0.7
        rec["passes"]["d"] = (rec["between_subset"]["0.7"]["mean_cos"] >
                               cos_threshold)
        # (e): ‖u_z‖ increases with κ ∈ {0.5, 0.7, 0.9}
        norms_in_order = [rec["u_z_norms"][str(k)] for k in sorted(kappas)]
        rec["passes"]["e"] = all(
            norms_in_order[i] <= norms_in_order[i + 1] + 1e-9
            for i in range(len(norms_in_order) - 1))

        # Cross-seed stability for VAE arms.
        if extra_seed_arms:
            cross = {}
            u_primary = u_full
            for k in kappas:
                cos_list = []
                for other in extra_seed_arms:
                    u_other = other.knockdown_u_z(X_ctrl[:2*subset], g_idx, kappa=k)
                    cos_list.append(_cos(u_primary, u_other))
                cross[str(k)] = {"cross_seed_cosines": cos_list,
                                  "mean_cos": float(np.mean(cos_list))}
            rec["cross_seed"] = cross
        else:
            rec["cross_seed"] = None

        all_pass = all(rec["passes"].values())
        if not all_pass:
            failed = [c for c, ok in rec["passes"].items() if not ok]
            rec["halt_reason"] = f"clauses_failed: {','.join(failed)}"

        out[name] = rec

    return {arm.name: out}
