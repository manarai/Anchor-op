#!/usr/bin/env python3
"""Backed descriptive response analysis for large normalized Perturb-seq H5ADs.

This script never materializes a full cell-by-gene matrix.  It fits a bounded,
control-derived randomized PCA basis, measures gem_group-matched guide responses
in that program space, and evaluates within-target replicate-guide direction
concordance.  Because it accepts signed normalized/residual data without a
cell-aligned raw/count assay, its results are expressly descriptive: no κ,
calibrated target input, inverse operator, or biological Jacobian is inferred.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONTROL_LABELS = {"non-targeting", "non_targeting", "nt", "control", "unassigned"}


def text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def categorical_values(obs: h5py.Group, column: str, n_cells: int) -> tuple[np.ndarray, list[str]]:
    if column not in obs or not isinstance(obs[column], h5py.Dataset):
        raise ValueError(f"Missing required obs/{column} dataset.")
    category_group = obs.get("__categories")
    if not isinstance(category_group, h5py.Group) or column not in category_group:
        raise ValueError(f"Missing required legacy category vocabulary for obs/{column}.")
    codes = np.asarray(obs[column][:], dtype=np.int64)
    if codes.ndim != 1 or codes.size != n_cells:
        raise ValueError(f"obs/{column} is not a cell-aligned one-dimensional code vector.")
    return codes, [text(value) for value in category_group[column][:]]


def cell_index_hash(obs: h5py.Group) -> str:
    index_name = text(obs.attrs.get("_index", "_index"))
    if index_name not in obs or not isinstance(obs[index_name], h5py.Dataset):
        raise ValueError("H5AD observation index is unavailable.")
    index = obs[index_name]
    digest = hashlib.sha256()
    for start in range(0, int(index.shape[0]), 4096):
        values = [text(value) for value in index[start : start + 4096]]
        digest.update("\x1e".join(values).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def feature_index_hash(var: h5py.Group) -> str:
    index_name = text(var.attrs.get("_index", "_index"))
    if index_name not in var or not isinstance(var[index_name], h5py.Dataset):
        raise ValueError("H5AD feature index is unavailable.")
    index = var[index_name]
    digest = hashlib.sha256()
    for start in range(0, int(index.shape[0]), 4096):
        values = [text(value) for value in index[start : start + 4096]]
        digest.update("\x1e".join(values).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def identify_control(target_categories: list[str]) -> int:
    matches = [i for i, value in enumerate(target_categories) if value.lower() in CONTROL_LABELS]
    if len(matches) != 1:
        raise ValueError(
            "Expected one non-targeting category in obs/gene; found "
            f"{[target_categories[i] for i in matches]!r}."
        )
    return int(matches[0])


def control_sample_and_mean(
    X: h5py.Dataset,
    target_codes: np.ndarray,
    control_code: int,
    max_controls: int,
    chunk_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, int]]:
    """Stream controls, using featurewise finite-value means for transparent imputation."""
    n_cells, n_genes = map(int, X.shape)
    control_rows = np.flatnonzero(target_codes == control_code)
    if control_rows.size < 3:
        raise ValueError("At least three non-targeting cells are required.")
    rng = np.random.default_rng(seed)
    sample_rows = np.sort(
        rng.choice(control_rows, size=min(int(max_controls), int(control_rows.size)), replace=False)
    )
    sample = np.empty((sample_rows.size, n_genes), dtype=np.float32)
    total = np.zeros(n_genes, dtype=np.float64)
    finite_counts = np.zeros(n_genes, dtype=np.int64)
    control_nonfinite_values = 0
    written = 0
    seen = 0
    for start in range(0, n_cells, chunk_rows):
        stop = min(start + chunk_rows, n_cells)
        block = np.asarray(X[start:stop, :], dtype=np.float32)
        local_controls = target_codes[start:stop] == control_code
        if np.any(local_controls):
            control_block = block[local_controls]
            finite = np.isfinite(control_block)
            total += np.where(finite, control_block, 0.0).sum(axis=0, dtype=np.float64)
            finite_counts += finite.sum(axis=0, dtype=np.int64)
            control_nonfinite_values += int((~finite).sum())
            seen += int(local_controls.sum())
        left = int(np.searchsorted(sample_rows, start, side="left"))
        right = int(np.searchsorted(sample_rows, stop, side="left"))
        if right > left:
            sample[written : written + right - left] = block[sample_rows[left:right] - start]
            written += right - left
        del block
    if written != len(sample_rows) or seen != len(control_rows):
        raise RuntimeError("Control streaming failed to recover the expected rows.")
    if np.any(finite_counts == 0):
        raise ValueError(
            f"{int(np.sum(finite_counts == 0))} features have no finite non-targeting control values; "
            "a control-derived basis cannot be defined safely."
        )
    control_mean = (total / finite_counts).astype(np.float32)
    sample_nonfinite = ~np.isfinite(sample)
    sample_nonfinite_values = int(sample_nonfinite.sum())
    if sample_nonfinite_values:
        sample[sample_nonfinite] = np.broadcast_to(control_mean, sample.shape)[sample_nonfinite]
    return sample, control_mean, int(seen), {
        "control_nonfinite_values": control_nonfinite_values,
        "control_sample_nonfinite_values_imputed": sample_nonfinite_values,
    }


def control_pca(sample: np.ndarray, control_mean: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_components >= min(sample.shape):
        raise ValueError("PCA dimension is too large for the sampled controls.")
    centered = sample - control_mean[None, :]
    total_ss = float(np.sum(centered * centered, dtype=np.float64))
    if total_ss <= 0:
        raise ValueError("Control sample has no variation after centering.")
    _, singular_values, vt = randomized_svd(
        centered,
        n_components=n_components,
        n_oversamples=10,
        n_iter=5,
        random_state=seed,
        power_iteration_normalizer="auto",
    )
    explained = np.asarray(singular_values, dtype=float) ** 2 / total_ss
    return np.asarray(vt, dtype=np.float32), np.asarray(singular_values, dtype=float), explained


def guide_target_mapping(guide_codes: np.ndarray, target_codes: np.ndarray, control_code: int) -> dict[int, int | None]:
    mapping: dict[int, set[int]] = defaultdict(set)
    valid = (guide_codes >= 0) & (target_codes >= 0) & (target_codes != control_code)
    for guide, target in np.unique(np.column_stack((guide_codes[valid], target_codes[valid])), axis=0):
        mapping[int(guide)].add(int(target))
    return {guide: next(iter(targets)) if len(targets) == 1 else None for guide, targets in mapping.items()}


def stream_group_statistics(
    X: h5py.Dataset,
    guide_codes: np.ndarray,
    batch_codes: np.ndarray,
    target_codes: np.ndarray,
    control_code: int,
    components: np.ndarray,
    control_mean: np.ndarray,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Project rows by chunk and accumulate guide/batch and control/batch statistics."""
    n_cells = int(X.shape[0])
    usable_batches = batch_codes >= 0
    batch_labels, batch_inverse = np.unique(batch_codes[usable_batches], return_inverse=True)
    # Preserve -1 as an invalid batch marker, never indexing a real group.
    full_batch_inverse = np.full(n_cells, -1, dtype=np.int64)
    full_batch_inverse[usable_batches] = batch_inverse
    n_guides = int(np.max(guide_codes[guide_codes >= 0])) + 1
    n_batches = int(len(batch_labels))
    d = int(components.shape[0])
    guide_counts = np.zeros((n_guides, n_batches), dtype=np.int64)
    guide_sums = np.zeros((n_guides, n_batches, d), dtype=np.float64)
    control_counts = np.zeros(n_batches, dtype=np.int64)
    control_sums = np.zeros((n_batches, d), dtype=np.float64)
    all_cell_nonfinite_values = 0
    cells_with_nonfinite_values = 0

    for start in range(0, n_cells, chunk_rows):
        stop = min(start + chunk_rows, n_cells)
        block = np.asarray(X[start:stop, :], dtype=np.float32)
        nonfinite = ~np.isfinite(block)
        if np.any(nonfinite):
            all_cell_nonfinite_values += int(nonfinite.sum())
            cells_with_nonfinite_values += int(np.any(nonfinite, axis=1).sum())
            block[nonfinite] = np.broadcast_to(control_mean, block.shape)[nonfinite]
        z = block @ components.T
        guides = guide_codes[start:stop]
        batches = full_batch_inverse[start:stop]
        valid = (guides >= 0) & (batches >= 0)
        if np.any(valid):
            np.add.at(guide_counts, (guides[valid], batches[valid]), 1)
            for component in range(d):
                np.add.at(guide_sums[:, :, component], (guides[valid], batches[valid]), z[valid, component])
        controls = valid & (target_codes[start:stop] == control_code)
        if np.any(controls):
            np.add.at(control_counts, batches[controls], 1)
            for component in range(d):
                np.add.at(control_sums[:, component], batches[controls], z[controls, component])
        del block, z
    return guide_counts, guide_sums, control_counts, control_sums, batch_labels, {
        "all_cell_nonfinite_values_imputed": all_cell_nonfinite_values,
        "cells_with_nonfinite_values_imputed": cells_with_nonfinite_values,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / scale) if scale > np.finfo(float).eps else float("nan")


def bootstrap_ci(values: np.ndarray, seed: int, n_draws: int = 2000) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None, None
    rng = np.random.default_rng(seed)
    medians = np.empty(n_draws, dtype=float)
    for draw in range(n_draws):
        medians[draw] = np.median(rng.choice(values, size=values.size, replace=True))
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def analyze(path: Path, output_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    with h5py.File(path, "r") as handle:
        X = handle.get("X")
        obs = handle.get("obs")
        var = handle.get("var")
        if not isinstance(X, h5py.Dataset) or not isinstance(obs, h5py.Group) or not isinstance(var, h5py.Group):
            raise ValueError(f"{path.name} is not the expected dense legacy H5AD structure.")
        n_cells, n_genes = map(int, X.shape)
        guide_codes, guide_categories = categorical_values(obs, "sgID_AB", n_cells)
        target_codes, target_categories = categorical_values(obs, "gene", n_cells)
        gene_id_codes, gene_id_categories = categorical_values(obs, "gene_id", n_cells)
        if "gem_group" not in obs:
            raise ValueError("obs/gem_group is required for matched-control analysis.")
        batch_codes = np.asarray(obs["gem_group"][:], dtype=np.int64)
        if batch_codes.ndim != 1 or batch_codes.size != n_cells:
            raise ValueError("obs/gem_group is not cell-aligned.")
        control_code = identify_control(target_categories)
        obs_hash = cell_index_hash(obs)
        var_hash = feature_index_hash(var)
        sample, control_mean, n_controls, control_nonfinite = control_sample_and_mean(
            X, target_codes, control_code, args.max_controls, args.chunk_rows, args.seed
        )
        components, singular_values, explained = control_pca(sample, control_mean, args.n_components, args.seed)
        del sample
        gc.collect()
        guide_counts, guide_sums, control_counts, control_sums, batch_labels, all_cell_nonfinite = stream_group_statistics(
            X, guide_codes, batch_codes, target_codes, control_code, components, control_mean, args.chunk_rows
        )

    mapping = guide_target_mapping(guide_codes, target_codes, control_code)
    control_means = np.divide(
        control_sums,
        control_counts[:, None],
        out=np.full_like(control_sums, np.nan),
        where=control_counts[:, None] > 0,
    )
    guide_rows: list[dict[str, Any]] = []
    dropped: dict[str, str] = {}
    for guide_code, target_code in sorted(mapping.items()):
        guide_name = guide_categories[guide_code]
        if target_code is None:
            dropped[guide_name] = "multiple_target_labels"
            continue
        batch_counts = guide_counts[guide_code]
        n_guide_cells = int(batch_counts.sum())
        if n_guide_cells < args.min_cells:
            dropped[guide_name] = f"fewer_than_{args.min_cells}_cells"
            continue
        observed_batches = batch_counts > 0
        if np.any(control_counts[observed_batches] == 0):
            dropped[guide_name] = "missing_matched_non_targeting_control_in_batch"
            continue
        centered_sum = guide_sums[guide_code] - batch_counts[:, None] * control_means
        response = np.nansum(centered_sum, axis=0) / n_guide_cells
        if not np.isfinite(response).all():
            dropped[guide_name] = "nonfinite_response"
            continue
        target_name = target_categories[target_code]
        gene_id_values = np.unique(gene_id_codes[(guide_codes == guide_code) & (target_codes == target_code)])
        gene_ids = [gene_id_categories[int(code)] for code in gene_id_values if code >= 0]
        row = {
            "guide": guide_name,
            "target": target_name,
            "target_gene_id": gene_ids[0] if len(gene_ids) == 1 else "|".join(gene_ids),
            "n_cells": n_guide_cells,
            "n_batches": int(observed_batches.sum()),
            "response_norm": float(np.linalg.norm(response)),
        }
        row.update({f"pc_{i + 1}": float(value) for i, value in enumerate(response)})
        guide_rows.append(row)

    guide_table = pd.DataFrame(guide_rows)
    if guide_table.empty:
        raise RuntimeError("No guide responses survived the declared quality filters.")
    guide_table = guide_table.sort_values(["target", "guide"]).reset_index(drop=True)
    pc_cols = [col for col in guide_table if col.startswith("pc_")]
    target_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for target, group in guide_table.groupby("target", sort=True):
        vectors = group[pc_cols].to_numpy(dtype=float)
        names = group["guide"].astype(str).to_list()
        pairwise: list[float] = []
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                value = cosine(vectors[left], vectors[right])
                pairwise.append(value)
                pair_rows.append({"target": target, "guide_left": names[left], "guide_right": names[right], "direction_cosine": value})
        consensus = np.median(vectors, axis=0)
        target_rows.append({
            "target": target,
            "target_gene_id": "|".join(sorted(group["target_gene_id"].astype(str).unique())),
            "n_guides": int(len(group)),
            "total_cells": int(group["n_cells"].sum()),
            "mean_pairwise_cosine": float(np.nanmean(pairwise)) if pairwise else float("nan"),
            "median_pairwise_cosine": float(np.nanmedian(pairwise)) if pairwise else float("nan"),
            "min_pairwise_cosine": float(np.nanmin(pairwise)) if pairwise else float("nan"),
            "median_guide_response_norm": float(np.median(group["response_norm"])),
            "consensus_response_norm": float(np.linalg.norm(consensus)),
            **{f"consensus_pc_{i + 1}": float(value) for i, value in enumerate(consensus)},
        })
    target_table = pd.DataFrame(target_rows).sort_values("target").reset_index(drop=True)
    pair_table = pd.DataFrame(pair_rows)
    if not pair_table.empty:
        pair_table = pair_table.sort_values(["target", "guide_left", "guide_right"]).reset_index(drop=True)

    at_least_2 = target_table.loc[target_table.n_guides >= 2, "mean_pairwise_cosine"].to_numpy(dtype=float)
    at_least_3 = target_table.loc[target_table.n_guides >= 3, "mean_pairwise_cosine"].to_numpy(dtype=float)
    strict_finite = at_least_3[np.isfinite(at_least_3)]
    ci_low, ci_high = bootstrap_ci(strict_finite, args.seed + 1000)
    stem = path.name.removesuffix(".h5ad")
    output_dir.mkdir(parents=True, exist_ok=True)
    guide_table.to_csv(output_dir / f"{stem}_guide_responses.csv", index=False)
    target_table.to_csv(output_dir / f"{stem}_target_response_atlas.csv", index=False)
    pair_table.to_csv(output_dir / f"{stem}_within_target_pairs.csv", index=False)
    np.savez_compressed(
        output_dir / f"{stem}_control_pca.npz",
        components=components,
        control_mean=control_mean,
        singular_values=singular_values,
        explained_variance_ratio=explained,
        obs_index_sha256=obs_hash,
        var_index_sha256=var_hash,
    )
    summary: dict[str, Any] = {
        "dataset": stem,
        "source_path": str(path),
        "source_file_size_bytes": int(path.stat().st_size),
        "analysis_kind": "descriptive_unscaled_program_response",
        "matrix_shape": [n_cells, n_genes],
        "n_non_targeting_controls": n_controls,
        "n_components": args.n_components,
        "control_sample_size": min(args.max_controls, n_controls),
        "control_pca_explained_variance_ratio": explained.tolist(),
        "control_pca_explained_variance_ratio_sum": float(explained.sum()),
        "n_batches_with_controls": int(np.sum(control_counts > 0)),
        "nonfinite_input_handling": {
            "method": "featurewise_global_non_targeting_control_mean_imputation_before_program_projection",
            **control_nonfinite,
            **all_cell_nonfinite,
        },
        "n_retained_guides": int(len(guide_table)),
        "n_dropped_guides": int(len(dropped)),
        "dropped_guide_reasons": {reason: int(sum(value == reason for value in dropped.values())) for reason in sorted(set(dropped.values()))},
        "n_retained_targets": int(len(target_table)),
        "n_targets_at_least_2_guides": int(np.sum(target_table.n_guides >= 2)),
        "n_targets_at_least_3_guides": int(np.sum(target_table.n_guides >= 3)),
        "median_target_mean_pairwise_cosine_at_least_2_guides": float(np.nanmedian(at_least_2)) if at_least_2.size else None,
        "median_target_mean_pairwise_cosine_at_least_3_guides": float(np.nanmedian(strict_finite)) if strict_finite.size else None,
        "bootstrap_95pct_ci_median_cosine_at_least_3_guides": [ci_low, ci_high],
        "fraction_targets_negative_mean_cosine_at_least_3_guides": float(np.mean(strict_finite < 0)) if strict_finite.size else None,
        "obs_index_sha256": obs_hash,
        "var_index_sha256": var_hash,
        "parameters": {
            "pca_basis": "control-derived randomized PCA",
            "control_matching": "gem_group matched non-targeting means",
            "guide_column": "sgID_AB",
            "target_column": "gene",
            "target_feature_identifier_column": "gene_id",
            "chunk_rows": args.chunk_rows,
            "minimum_cells_per_guide": args.min_cells,
            "seed": args.seed,
        },
        "not_supported_without_paired_raw_count_assay": [
            "quantitative_knockdown_efficiency_kappa",
            "kappa_calibrated_target_inputs",
            "guide_efficiency_dose_response",
            "biological_operator_or_jacobian",
        ],
    }
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary, target_table


def make_figure(tables: list[tuple[str, pd.DataFrame]], destination: Path) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4.8), sharey=True)
    if len(tables) == 1:
        axes = [axes]
    for ax, (name, table) in zip(axes, tables, strict=True):
        all_values = table.loc[table.n_guides >= 2, "mean_pairwise_cosine"].dropna().to_numpy()
        strict_values = table.loc[table.n_guides >= 3, "mean_pairwise_cosine"].dropna().to_numpy()
        bins = np.linspace(-1, 1, 41)
        if all_values.size:
            ax.hist(all_values, bins=bins, color="#708090", alpha=0.78, label=f"≥2 guides (n={all_values.size})")
        if strict_values.size:
            ax.hist(strict_values, bins=bins, histtype="step", linewidth=2.2, color="#c0392b", label=f"≥3 guides (n={strict_values.size})")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(0.5, color="#34495e", linestyle="--", linewidth=1.0, label="0.5 reference")
        ax.set_title(name)
        ax.set_xlabel("Mean within-target guide-direction cosine")
        ax.set_xlim(-1, 1)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Targets")
    fig.suptitle("Descriptive replicate-guide response concordance", y=1.02)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for item in summaries:
        ci = item["bootstrap_95pct_ci_median_cosine_at_least_3_guides"]
        ci_text = "not estimable" if ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        median = item["median_target_mean_pairwise_cosine_at_least_3_guides"]
        rows.append(
            f"| {item['dataset']} | {item['matrix_shape'][0]:,} × {item['matrix_shape'][1]:,} | "
            f"{item['n_non_targeting_controls']:,} | {item['n_retained_guides']:,} | "
            f"{item['n_targets_at_least_2_guides']:,} | {item['n_targets_at_least_3_guides']:,} | "
            f"{'not estimable' if median is None else f'{median:.3f}'} | {ci_text} |"
        )
    content = """# Backed Exploratory Perturb-seq Response Analysis

## Scope

The source H5AD files contain signed normalized/residual expression values. This workflow reads them in backed row chunks, uses a bounded control-derived randomized PCA coordinate system, and subtracts `gem_group`-matched non-targeting means. It produces a reproducible descriptive response atlas; it does **not** estimate κ, calibrate perturbation dose, infer a linear operator, or recover a biological Jacobian.

| Dataset | Cells × genes | Non-targeting controls | Retained guides | Targets with ≥2 guides | Targets with ≥3 guides | Median mean cosine, ≥3 guides | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

## Interpretation

Within-target guide-direction cosine measures whether independent guide labels for the same target have consistent program-space response directions. It is not a knockdown dose-response test without paired raw/count-like target expression. Low or negative concordance is evidence against treating guide labels as scalar doses of a shared target perturbation; in that case, the target-response atlas is the correct output and an inverse-operator claim should be withheld.

## Files

For every dataset: `*_guide_responses.csv`, `*_target_response_atlas.csv`, `*_within_target_pairs.csv`, `*_control_pca.npz`, and `*_summary.json`. The cross-dataset visual summary is `within_target_concordance.png`.

## Requirement for calibrated Anchor-op inference

Provide a raw/count-like H5AD for each normalized response H5AD, preserving exact cell barcode order. Only then can a paired workflow use raw data to estimate κ, test within-target guide efficiency-response scaling, and conditionally estimate a regularized action after held-out and identifiability diagnostics pass.
"""
    (output_dir / "README.md").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--max-controls", type=int, default=6000)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.n_components < 2 or args.max_controls <= args.n_components or args.min_cells < 2 or args.chunk_rows < 1:
        parser.error("Invalid component, control-sample, minimum-cell, or chunk-row setting.")
    summaries: list[dict[str, Any]] = []
    tables: list[tuple[str, pd.DataFrame]] = []
    for path in args.paths:
        summary, target_table = analyze(path, args.output_dir, args)
        summaries.append(summary)
        tables.append((summary["dataset"], target_table))
        gc.collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    make_figure(tables, args.output_dir / "within_target_concordance.png")
    write_readme(args.output_dir, summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
