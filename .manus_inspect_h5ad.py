#!/usr/bin/env python3
"""Read-only structural inspection for large .h5ad files.

Avoids loading an expression matrix into memory. It emits a compact JSON record
suitable for deciding whether an experiment can run the Anchor-op paired-assay
workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def dataset_summary(node: h5py.Dataset) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": "dataset",
        "shape": list(node.shape),
        "dtype": str(node.dtype),
        "chunks": list(node.chunks) if node.chunks else None,
        "compression": node.compression,
    }
    if node.ndim == 2 and node.shape[0] and node.shape[1]:
        row_count = min(8, node.shape[0])
        col_count = min(512, node.shape[1])
        # Sparse positional blocks across the matrix; bounded at 8*8*512 values.
        blocks: list[np.ndarray] = []
        row_starts = np.linspace(0, max(0, node.shape[0] - row_count), num=min(8, node.shape[0]), dtype=int)
        for start in np.unique(row_starts):
            block = np.asarray(node[start : start + row_count, :col_count], dtype=float)
            blocks.append(block.ravel())
        values = np.concatenate(blocks) if blocks else np.empty(0, dtype=float)
        finite = values[np.isfinite(values)]
        summary["bounded_sample"] = {
            "n_values": int(values.size),
            "finite_fraction": float(np.mean(np.isfinite(values))) if values.size else None,
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
            "negative_fraction": float(np.mean(finite < 0)) if finite.size else None,
            "zero_fraction": float(np.mean(finite == 0)) if finite.size else None,
        }
    return summary


def node_summary(node: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    if isinstance(node, h5py.Dataset):
        return dataset_summary(node)
    result: dict[str, Any] = {"kind": "group", "keys": sorted(map(str, node.keys()))}
    if "shape" in node.attrs:
        result["shape"] = [int(x) for x in node.attrs["shape"]]
    if "data" in node and isinstance(node["data"], h5py.Dataset):
        result["data"] = dataset_summary(node["data"])
    return result


def stream_hash(dataset: h5py.Dataset, chunk: int = 4096) -> dict[str, Any]:
    """Hash a one-dimensional index without retaining it in memory."""
    digest = hashlib.sha256()
    n = int(dataset.shape[0])
    first: list[str] = []
    last: list[str] = []
    for start in range(0, n, chunk):
        values = dataset[start : min(start + chunk, n)]
        texts = [decode(value) for value in values]
        if len(first) < 5:
            first.extend(texts[: 5 - len(first)])
        last = (last + texts)[-5:]
        digest.update("\x1e".join(texts).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\x1f")
    return {"n": n, "sha256": digest.hexdigest(), "first": first, "last": last}


def column_summary(obs: h5py.Group, name: str) -> dict[str, Any]:
    node = obs[name]
    if isinstance(node, h5py.Dataset):
        summary = dataset_summary(node)
        # Older AnnData h5ads store categorical codes directly at obs/<column>
        # and the category vocabulary under obs/__categories/<column>.
        legacy_categories = obs.get("__categories")
        if isinstance(legacy_categories, h5py.Group) and name in legacy_categories:
            categories = legacy_categories[name]
            if isinstance(categories, h5py.Dataset):
                category_text = [decode(value) for value in categories[:]]
                codes = np.asarray(node[:])
                valid = codes[(codes >= 0) & (codes < len(category_text))]
                counts = np.bincount(valid.astype(int), minlength=len(category_text))
                nonzero = np.flatnonzero(counts)
                order = sorted(nonzero, key=lambda index: (-int(counts[index]), category_text[int(index)]))
                summary["legacy_categorical"] = {
                    "n_categories": int(len(category_text)),
                    "n_observed_categories": int(len(nonzero)),
                    "missing_count": int(np.sum(codes < 0)) if np.issubdtype(codes.dtype, np.signedinteger) else 0,
                    "category_sample": category_text[:10],
                    "most_frequent": [
                        {"value": category_text[int(index)], "n_cells": int(counts[index])}
                        for index in order[:10]
                    ],
                    "control_like_categories": [
                        value for value in category_text
                        if value.lower() in {"non-targeting", "non_targeting", "nt", "control", "unassigned"}
                    ],
                }
        return summary
    summary: dict[str, Any] = {"kind": "group", "keys": sorted(map(str, node.keys()))}
    if "categories" in node:
        categories = node["categories"]
        sample = [decode(x) for x in categories[: min(10, categories.shape[0])]]
        summary["categories"] = {"n": int(categories.shape[0]), "sample": sample}
    if "codes" in node:
        codes = node["codes"]
        # Categorical codes are small metadata; read all for useful cardinality.
        values = np.asarray(codes[:])
        summary["codes"] = {
            "n": int(values.size),
            "n_unique_including_missing": int(np.unique(values).size),
            "missing_count": int(np.sum(values < 0)) if np.issubdtype(values.dtype, np.signedinteger) else 0,
        }
    return summary


def inspect(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        result: dict[str, Any] = {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "root_encoding": {
                str(key): decode(value)
                for key, value in handle.attrs.items()
                if key in {"encoding-type", "encoding-version"}
            },
            "root_keys": sorted(map(str, handle.keys())),
        }
        if "X" in handle:
            result["X"] = node_summary(handle["X"])
        for axis in ("obs", "var"):
            if axis not in handle or not isinstance(handle[axis], h5py.Group):
                continue
            group = handle[axis]
            index_name = decode(group.attrs.get("_index", "_index"))
            axis_result: dict[str, Any] = {
                "n_columns": len(group.keys()),
                "columns": sorted(map(str, group.keys())),
                "index_name": index_name,
            }
            if index_name in group and isinstance(group[index_name], h5py.Dataset):
                axis_result["index"] = stream_hash(group[index_name])
            if axis == "obs":
                preferred_names = {
                    "gene", "target_gene", "target", "gene_symbol", "gene_id", "gene_transcript",
                    "guide_identity", "sgid_ab", "sgid", "sgrna", "guide", "protospacer",
                    "gemgroup", "gem_group", "batch", "lane", "condition", "perturbation",
                }
                preferred = [
                    key for key in group.keys()
                    if key.lower() in preferred_names
                    or any(token in key.lower() for token in ("guide", "target", "gene", "batch", "perturb", "condition"))
                ]
                axis_result["candidate_analysis_columns"] = {
                    str(name): column_summary(group, str(name)) for name in sorted(preferred)
                }
            result[axis] = axis_result
        if "uns" in handle and isinstance(handle["uns"], h5py.Group):
            result["uns_keys"] = sorted(map(str, handle["uns"].keys()))
    return result


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: .manus_inspect_h5ad.py FILE [FILE ...]")
    records = [inspect(Path(value).expanduser()) for value in sys.argv[1:]]
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
