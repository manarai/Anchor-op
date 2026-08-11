"""Loader for Replogle 2022-style processed Perturb-seq h5ads.

The Replogle group's processed data (`gwps.wi.mit.edu`) is distributed as
AnnData h5ad files with a documented schema. This module wraps
`anndata.read_h5ad` with:
- schema auto-detection for the common column-name variants
  (``gene`` vs ``target_gene``; ``guide_identity`` vs ``sgID``, etc.)
- a helper that returns an anchor-op-ready AnnData whose ``obs`` has the exact
  ``guide`` / ``target_gene`` columns that ``measure_operator`` expects, and
  whose non-targeting cells carry the canonical NT label.

Only used when the user opts in (via the ``01b_measure_k562_replogle.ipynb``
notebook or a direct call). The core anchor-op API accepts any AnnData-like
input, so this is convenience, not requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import AnchorOpError

if TYPE_CHECKING:
    from anndata import AnnData


# Known Replogle schema variants — the first key found in obs wins.
_TARGET_COL_CANDIDATES = ("gene", "target_gene", "target", "gene_symbol")
_GUIDE_COL_CANDIDATES = ("guide_identity", "sgID_AB", "sgID", "sgRNA", "guide", "protospacer")
_BATCH_COL_CANDIDATES = ("gemgroup", "gem_group", "batch", "lane")

# Non-targeting labels observed across Replogle releases.
_NT_LABEL_CANDIDATES = (
    "non-targeting",
    "Non-Targeting",
    "non_targeting",
    "NT",
    "control",
    "unassigned",
)


def _pick(obs_cols, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate that appears in obs_cols, else None."""
    for c in candidates:
        if c in obs_cols:
            return c
    return None


def _detect_nt_label(obs_series) -> str:
    """Return the observed NT label in a target column, matching known aliases."""
    values = set(map(str, obs_series.unique()))
    for candidate in _NT_LABEL_CANDIDATES:
        if candidate in values:
            return candidate
    # Case-insensitive fallback
    lower_map = {v.lower(): v for v in values}
    for candidate in _NT_LABEL_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise AnchorOpError(
        f"No non-targeting label detected among {sorted(values)[:8]}; "
        f"tried {_NT_LABEL_CANDIDATES}. Pass control_label explicitly."
    )


def _looks_like_ensembl(var_names) -> bool:
    """Return True if the first few var_names look like Ensembl gene IDs."""
    sample = [str(v) for v in list(var_names)[:5]]
    return sum(1 for v in sample if v.startswith("ENSG") and len(v) >= 15) >= 3


def load_replogle_h5ad(
    path: str,
    *,
    target_col: str | None = None,
    guide_col: str | None = None,
    batch_col: str | None = None,
    control_label: str | None = None,
    canonical_nt_label: str = "non-targeting",
    canonical_guide_key: str = "guide",
    canonical_target_key: str = "target_gene",
    backed: str | None = None,
) -> "AnnData":
    """Load a Replogle processed h5ad and normalize its schema for anchor-op.

    Parameters
    ----------
    path
        Path to the h5ad file.
    target_col
        Column in ``adata.obs`` naming each cell's target gene. If ``None``,
        the loader auto-detects among ``gene``, ``target_gene``, ``target``,
        ``gene_symbol``.
    guide_col
        Column naming each cell's sgRNA. If ``None``, auto-detects among
        ``guide_identity``, ``sgID``, ``sgRNA``, ``guide``, ``protospacer``.
    batch_col
        Optional batch column. If ``None``, auto-detects among ``gemgroup``,
        ``gem_group``, ``batch``, ``lane``.
    control_label
        The label used in ``target_col`` for non-targeting cells. If ``None``,
        auto-detects among ``non-targeting``, ``Non-Targeting``, ``NT``, etc.
    canonical_nt_label, canonical_guide_key, canonical_target_key
        The labels the returned AnnData will present. Defaults match what the
        rest of anchor-op assumes (``measure_operator(..., control_label="non-targeting",
        guide_key="guide", target_key="target_gene")``).
    backed
        Optional AnnData backed mode (for example ``"r"``) used to keep a
        multi-gigabyte H5AD expression matrix on disk. Metadata normalization
        remains in memory, but ``adata.X`` is not materialized.

    Returns
    -------
    AnnData with:
      * ``obs[canonical_guide_key]``: sgRNA id for perturbed cells, ``canonical_nt_label`` for controls.
      * ``obs[canonical_target_key]``: target gene symbol for perturbed cells, empty string for controls.
      * ``obs["gemgroup"]``: preserved batch column if present.
      * everything else from the source file is left intact.
    """
    try:
        import anndata
    except ImportError as error:
        raise ImportError(
            "load_replogle_h5ad requires anndata. Install with `pip install anndata`."
        ) from error

    adata = anndata.read_h5ad(path, backed=backed)

    obs_cols = set(adata.obs.columns)
    target_col = target_col or _pick(obs_cols, _TARGET_COL_CANDIDATES)
    guide_col = guide_col or _pick(obs_cols, _GUIDE_COL_CANDIDATES)
    batch_col = batch_col or _pick(obs_cols, _BATCH_COL_CANDIDATES)

    if target_col is None:
        raise AnchorOpError(
            f"Could not detect a target-gene column in obs (tried {_TARGET_COL_CANDIDATES}). "
            f"Available: {sorted(obs_cols)[:12]}. Pass target_col= explicitly."
        )
    if guide_col is None:
        raise AnchorOpError(
            f"Could not detect a guide/sgRNA column in obs (tried {_GUIDE_COL_CANDIDATES}). "
            f"Available: {sorted(obs_cols)[:12]}. Pass guide_col= explicitly."
        )

    if control_label is None:
        control_label = _detect_nt_label(adata.obs[target_col])

    # If var_names look like Ensembl IDs and an obs 'gene_id' column exists,
    # route target lookup through the gene_id column so target_gene values
    # match var_names directly. This is standard in Replogle 2022 h5ads,
    # where var_names are ENSG and obs['gene'] is symbol.
    used_gene_id_for_target = False
    if _looks_like_ensembl(adata.var_names) and "gene_id" in adata.obs.columns:
        target_source = adata.obs["gene_id"].astype(str)
        used_gene_id_for_target = True
    else:
        target_source = adata.obs[target_col].astype(str)

    is_ctrl = adata.obs[target_col].astype(str) == str(control_label)
    n_ctrl = int(is_ctrl.sum())
    n_cells = adata.n_obs
    if n_ctrl == 0:
        raise AnchorOpError(
            f"No non-targeting cells found under label {control_label!r} in column {target_col!r}."
        )

    # Build canonical obs columns without mutating the originals (kept for provenance).
    adata.obs[canonical_target_key] = target_source
    adata.obs.loc[is_ctrl, canonical_target_key] = ""
    adata.obs[canonical_guide_key] = adata.obs[guide_col].astype(str)
    adata.obs.loc[is_ctrl, canonical_guide_key] = canonical_nt_label

    if batch_col is not None and batch_col != "gemgroup":
        adata.obs["gemgroup"] = adata.obs[batch_col].astype(str)

    adata.uns.setdefault("anchorop_replogle_provenance", {}).update({
        "source_path": str(path),
        "detected_target_col": target_col,
        "detected_guide_col": guide_col,
        "detected_batch_col": batch_col,
        "detected_control_label": control_label,
        "used_gene_id_for_target": used_gene_id_for_target,
        "var_names_look_like_ensembl": _looks_like_ensembl(adata.var_names),
        "n_cells": n_cells,
        "n_control_cells": n_ctrl,
        "n_perturbed_cells": n_cells - n_ctrl,
        "n_unique_targets": int(adata.obs.loc[~is_ctrl, canonical_target_key].nunique()),
        "n_unique_guides": int(adata.obs.loc[~is_ctrl, canonical_guide_key].nunique()),
        "canonical_control_label": canonical_nt_label,
        "canonical_guide_key": canonical_guide_key,
        "canonical_target_key": canonical_target_key,
        "backed": backed,
    })

    return adata
