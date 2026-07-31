"""I/O helpers for externally inferred operators.

anchor-op deliberately consumes external CellOracle, dynamo, or scJDO operator
matrices; it does not reimplement network or drift inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._utils import to_dense
from .types import AnchorOpError, ProgramBasis


def validate_operator(
    operator: Any,
    *,
    d: int | None = None,
    name: str = "operator",
) -> np.ndarray:
    """Validate a finite square inferred operator in the declared program coordinates."""
    matrix = to_dense(operator)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AnchorOpError(f"{name} must be a finite square matrix.")
    if d is not None and matrix.shape != (d, d):
        raise AnchorOpError(
            f"{name} has shape {matrix.shape}, but the measurement uses d={d} program coordinates. "
            "Transform the inferred operator into exactly the same basis before comparison."
        )
    return matrix


def load_operator(
    path: str | Path,
    *,
    d: int | None = None,
    delimiter: str = ",",
) -> np.ndarray:
    """Load a dense `.npy`, `.csv`, or tab-delimited inferred operator matrix."""
    path = Path(path)
    if not path.exists():
        raise AnchorOpError(f"Operator file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif suffix in {".csv", ".tsv", ".txt"}:
        inferred_delimiter = "\t" if suffix == ".tsv" else delimiter
        frame = pd.read_csv(path, header=None, sep=inferred_delimiter)
        matrix = frame.to_numpy(dtype=float)
    else:
        raise AnchorOpError("Supported operator file formats are .npy, .csv, .tsv, and .txt.")
    return validate_operator(matrix, d=d, name=str(path))


def require_same_program_coordinates(
    inferred_program_names: Sequence[str],
    basis: ProgramBasis,
) -> None:
    """Refuse an operator comparison unless coordinate labels exactly agree.

    Equal dimensions alone are insufficient: a matrix in a rotated, permuted, or
    differently scaled latent space is not comparable to the measured action.
    """
    inferred = tuple(map(str, inferred_program_names))
    expected = tuple(f"program_{i}" for i in range(basis.d))
    if inferred != expected:
        raise AnchorOpError(
            "Inferred operator program labels do not match anchor-op's canonical ordering. "
            "Supply the explicit coordinate transform before importing the matrix."
        )
