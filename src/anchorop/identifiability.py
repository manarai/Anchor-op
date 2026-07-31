"""Regularized inversion and mandatory identifiability disclosure.

For ``J S = -U``, the directly identified object is ``J P_X = -U S^+``, where
``P_X`` projects onto the retained response subspace of ``S``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from ._utils import orthogonal_projector, to_dense
from .types import AnchorOpError, AnchorReport, RegularizationPathEntry


def _svd(
    sensitivity: np.ndarray, *, rank_tol: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """SVD plus the rank cutoff used for identifiability decisions.

    ``rank_tol`` is a relative threshold on ``singular_values / singular_values[0]``.
    ``None`` uses the eps-based numerical rank; measurement APIs should pass a
    scientifically justified value (typically 1e-2 for real single-cell data).
    """
    sensitivity = to_dense(sensitivity)
    if sensitivity.ndim != 2:
        raise AnchorOpError("Sensitivity matrix S must be two-dimensional.")
    if sensitivity.shape[1] == 0:
        raise AnchorOpError("Sensitivity matrix S has no retained perturbation columns.")
    left, singular_values, right_t = np.linalg.svd(sensitivity, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0:
        raise AnchorOpError("Sensitivity matrix S has no nonzero singular direction.")
    if rank_tol is None:
        tolerance = np.finfo(float).eps * max(sensitivity.shape) * singular_values[0]
    else:
        if not np.isfinite(rank_tol) or rank_tol < 0:
            raise AnchorOpError("rank_tol must be a nonnegative real number or None.")
        tolerance = float(rank_tol) * singular_values[0]
    numerical_rank = int(np.sum(singular_values > tolerance))
    if numerical_rank == 0:
        raise AnchorOpError(
            "No singular direction of S survives rank_tol; loosen the tolerance or refit inputs."
        )
    return left, singular_values, right_t, numerical_rank, tolerance


def _condition_number(singular_values: np.ndarray, retained_mask: np.ndarray) -> float:
    retained = singular_values[retained_mask]
    if retained.size == 0:
        return float("inf")
    smallest = float(np.min(retained))
    if smallest <= 0:
        return float("inf")
    return float(np.max(retained) / smallest)


def _tsvd_entry(singular_values: np.ndarray, rank: int) -> RegularizationPathEntry:
    retained = np.zeros_like(singular_values, dtype=bool)
    retained[:rank] = True
    filters = np.zeros_like(singular_values, dtype=float)
    filters[retained] = 1.0 / singular_values[retained]
    return RegularizationPathEntry(
        method="tsvd",
        parameter=int(rank),
        effective_rank=int(rank),
        retained_mask=retained,
        singular_values=singular_values,
        filter_factors=filters,
        condition_number=_condition_number(singular_values, retained),
    )


def _tikhonov_entry(
    singular_values: np.ndarray, alpha: float, tolerance: float
) -> RegularizationPathEntry:
    if alpha < 0:
        raise AnchorOpError("Tikhonov alpha must be nonnegative.")
    filters = singular_values / (singular_values**2 + alpha)
    retained = (singular_values > tolerance) & (singular_values**2 > alpha)
    return RegularizationPathEntry(
        method="tikhonov",
        parameter=float(alpha),
        effective_rank=int(np.sum(retained)),
        retained_mask=retained,
        singular_values=singular_values,
        filter_factors=filters,
        condition_number=_condition_number(singular_values, retained),
    )


def regularization_path(
    sensitivity: np.ndarray,
    *,
    method: str,
    parameters: Iterable[float | int] | None = None,
    rank_tol: float | None = None,
) -> tuple[RegularizationPathEntry, ...]:
    """Compute a complete, inspectable TSVD or Tikhonov regularization path."""
    method = method.lower()
    _, singular_values, _, numerical_rank, tolerance = _svd(sensitivity, rank_tol=rank_tol)
    if method == "tsvd":
        ranks = range(1, numerical_rank + 1) if parameters is None else parameters
        entries: list[RegularizationPathEntry] = []
        for rank in ranks:
            if not isinstance(rank, (int, np.integer)) or not 1 <= int(rank) <= numerical_rank:
                raise AnchorOpError(f"TSVD ranks must be integers in [1, {numerical_rank}].")
            entries.append(_tsvd_entry(singular_values, int(rank)))
        return tuple(entries)
    if method == "tikhonov":
        if parameters is None:
            scale = float(singular_values[0] ** 2)
            # Covers a broad scale without representing alpha=0 as regularization.
            parameters = scale * np.logspace(-8, 0, 17)
        entries = []
        for alpha in parameters:
            if not isinstance(alpha, (int, float, np.number)) or float(alpha) < 0:
                raise AnchorOpError("Tikhonov parameters must be nonnegative real numbers.")
            entries.append(_tikhonov_entry(singular_values, float(alpha), tolerance))
        return tuple(entries)
    raise AnchorOpError("method must be 'tsvd' or 'tikhonov'.")


def _gcv_alpha(sensitivity: np.ndarray, entries: tuple[RegularizationPathEntry, ...]) -> float:
    """Select a reproducible Tikhonov alpha by generalized cross validation."""
    _, singular_values, _, _, _ = _svd(sensitivity)
    d = sensitivity.shape[0]
    scores: list[float] = []
    for entry in entries:
        alpha = float(entry.parameter)
        shrinkage = singular_values**2 / (singular_values**2 + alpha)
        residual_energy = float(np.sum((1.0 - shrinkage) ** 2 * singular_values**2))
        denominator = max((d - float(np.sum(shrinkage))) ** 2, np.finfo(float).eps)
        scores.append(residual_energy / denominator)
    return float(entries[int(np.argmin(scores))].parameter)


def regularized_pseudoinverse(
    sensitivity: np.ndarray,
    *,
    method: str = "tsvd",
    parameter: str | float | int = "path",
    rank_tol: float | None = None,
) -> tuple[np.ndarray, RegularizationPathEntry, tuple[RegularizationPathEntry, ...], np.ndarray]:
    """Return ``S⁺``, the selected entry, full path, and retained-domain projector.

    ``parameter='path'`` means that a complete path is recorded. TSVD selects the
    numerical rank at the given ``rank_tol``; Tikhonov selects a deterministic GCV
    alpha from the path. ``rank_tol=None`` (default) uses eps-based numerical
    rank, which is appropriate for algebraic use; measurement pipelines pass a
    scientifically motivated value (typically ``1e-2``) so that below-noise
    singular directions are not treated as identified.
    """
    method = method.lower()
    left, singular_values, right_t, numerical_rank, tolerance = _svd(
        sensitivity, rank_tol=rank_tol
    )
    if method == "tsvd":
        path = regularization_path(sensitivity, method="tsvd", rank_tol=rank_tol)
        if parameter == "path":
            selected_rank = numerical_rank
        elif isinstance(parameter, (int, np.integer)):
            selected_rank = int(parameter)
        else:
            raise AnchorOpError("For TSVD, parameter must be an integer rank or 'path'.")
        matches = [entry for entry in path if int(entry.parameter) == selected_rank]
        if not matches:
            raise AnchorOpError(
                f"Requested TSVD rank {selected_rank} is unavailable; numerical rank is {numerical_rank}."
            )
        selected = matches[0]
    elif method == "tikhonov":
        path = regularization_path(sensitivity, method="tikhonov", rank_tol=rank_tol)
        if parameter == "path":
            selected_alpha = _gcv_alpha(sensitivity, path)
        elif isinstance(parameter, (int, float, np.number)):
            selected_alpha = float(parameter)
            path = tuple(
                list(path)
                + list(
                    regularization_path(
                        sensitivity,
                        method="tikhonov",
                        parameters=[selected_alpha],
                        rank_tol=rank_tol,
                    )
                )
            )
        else:
            raise AnchorOpError("For Tikhonov, parameter must be a nonnegative alpha or 'path'.")
        candidates = [entry for entry in path if np.isclose(float(entry.parameter), selected_alpha)]
        selected = candidates[-1]
    else:
        raise AnchorOpError("method must be 'tsvd' or 'tikhonov'.")

    pseudoinverse = (right_t.T * selected.filter_factors) @ left.T
    retained_left = left[:, selected.retained_mask]
    response_projector = (
        retained_left @ retained_left.T
        if retained_left.size
        else np.zeros((sensitivity.shape[0],) * 2)
    )
    return pseudoinverse, selected, path, response_projector


def make_anchor_report(
    U: np.ndarray,
    S: np.ndarray,
    *,
    guide_names: Sequence[str],
    dropped_guides: Mapping[str, str],
    guide_efficiencies: Mapping[str, float],
    method: str,
    selected: RegularizationPathEntry,
    path: tuple[RegularizationPathEntry, ...],
    response_projector: np.ndarray,
    rank_tol: float | None = None,
    bootstrap_actions: np.ndarray | None = None,
    notes: Sequence[str] = (),
) -> AnchorReport:
    """Build the inseparable report that accompanies every measured action.

    ``rank_tol`` is stored on the report so that downstream analyses (linearity
    checks, archetype cross-validation) reuse the same effective-rank criterion.
    """
    U = to_dense(U)
    S = to_dense(S)
    if U.shape != S.shape:
        raise AnchorOpError("U and S must have equal d-by-m shape.")
    if len(guide_names) != S.shape[1]:
        raise AnchorOpError("guide_names must align with sensitivity columns.")
    d = S.shape[0]
    # Pass the same rank_tol to input/response projector rank estimates so that
    # a measurement-scale tolerance is not silently ignored on U and S.
    input_projector, input_rank, _ = orthogonal_projector(U, rtol=rank_tol)
    _, response_numerical_rank, _ = orthogonal_projector(S, rtol=rank_tol)
    full_domain = selected.effective_rank == d
    covariance: np.ndarray | None = None
    if bootstrap_actions is not None:
        bootstrap_actions = np.asarray(bootstrap_actions, dtype=float)
        if bootstrap_actions.shape[0] > 1:
            covariance = np.cov(
                bootstrap_actions.reshape(bootstrap_actions.shape[0], -1), rowvar=False
            )
    report_notes = list(notes)
    report_notes.append("Measured object: J P_X = -U S^+, where P_X is response_projector.")
    if rank_tol is not None:
        report_notes.append(
            f"Identifiability tolerance: singular directions below {rank_tol:.2e} * sigma_max "
            "were treated as noise, not identified."
        )
    if not full_domain:
        report_notes.append(
            "Partial identification: full spectra and hyperbolicity are blocked because the zero extension outside P_X is not measured."
        )
    return AnchorReport(
        d=d,
        input_subspace_dim=input_rank,
        response_subspace_dim=response_numerical_rank,
        effective_response_rank=selected.effective_rank,
        n_guides_input=len(guide_names) + len(dropped_guides),
        n_guides_retained=len(guide_names),
        retained_guides=tuple(map(str, guide_names)),
        dropped_guides=dict(dropped_guides),
        input_projector=input_projector,
        response_projector=response_projector,
        singular_values=selected.singular_values,
        retained_singular_directions=selected.retained_mask,
        condition_number=selected.condition_number,
        regularization_method=method,
        selected_regularization=selected.parameter,
        regularization_path=path,
        full_domain_identified=full_domain,
        rank_tol=rank_tol,
        guide_efficiencies=dict(guide_efficiencies),
        bootstrap_covariance=covariance,
        bootstrap_actions=bootstrap_actions,
        notes=tuple(report_notes),
    )
