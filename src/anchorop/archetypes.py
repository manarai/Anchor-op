"""Learn interpretable spectral and operator archetypes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._utils import project_to_simplex, stable_rng, to_dense
from .measure import measure_from_sensitivity
from .types import AnchorOpError, ArchetypeResult, MeasuredOperator, TransferResult


def spectral_summary(operator: np.ndarray) -> np.ndarray:
    """Return a deterministic fixed-length spectrum summary for archetype fitting."""
    eigenvalues = np.linalg.eigvals(to_dense(operator))
    ordering = np.lexsort((eigenvalues.imag, eigenvalues.real))
    ordered = eigenvalues[ordering]
    return np.concatenate([ordered.real, ordered.imag])


def _require_full_operator(measured: MeasuredOperator, purpose: str) -> np.ndarray:
    report = measured.report
    if report is None:
        raise AnchorOpError(f"{purpose} requires a measurement with an AnchorReport.")
    if not report.full_domain_identified:
        raise AnchorOpError(
            f"{purpose} requires a full effective-rank operator. Partial zero-extended actions cannot "
            "be safely mixed as operator archetypes."
        )
    return measured.J


def _features(
    measurements: Sequence[MeasuredOperator | np.ndarray],
    mode: str,
) -> tuple[np.ndarray, tuple[int, ...], list[np.ndarray]]:
    mode = mode.lower()
    if mode not in {"operator", "spectral"}:
        raise AnchorOpError("mode must be 'operator' or 'spectral'.")
    matrices: list[np.ndarray] = []
    for item in measurements:
        if isinstance(item, MeasuredOperator):
            matrix = _require_full_operator(item, f"{mode} archetype fitting")
        else:
            matrix = to_dense(item)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise AnchorOpError("Raw archetype inputs must be square matrices.")
        matrices.append(matrix)
    if len(matrices) < 1:
        raise AnchorOpError("At least one operator is required for archetype fitting.")
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices):
        raise AnchorOpError("All operators must share the same program-space dimension.")
    if mode == "operator":
        symmetric = [0.5 * (matrix + matrix.T) for matrix in matrices]
        return np.vstack([matrix.reshape(1, -1) for matrix in symmetric]), shape, symmetric
    return (
        np.vstack([spectral_summary(matrix).reshape(1, -1) for matrix in matrices]),
        (2 * shape[0],),
        matrices,
    )


def _farthest_archetype_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Choose observed extreme points via deterministic farthest-point traversal."""
    n_samples = X.shape[0]
    if not 1 <= k <= n_samples:
        raise AnchorOpError(f"k must lie in [1, {n_samples}] for observed-extreme archetypes.")
    centered = X - X.mean(axis=0, keepdims=True)
    selected = [int(np.argmax(np.sum(centered**2, axis=1)))]
    min_distances = np.sum((X - X[selected[0]]) ** 2, axis=1)
    min_distances[selected[0]] = -np.inf
    while len(selected) < k:
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        next_distances = np.sum((X - X[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, next_distances)
        min_distances[np.asarray(selected, dtype=int)] = -np.inf
    return np.asarray(selected, dtype=int)


def simplex_weights(
    data: np.ndarray,
    archetypes: np.ndarray,
    *,
    max_iter: int = 1000,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Fit nonnegative, sum-to-one reconstruction weights by projected gradient descent."""
    data = to_dense(data)
    archetypes = to_dense(archetypes)
    if data.ndim != 2 or archetypes.ndim != 2 or data.shape[1] != archetypes.shape[1]:
        raise AnchorOpError(
            "Data and flattened archetypes must be two-dimensional with matched features."
        )
    k = archetypes.shape[0]
    weights = np.full((data.shape[0], k), 1.0 / k)
    gram = archetypes @ archetypes.T
    lipschitz = max(2.0 * float(np.linalg.norm(gram, ord=2)), 1e-12)
    step = 1.0 / lipschitz
    for _ in range(max_iter):
        previous = weights.copy()
        gradient = 2.0 * (weights @ archetypes - data) @ archetypes.T
        for row in range(weights.shape[0]):
            weights[row] = project_to_simplex(weights[row] - step * gradient[row])
        if float(np.max(np.abs(weights - previous))) < tolerance:
            break
    return weights


def _fit_observed_extreme(
    X: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    indices = _farthest_archetype_indices(X, k)
    archetypes = X[indices]
    weights = simplex_weights(X, archetypes)
    reconstruction = weights @ archetypes
    error = float(np.mean(np.sum((X - reconstruction) ** 2, axis=1)))
    return archetypes, weights, error, indices


def _heldout_guide_error(
    measurements: Sequence[MeasuredOperator],
    k: int,
    *,
    n_splits: int,
    holdout_fraction: float,
    seed: int | None,
) -> float:
    """Select `k` through guide-held-out equation residuals.

    Each split estimates training actions from one subset of perturbation guides,
    learns observed-extreme symmetric archetypes across states, projects each
    training action into the simplex, and evaluates ``||J_pred S_test + U_test||``
    on guides excluded from fitting. This is intentionally more stringent than
    an information criterion on the same operator matrix.
    """
    rng = stable_rng(seed)
    errors: list[float] = []
    for _ in range(n_splits):
        train_actions: list[np.ndarray] = []
        heldout_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        valid = True
        for measurement in measurements:
            m = measurement.S.shape[1]
            if m < 2:
                valid = False
                break
            n_test = max(1, int(round(m * holdout_fraction)))
            if n_test >= m:
                n_test = m - 1
            order = rng.permutation(m)
            test_columns = order[:n_test]
            train_columns = order[n_test:]
            report = measurement.report
            assert report is not None
            try:
                train_measurement = measure_from_sensitivity(
                    measurement.S[:, train_columns],
                    measurement.U[:, train_columns],
                    guide_names=np.asarray(measurement.guide_names)[train_columns],
                    reg=report.regularization_method,
                    reg_param="path",
                    rank_tol=report.rank_tol,
                )
            except AnchorOpError:
                valid = False
                break
            train_report = train_measurement.report
            if train_report is None or not train_report.full_domain_identified:
                valid = False
                break
            train_action = train_measurement.J
            train_actions.append(0.5 * (train_action + train_action.T))
            heldout_pairs.append((measurement.S[:, test_columns], measurement.U[:, test_columns]))
        if not valid or len(train_actions) < k:
            continue
        X = np.vstack([action.reshape(1, -1) for action in train_actions])
        archetypes, weights, _, _ = _fit_observed_extreme(X, k)
        predicted = weights @ archetypes
        for index, (S_test, U_test) in enumerate(heldout_pairs):
            predicted_operator = predicted[index].reshape(train_actions[index].shape)
            residual = np.linalg.norm(predicted_operator @ S_test + U_test, ord="fro")
            scale = max(np.linalg.norm(U_test, ord="fro"), np.finfo(float).eps)
            errors.append(float(residual / scale))
    if not errors:
        raise AnchorOpError(
            "Guide-held-out k selection could not form valid splits. Supply at least k full-rank measurements "
            "with two or more retained guides each, or choose k explicitly."
        )
    return float(np.mean(errors))


def fit_archetypes(
    measurements: Sequence[MeasuredOperator | np.ndarray],
    *,
    mode: str = "operator",
    k: int | str = "cv",
    max_k: int = 8,
    n_splits: int = 5,
    holdout_fraction: float = 0.2,
    seed: int | None = 0,
) -> ArchetypeResult:
    """Fit spectral or symmetric-operator archetypes with simplex coordinates.

    Operator archetypes are intentionally learned from symmetric operators. The
    dictionary consists of observed extreme profiles selected by farthest-point
    traversal; weights are nonnegative and sum to one. For ``k='cv'`` in
    operator mode, candidate counts are selected by guide-held-out equation
    reconstruction, never by an information criterion.
    """
    if not measurements:
        raise AnchorOpError("At least one measurement is required.")
    mode = mode.lower()
    X, shape, matrices = _features(measurements, mode)
    n_samples = X.shape[0]
    if isinstance(k, str):
        if k != "cv":
            raise AnchorOpError("k must be a positive integer or 'cv'.")
        upper = min(max_k, n_samples)
        candidates = range(1, upper + 1)
        candidate_errors: dict[int, float] = {}
        if mode == "operator" and all(isinstance(item, MeasuredOperator) for item in measurements):
            typed_measurements = [
                item for item in measurements if isinstance(item, MeasuredOperator)
            ]
            for candidate in candidates:
                candidate_errors[candidate] = _heldout_guide_error(
                    typed_measurements,
                    candidate,
                    n_splits=n_splits,
                    holdout_fraction=holdout_fraction,
                    seed=seed,
                )
        else:
            # Spectral archetypes have no guide-level response matrices. Their
            # validation is clearly labeled as leave-one-state-out reconstruction.
            for candidate in candidates:
                _, _, error, _ = _fit_observed_extreme(X, candidate)
                candidate_errors[candidate] = error
        selected_k = min(candidate_errors, key=candidate_errors.get)
    elif isinstance(k, (int, np.integer)):
        selected_k = int(k)
        if not 1 <= selected_k <= n_samples:
            raise AnchorOpError(f"k must lie in [1, {n_samples}].")
        candidate_errors = {}
    else:
        raise AnchorOpError("k must be a positive integer or 'cv'.")

    archetypes_flat, weights, error, selected_indices = _fit_observed_extreme(X, selected_k)
    if mode == "operator":
        archetypes = archetypes_flat.reshape((selected_k, *shape))
    else:
        archetypes = archetypes_flat
    return ArchetypeResult(
        mode=mode,
        archetypes=archetypes,
        weights=weights,
        reconstruction_error=error,
        selected_k=selected_k,
        candidate_errors=candidate_errors,
        metadata={
            "selection": "guide-held-out equation residual"
            if mode == "operator" and isinstance(k, str)
            else "reconstruction",
            "observed_extreme_indices": selected_indices.tolist(),
            "simplex_constraint": "nonnegative weights summing to one",
            "operator_part": "symmetric" if mode == "operator" else None,
        },
    )


def transfer_test(
    source: Sequence[MeasuredOperator | np.ndarray],
    target: Sequence[MeasuredOperator | np.ndarray],
    *,
    mode: str = "operator",
    k: int | str = "cv",
    seed: int | None = 0,
) -> TransferResult:
    """Fit source archetypes, project target states, and compare with a target refit."""
    source_fit = fit_archetypes(source, mode=mode, k=k, seed=seed)
    if len(target) < source_fit.selected_k:
        raise AnchorOpError(
            "Target transfer set has fewer states than the source-selected k; a same-k target refit is required."
        )
    target_X, _, _ = _features(target, mode)
    source_archetypes = source_fit.archetypes.reshape(source_fit.selected_k, -1)
    transferred_weights = simplex_weights(target_X, source_archetypes)
    transferred_reconstruction = transferred_weights @ source_archetypes
    transfer_error = float(np.mean(np.sum((target_X - transferred_reconstruction) ** 2, axis=1)))
    target_refit = fit_archetypes(target, mode=mode, k=source_fit.selected_k, seed=seed)
    refit_error = target_refit.reconstruction_error
    return TransferResult(
        weights=transferred_weights,
        transfer_error=transfer_error,
        refit_error=refit_error,
        error_ratio=float(transfer_error / max(refit_error, np.finfo(float).eps)),
    )
