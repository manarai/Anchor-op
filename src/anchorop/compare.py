"""Benchmark inferred operators against a measured, identified action."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._utils import frobenius_relative_error, stable_rng, to_dense
from .io import validate_operator
from .types import AnchorOpError, ComparisonResult, IdentifiabilityError, MeasuredOperator


def _hungarian_minimum_assignment(cost: np.ndarray) -> np.ndarray:
    """Solve a square minimum-cost assignment without a SciPy dependency."""
    cost = to_dense(cost)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise AnchorOpError("The Wasserstein matching cost matrix must be square.")
    n = cost.shape[0]
    # 1-indexed primal-dual implementation of the Kuhn-Munkres algorithm.
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j] = current
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def spectral_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    """Exact equal-cardinality 2-Wasserstein distance between complex spectra.

    **Warning: unstable for non-normal operators.** Biological Jacobians are
    typically far from normal, and eigenvalues of a non-normal matrix can move
    on the order of ``||delta||^(1/k)`` for a size-``k`` defective block under a
    perturbation of size ``||delta||``. A 0.1% element perturbation to a
    Jordan-like matrix can therefore produce Wasserstein distances comparable
    to ``||J||_F`` itself. Report this metric alongside
    :func:`spectral_abscissa_difference` (a Lipschitz-stable scalar summary of
    the biologically relevant hyperbolicity magnitude) and the projected
    Frobenius operator error, not on its own.
    """
    left_eigenvalues = np.linalg.eigvals(to_dense(left))
    right_eigenvalues = np.linalg.eigvals(to_dense(right))
    if left_eigenvalues.size != right_eigenvalues.size:
        raise AnchorOpError("Spectra must contain the same number of eigenvalues.")
    cost = np.abs(left_eigenvalues[:, None] - right_eigenvalues[None, :]) ** 2
    assignment = _hungarian_minimum_assignment(cost)
    return float(np.sqrt(np.mean(cost[np.arange(cost.shape[0]), assignment])))


def spectral_abscissa_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Absolute difference of spectral abscissae ``|max Re(lambda(L)) - max Re(lambda(R))|``.

    The spectral abscissa directly measures hyperbolicity magnitude (positive
    means the operator has an unstable direction; negative means all directions
    are damped). For diagonalizable matrices it varies smoothly with the
    operator, so this scalar tolerates small measurement noise better than the
    full-spectrum Wasserstein matching. For fully-defective operators (single
    large Jordan block spanning the leading eigenvalue) every eigenvalue-based
    metric is only Hölder-continuous; report both metrics in that regime.
    Use this as the primary hyperbolicity comparison and reserve
    :func:`spectral_wasserstein` as a supplementary summary.
    """
    left_max = float(np.max(np.linalg.eigvals(to_dense(left)).real))
    right_max = float(np.max(np.linalg.eigvals(to_dense(right)).real))
    return float(abs(left_max - right_max))


def hyperbolicity_sign(operator: np.ndarray, tolerance: float = 1e-8) -> int:
    """Classify the sign of the largest real eigenvalue with a neutral tolerance."""
    maximum = float(np.max(np.linalg.eigvals(to_dense(operator)).real))
    if maximum > tolerance:
        return 1
    if maximum < -tolerance:
        return -1
    return 0


def subspace_grassmann_distance(left: np.ndarray, right: np.ndarray, k: int = 3) -> float:
    """Compute principal-angle Grassmann distance between leading complex invariant subspaces."""
    left = to_dense(left)
    right = to_dense(right)
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise AnchorOpError("Subspace comparison requires equal square matrices.")
    k = int(k)
    if not 1 <= k <= left.shape[0]:
        raise AnchorOpError(f"k must lie in [1, {left.shape[0]}].")
    values_left, vectors_left = np.linalg.eig(left)
    values_right, vectors_right = np.linalg.eig(right)
    indices_left = np.argsort(values_left.real)[::-1][:k]
    indices_right = np.argsort(values_right.real)[::-1][:k]
    q_left, _ = np.linalg.qr(vectors_left[:, indices_left])
    q_right, _ = np.linalg.qr(vectors_right[:, indices_right])
    cosines = np.linalg.svd(q_left.conj().T @ q_right, compute_uv=False)
    angles = np.arccos(np.clip(cosines.real, -1.0, 1.0))
    return float(np.linalg.norm(angles))


def _split_errors(
    inferred: np.ndarray, measured_action: np.ndarray, projector: np.ndarray
) -> dict[str, float]:
    symmetric_inferred = 0.5 * (inferred + inferred.T)
    symmetric_measured = 0.5 * (measured_action + measured_action.T)
    antisymmetric_inferred = 0.5 * (inferred - inferred.T)
    antisymmetric_measured = 0.5 * (measured_action - measured_action.T)
    symmetric_error = frobenius_relative_error(
        symmetric_inferred @ projector,
        symmetric_measured @ projector,
    )
    antisymmetric_error = frobenius_relative_error(
        antisymmetric_inferred @ projector,
        antisymmetric_measured @ projector,
    )
    return {
        "symmetric_relative_error": symmetric_error,
        "antisymmetric_relative_error": antisymmetric_error,
        "symmetric_agreement": 1.0 / (1.0 + symmetric_error),
        "antisymmetric_agreement": 1.0 / (1.0 + antisymmetric_error),
    }


def _comparison_metrics(
    measured: MeasuredOperator,
    inferred: np.ndarray,
    *,
    leading_subspace_dim: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    report = measured.report
    if report is None:
        raise IdentifiabilityError("Comparison is impossible without an AnchorReport.")
    action = measured.identified_action
    projector = measured.response_projector
    metrics: dict[str, float] = {
        "operator_relative_error": frobenius_relative_error(
            inferred @ projector,
            action @ projector,
        ),
        "equation_relative_residual": float(
            np.linalg.norm(inferred @ measured.S + measured.U, ord="fro")
            / max(np.linalg.norm(measured.U, ord="fro"), np.finfo(float).eps)
        ),
    }
    metrics.update(_split_errors(inferred, action, projector))
    metadata: dict[str, Any] = {
        "identified_domain_rank": report.effective_response_rank,
        "full_domain_identified": report.full_domain_identified,
        "operator_metric": "||(J_inf - J_meas) P_X||_F / ||J_meas P_X||_F",
    }
    if not report.full_domain_identified:
        metrics.update(
            {
                "spectral_wasserstein": float("nan"),
                "spectral_abscissa_difference": float("nan"),
                "hyperbolicity_agreement": float("nan"),
                "subspace_angle": float("nan"),
            }
        )
        metadata["spectral_status"] = (
            "blocked: a partial zero-extended action has no generally measured full spectrum"
        )
        return metrics, metadata

    measured_full = measured.J
    metrics.update(
        {
            "spectral_wasserstein": spectral_wasserstein(inferred, measured_full),
            "spectral_abscissa_difference": spectral_abscissa_difference(inferred, measured_full),
            "hyperbolicity_agreement": float(
                hyperbolicity_sign(inferred) == hyperbolicity_sign(measured_full)
            ),
            "subspace_angle": subspace_grassmann_distance(
                inferred,
                measured_full,
                k=min(leading_subspace_dim, measured_full.shape[0]),
            ),
        }
    )
    metadata["spectral_status"] = "computed: full effective response-domain rank identified"
    metadata["primary_hyperbolicity_metric"] = (
        "spectral_abscissa_difference (Lipschitz-stable); spectral_wasserstein is "
        "supplementary and can be unstable for non-normal J"
    )
    return metrics, metadata


def _draw_null(inferred: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a declared operator-level null surrogate."""
    kind = kind.lower()
    if kind == "shuffled_edges":
        values = inferred.reshape(-1).copy()
        rng.shuffle(values)
        return values.reshape(inferred.shape)
    if kind == "random_init":
        random = rng.normal(size=inferred.shape)
        target_norm = np.linalg.norm(inferred, ord="fro")
        return random * target_norm / max(np.linalg.norm(random, ord="fro"), np.finfo(float).eps)
    raise AnchorOpError("Supported nulls are 'shuffled_edges' and 'random_init'.")


def compare(
    measured: MeasuredOperator,
    inferred_operators: Mapping[str, Any],
    *,
    nulls: Sequence[str] = ("shuffled_edges", "random_init"),
    n_null: int = 100,
    seed: int | None = 0,
    leading_subspace_dim: int = 3,
) -> dict[str, ComparisonResult]:
    """Compare named inferred operators with a measured action and declared nulls.

    All primary matrix comparisons act on the **right** of the response-domain
    projector, which evaluates the experimentally identified action ``J P_X``.
    Spectral metrics are intentionally marked unavailable when the experiment
    identifies only a proper subspace.
    """
    if not isinstance(measured, MeasuredOperator):
        raise AnchorOpError("measured must be a MeasuredOperator returned by anchor-op.")
    report = measured.report
    if report is None:
        raise IdentifiabilityError("A measured action without AnchorReport cannot be compared.")
    if n_null < 0:
        raise AnchorOpError("n_null must be nonnegative.")
    rng = stable_rng(seed)
    output: dict[str, ComparisonResult] = {}
    for method, candidate in inferred_operators.items():
        inferred = validate_operator(candidate, d=report.d, name=f"inferred operator {method!r}")
        metrics, metadata = _comparison_metrics(
            measured,
            inferred,
            leading_subspace_dim=leading_subspace_dim,
        )
        null_metrics: dict[str, list[float]] = {}
        null_provenance: dict[str, Any] = {}
        for null_name in nulls:
            values: list[dict[str, float]] = []
            for _ in range(n_null):
                null_operator = _draw_null(inferred, null_name, rng)
                null_result, _ = _comparison_metrics(
                    measured,
                    null_operator,
                    leading_subspace_dim=leading_subspace_dim,
                )
                values.append(null_result)
            valid_keys = metrics.keys()
            for key in valid_keys:
                collected = np.asarray([value[key] for value in values], dtype=float)
                if np.isfinite(collected).any():
                    null_metrics[f"{null_name}:{key}"] = collected
            null_provenance[null_name] = {"n_draws": n_null}
        metadata["nulls"] = null_provenance
        output[str(method)] = ComparisonResult(
            method=str(method),
            metrics=metrics,
            null_metrics=null_metrics,
            metadata=metadata,
        )
    return output


def comparison_table(results: Mapping[str, ComparisonResult]) -> pd.DataFrame:
    """Convert a comparison mapping into a compact tabular summary."""
    rows: list[dict[str, Any]] = []
    for method, result in results.items():
        row: dict[str, Any] = {"method": method, **result.metrics}
        for null_key, values in result.null_metrics.items():
            finite = values[np.isfinite(values)]
            if finite.size:
                row[f"{null_key}:mean"] = float(np.mean(finite))
                row[f"{null_key}:sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
