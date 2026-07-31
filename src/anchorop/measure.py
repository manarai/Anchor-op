"""Measure identifiable local dynamical actions from Perturb-seq data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._utils import (
    expression_and_metadata,
    frobenius_relative_error,
    require_column,
    stable_rng,
    to_dense,
)
from .identifiability import make_anchor_report, regularized_pseudoinverse
from .programs import project_expression
from .types import AnchorOpError, LinearityResult, MeasuredOperator, ProgramBasis


@dataclass(frozen=True)
class GuideResponse:
    """Guide-level response and input encoding retained for a measurement."""

    guide: str
    target: str
    response: np.ndarray
    input_vector: np.ndarray
    efficiency: float
    n_cells: int


def _matched_mean(
    values: np.ndarray,
    *,
    perturbed_mask: np.ndarray,
    control_mask: np.ndarray,
    batches: np.ndarray | None,
) -> np.ndarray:
    """Compute a control mean matched to perturbed-cell batch composition."""
    if batches is None:
        return values[control_mask].mean(axis=0)
    pert_batches = batches[perturbed_mask]
    result = np.zeros(values.shape[1], dtype=float)
    total = 0
    for batch in np.unique(pert_batches):
        n_perturbed = int(np.sum(pert_batches == batch))
        batch_controls = control_mask & (batches == batch)
        if not np.any(batch_controls):
            raise AnchorOpError(f"No matched control cells for batch {batch!r}.")
        result += n_perturbed * values[batch_controls].mean(axis=0)
        total += n_perturbed
    return result / total


def _derive_targets(
    guide_values: np.ndarray,
    target_values: np.ndarray | None,
    guide_to_target: Mapping[str, str] | None,
    control_label: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Derive one unambiguous target per guide without parsing guide sequences."""
    guide_values = guide_values.astype(str)
    non_controls = sorted(set(guide_values) - {str(control_label)})
    retained: dict[str, str] = {}
    dropped: dict[str, str] = {}
    if target_values is None and guide_to_target is None:
        raise AnchorOpError(
            "Provide target_key or guide_to_target. anchor-op will not infer targets by parsing guide identifiers."
        )
    for guide in non_controls:
        if guide_to_target is not None:
            target = guide_to_target.get(guide)
            if target is None:
                dropped[guide] = "target_missing_from_guide_to_target"
                continue
            retained[guide] = str(target)
            continue
        assert target_values is not None
        targets = np.unique(target_values[guide_values == guide].astype(str))
        targets = targets[targets != ""]
        if len(targets) != 1:
            dropped[guide] = "ambiguous_or_missing_target_annotation"
            continue
        retained[guide] = str(targets[0])
    return retained, dropped


def estimate_knockdown_efficiency(
    expression: np.ndarray,
    *,
    target_index: int,
    perturbed_mask: np.ndarray,
    control_mask: np.ndarray,
    batches: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> float:
    """Estimate guide efficiency from target-transcript mean reduction.

    ``efficiency = 1 - mean(target_pert) / mean(target_ctrl)``, clipped to [0, 1].

    This is the most precise estimator when the target gene is well expressed
    in controls. It degenerates to 1.0 whenever the control mean approaches
    zero — the "dropout-driven pseudo-perfect knockdown" pathology documented
    in ``examples/05_linearity_diagnostics.ipynb`` for lncRNA / low-baseline
    targets. Use :func:`estimate_knockdown_efficiency_detection_rate` on data
    where target baseline expression may be near the dropout floor.
    """
    target_values = expression[:, [target_index]]
    control_mean = float(
        _matched_mean(
            target_values,
            perturbed_mask=perturbed_mask,
            control_mask=control_mask,
            batches=batches,
        )[0]
    )
    perturbed_mean = float(target_values[perturbed_mask].mean())
    if control_mean <= epsilon:
        return 0.0
    return float(np.clip(1.0 - perturbed_mean / control_mean, 0.0, 1.0))


def estimate_knockdown_efficiency_detection_rate(
    expression: np.ndarray,
    *,
    target_index: int,
    perturbed_mask: np.ndarray,
    control_mask: np.ndarray,
    batches: np.ndarray | None = None,
) -> float:
    """Estimate guide efficiency from the drop in target-transcript detection rate.

    ``detection_rate(cells) = fraction of cells with any UMI for the target``.
    ``efficiency = max(0, detection_ctrl - detection_pert)``.

    Robust to dropout at low-baseline targets because it does not divide by a
    near-zero mean. Slightly less precise than the mean-ratio estimator at
    high-baseline targets, where both estimators agree closely. On the K562
    aggregate this reduces the 1.0-spike from 35 to 0 guides and improves the
    linearity check by ~12×. See ``examples/05_linearity_diagnostics.ipynb``.
    """
    target_values = expression[:, target_index]
    if batches is None:
        ctrl_det = float((target_values[control_mask] > 0).mean())
    else:
        pert_batches = batches[perturbed_mask]
        total = 0
        det_sum = 0.0
        for batch in np.unique(pert_batches):
            n_perturbed = int(np.sum(pert_batches == batch))
            batch_controls = control_mask & (batches == batch)
            if not np.any(batch_controls):
                raise AnchorOpError(f"No matched control cells for batch {batch!r}.")
            det_sum += n_perturbed * float((target_values[batch_controls] > 0).mean())
            total += n_perturbed
        ctrl_det = det_sum / total
    pert_det = float((target_values[perturbed_mask] > 0).mean())
    return float(np.clip(ctrl_det - pert_det, 0.0, 1.0))


_EFFICIENCY_ESTIMATORS = {
    "mean_ratio": estimate_knockdown_efficiency,
    "detection_rate": estimate_knockdown_efficiency_detection_rate,
}


def build_guide_responses(
    adata: Any,
    basis: ProgramBasis,
    *,
    guide_key: str,
    control_label: str,
    target_key: str | None = None,
    guide_to_target: Mapping[str, str] | None = None,
    batch_key: str | None = None,
    min_cells_per_guide: int = 10,
    min_knockdown_efficiency: float = 0.05,
    loading_tol: float = 1e-8,
    efficiency_estimator: str = "detection_rate",
) -> tuple[list[GuideResponse], dict[str, str]]:
    """Estimate guide-level ``Δz`` and perturbation inputs from an AnnData-like input.

    The target transcript is used only to estimate guide efficacy. Its program
    encoding is the corresponding loading row, not a one-hot program vector.

    ``efficiency_estimator`` selects how per-guide knockdown efficiency `κ` is
    computed from the target transcript:
    - ``"detection_rate"`` (default): fraction of cells with any UMI for the
      target drops from control to perturbed. Robust to dropout at low-baseline
      targets — dissolves the pseudo-perfect-knockdown pathology.
    - ``"mean_ratio"``: classical `1 - mean_pert / mean_ctrl`. More precise for
      high-baseline targets but degenerates to 1.0 whenever control mean is at
      the dropout floor. Kept for backward compatibility.
    """
    if min_cells_per_guide < 1:
        raise AnchorOpError("min_cells_per_guide must be positive.")
    if not 0 <= min_knockdown_efficiency <= 1:
        raise AnchorOpError("min_knockdown_efficiency must lie in [0, 1].")
    if efficiency_estimator not in _EFFICIENCY_ESTIMATORS:
        raise AnchorOpError(
            f"efficiency_estimator must be one of {sorted(_EFFICIENCY_ESTIMATORS)}; "
            f"got {efficiency_estimator!r}."
        )
    efficiency_fn = _EFFICIENCY_ESTIMATORS[efficiency_estimator]
    X, obs, gene_names = expression_and_metadata(adata)
    guides = require_column(obs, guide_key).astype(str).to_numpy()
    controls = guides == str(control_label)
    if not np.any(controls):
        raise AnchorOpError(
            f"No matched controls found: obs[{guide_key!r}] contains no {control_label!r} label."
        )
    target_values = require_column(obs, target_key).to_numpy() if target_key is not None else None
    batches = (
        require_column(obs, batch_key).astype(str).to_numpy() if batch_key is not None else None
    )
    targets_by_guide, dropped = _derive_targets(
        guides, target_values, guide_to_target, control_label
    )
    source_gene_index = {gene: index for index, gene in enumerate(gene_names)}
    z = project_expression(X, basis, gene_names=gene_names)
    responses: list[GuideResponse] = []

    for guide, target in targets_by_guide.items():
        perturbed = guides == guide
        n_cells = int(np.sum(perturbed))
        if n_cells < min_cells_per_guide:
            dropped[guide] = f"fewer_than_{min_cells_per_guide}_guide_positive_cells"
            continue
        if target not in source_gene_index:
            dropped[guide] = "target_gene_absent_from_expression_matrix"
            continue
        try:
            matched_z = _matched_mean(
                z,
                perturbed_mask=perturbed,
                control_mask=controls,
                batches=batches,
            )
            efficiency = efficiency_fn(
                X,
                target_index=source_gene_index[target],
                perturbed_mask=perturbed,
                control_mask=controls,
                batches=batches,
            )
        except AnchorOpError as error:
            dropped[guide] = f"unmatched_controls: {error}"
            continue
        if efficiency < min_knockdown_efficiency:
            dropped[guide] = "insufficient_target_transcript_knockdown"
            continue
        basis_gene_index = {gene: index for index, gene in enumerate(basis.gene_names)}
        if target not in basis_gene_index:
            # Possible only where a custom projection subselected basis genes.
            dropped[guide] = "target_gene_absent_from_program_basis"
            continue
        program_loading = basis.loadings[basis_gene_index[target]]
        if float(np.linalg.norm(program_loading)) < loading_tol:
            dropped[guide] = "negligible_program_loading"
            continue
        response = z[perturbed].mean(axis=0) - matched_z
        input_vector = -efficiency * program_loading
        responses.append(
            GuideResponse(
                guide=guide,
                target=target,
                response=response,
                input_vector=input_vector,
                efficiency=efficiency,
                n_cells=n_cells,
            )
        )
    if not responses:
        raise AnchorOpError("No informative perturbation guides remain after required filtering.")
    return responses, dropped


def _bootstrap_actions(
    S: np.ndarray,
    U: np.ndarray,
    *,
    n_bootstrap: int,
    method: str,
    parameter: str | float | int,
    rank_tol: float | None,
    seed: int | None,
) -> np.ndarray | None:
    if n_bootstrap <= 0:
        return None
    if n_bootstrap < 2:
        raise AnchorOpError("bootstrap must be zero or at least two guide-resampling replicates.")
    rng = stable_rng(seed)
    actions = np.empty((n_bootstrap, S.shape[0], S.shape[0]), dtype=float)
    m = S.shape[1]
    for index in range(n_bootstrap):
        columns = rng.integers(0, m, size=m)
        S_sample = S[:, columns]
        U_sample = U[:, columns]
        pseudoinverse, _, _, _ = regularized_pseudoinverse(
            S_sample,
            method=method,
            parameter=parameter,
            rank_tol=rank_tol,
        )
        actions[index] = -U_sample @ pseudoinverse
    return actions


def measure_from_sensitivity(
    S: np.ndarray,
    U: np.ndarray,
    *,
    guide_names: Sequence[str] | None = None,
    guide_efficiencies: Mapping[str, float] | None = None,
    dropped_guides: Mapping[str, str] | None = None,
    reg: str = "tsvd",
    reg_param: str | float | int = "path",
    rank_tol: float | None = None,
    bootstrap: int = 0,
    bootstrap_seed: int | None = 0,
    state_label: str | None = None,
    notes: Sequence[str] = (),
) -> MeasuredOperator:
    """Construct a measurement from already estimated ``S`` and ``U`` matrices.

    This lower-level API is intended for simulations, reproducibility tests, and
    upstream pipelines that have already performed guide-level matching. It
    still returns the mandatory report; there is no unchecked matrix-inverse API.

    ``rank_tol=None`` (default) uses the eps-based numerical rank, which is what
    algebraic tests want. Real measurement data should pass a scientifically
    motivated relative threshold (e.g., 1e-2); :func:`measure_operator` applies
    that default automatically.
    """
    S = to_dense(S)
    U = to_dense(U)
    if S.ndim != 2 or U.ndim != 2 or S.shape != U.shape:
        raise AnchorOpError("S and U must be finite, equally shaped d-by-m matrices.")
    if S.shape[0] > 200:
        raise AnchorOpError("anchor-op refuses program spaces with d > 200.")
    if S.shape[1] < 1:
        raise AnchorOpError("At least one retained perturbation is required.")
    raw_names = [f"guide_{i}" for i in range(S.shape[1])] if guide_names is None else guide_names
    names = tuple(str(value) for value in raw_names)
    if len(names) != S.shape[1]:
        raise AnchorOpError("guide_names must have one entry per sensitivity column.")
    pseudoinverse, selected, path, response_projector = regularized_pseudoinverse(
        S,
        method=reg,
        parameter=reg_param,
        rank_tol=rank_tol,
    )
    action = -U @ pseudoinverse
    bootstrap_actions = _bootstrap_actions(
        S,
        U,
        n_bootstrap=bootstrap,
        method=reg,
        parameter=reg_param,
        rank_tol=rank_tol,
        seed=bootstrap_seed,
    )
    report = make_anchor_report(
        U,
        S,
        guide_names=names,
        dropped_guides={} if dropped_guides is None else dropped_guides,
        guide_efficiencies={} if guide_efficiencies is None else guide_efficiencies,
        method=reg,
        selected=selected,
        path=path,
        response_projector=response_projector,
        rank_tol=rank_tol,
        bootstrap_actions=bootstrap_actions,
        notes=notes,
    )
    return MeasuredOperator(
        _identified_action=action,
        S=S,
        U=U,
        report=report,
        guide_names=names,
        state_label=state_label,
    )


def measure_operator(
    adata: Any,
    basis: ProgramBasis,
    *,
    guide_key: str,
    control_label: str,
    target_key: str | None = None,
    guide_to_target: Mapping[str, str] | None = None,
    batch_key: str | None = None,
    min_cells_per_guide: int = 10,
    min_knockdown_efficiency: float = 0.05,
    loading_tol: float = 1e-8,
    reg: str = "tsvd",
    reg_param: str | float | int = "path",
    rank_tol: float = 1e-2,
    bootstrap: int = 0,
    bootstrap_seed: int | None = 0,
    state_label: str | None = None,
    efficiency_estimator: str = "detection_rate",
) -> MeasuredOperator:
    """Estimate a measured action and mandatory report from pooled Perturb-seq data.

    Every guide is analyzed separately. Perturbation cells are matched to control
    cells by batch when ``batch_key`` is supplied. Uninformative perturbations
    are dropped with a reason recorded in ``MeasuredOperator.report``.

    ``rank_tol`` defaults to ``1e-2`` (a singular direction of ``S`` must exceed
    one percent of the leading direction to be treated as identified). This
    guards against below-noise directions being called "full rank" — a real
    hazard when guides are collinear (paralogs, complex subunits, shared
    pathway members). Publications should record any deviation from this default.

    ``efficiency_estimator`` defaults to ``"detection_rate"``, which uses the
    drop in target-transcript detection rate between control and perturbed
    cells. It is robust to scRNA-seq dropout at low-baseline targets — the
    ``"mean_ratio"`` alternative degenerates to 1.0 whenever control mean is at
    the dropout floor. On the K562 aggregate the switch reduces the artifact
    spike at efficiency ≈ 1.0 from 35/72 guides to zero and improves the
    linearity check by ~12×. See ``examples/05_linearity_diagnostics.ipynb``.
    """
    responses, dropped = build_guide_responses(
        adata,
        basis,
        guide_key=guide_key,
        control_label=control_label,
        target_key=target_key,
        guide_to_target=guide_to_target,
        batch_key=batch_key,
        min_cells_per_guide=min_cells_per_guide,
        min_knockdown_efficiency=min_knockdown_efficiency,
        loading_tol=loading_tol,
        efficiency_estimator=efficiency_estimator,
    )
    guide_names = tuple(record.guide for record in responses)
    S = np.column_stack([record.response for record in responses])
    U = np.column_stack([record.input_vector for record in responses])
    efficiencies = {record.guide: record.efficiency for record in responses}
    return measure_from_sensitivity(
        S,
        U,
        guide_names=guide_names,
        guide_efficiencies=efficiencies,
        dropped_guides=dropped,
        reg=reg,
        reg_param=reg_param,
        rank_tol=rank_tol,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
        state_label=state_label,
        notes=(
            "Guide-level S and U were estimated from matched perturbation and control means.",
            f"Knockdown efficiency was estimated per target via '{efficiency_estimator}'.",
            f"rank_tol={rank_tol:.2e} was used to decide which singular directions of S are identified.",
        ),
    )


def _split_rel_diff(
    measured: MeasuredOperator,
    mask_A: np.ndarray,
    method: str,
    parameter: str | float | int,
    rank_tol: float | None,
) -> tuple[float, int, MeasuredOperator | None, MeasuredOperator | None]:
    """Rel-diff on the common identified subspace between two bin masks.

    Returns ``(rel_diff, overlap_rank, mA, mB)``. ``rel_diff`` is ``inf`` when
    the two bins share no common identified subspace. Used both for the
    efficiency split and for random-split null draws.
    """
    mask_B = ~mask_A
    if mask_A.sum() < 1 or mask_B.sum() < 1:
        return float("inf"), 0, None, None
    guide_names = np.asarray(measured.guide_names)
    m_A = measure_from_sensitivity(
        measured.S[:, mask_A],
        measured.U[:, mask_A],
        guide_names=guide_names[mask_A],
        reg=method,
        reg_param=parameter,
        rank_tol=rank_tol,
    )
    m_B = measure_from_sensitivity(
        measured.S[:, mask_B],
        measured.U[:, mask_B],
        guide_names=guide_names[mask_B],
        reg=method,
        reg_param=parameter,
        rank_tol=rank_tol,
    )
    Pa = m_A.response_projector
    Pb = m_B.response_projector
    va, wa = np.linalg.eigh(0.5 * (Pa + Pa.T))
    vb, wb = np.linalg.eigh(0.5 * (Pb + Pb.T))
    Ba = wa[:, va > 0.5]
    Bb = wb[:, vb > 0.5]
    sv = np.linalg.svd(Ba.T @ Bb, compute_uv=False)
    overlap = int(np.sum(sv > 1.0 - 1e-7))
    if overlap == 0:
        return float("inf"), 0, m_A, m_B
    left, _, _ = np.linalg.svd(Ba.T @ Bb, full_matrices=False)
    common = Ba @ left[:, :overlap]
    Pc = common @ common.T
    # Symmetric relative difference so the metric doesn't depend on which bin
    # is arbitrarily labeled "reference" — critical for the null distribution
    # to be meaningful under random splits (where the labeling is arbitrary).
    a_proj = m_A.identified_action @ Pc
    b_proj = m_B.identified_action @ Pc
    numerator = float(np.linalg.norm(a_proj - b_proj, ord="fro"))
    denominator = 0.5 * (float(np.linalg.norm(a_proj, ord="fro")) + float(np.linalg.norm(b_proj, ord="fro")))
    diff = numerator / max(denominator, np.finfo(float).eps)
    return diff, overlap, m_A, m_B


def linearity_check(
    measured: MeasuredOperator,
    *,
    reg: str | None = None,
    reg_param: str | float | int | None = None,
    threshold: float = 0.25,
    n_null: int = 0,
    null_seed: int | None = 0,
) -> LinearityResult:
    """Compare weak- and strong-efficiency bin operators as a linearity diagnostic.

    Under perfect linearity the two bins should identify the same operator on
    their common subspace. In practice, comparing operators from disjoint
    guide subsets on real Perturb-seq data has an irreducible **bin-composition
    floor** (each subset samples different columns of `J`) that dominates the
    raw ``rel_diff`` even when linearity holds. Set ``n_null > 0`` to draw
    that floor as an explicit random-split null; the returned
    ``excess_above_null`` is the part of the observed disagreement that the
    null does not reproduce — i.e. the real dose-response / model-mismatch
    contribution.

    When ``n_null > 0`` the ``passed`` decision uses ``excess_above_null`` vs
    ``threshold`` instead of raw ``relative_difference``. Backward-compatible:
    with ``n_null=0`` (default) the raw rel-diff rule is unchanged and null
    fields on ``LinearityResult`` are ``None``.
    """
    report = measured.report
    if report is None:
        raise AnchorOpError("A report is required for a linearity diagnostic.")
    efficiencies = np.asarray(
        [report.guide_efficiencies.get(name, np.nan) for name in measured.guide_names]
    )
    if np.any(~np.isfinite(efficiencies)):
        raise AnchorOpError(
            "Guide efficiencies are unavailable; rerun measure_operator() rather than measure_from_sensitivity()."
        )
    median = float(np.median(efficiencies))
    weak = efficiencies <= median
    strong = efficiencies > median
    if weak.sum() < 1 or strong.sum() < 1:
        raise AnchorOpError("Linearity check needs at least one guide in each efficiency bin.")
    method = report.regularization_method if reg is None else reg
    parameter: str | float | int = "path" if reg_param is None else reg_param

    difference, overlap_rank, weak_measurement, strong_measurement = _split_rel_diff(
        measured, weak, method, parameter, report.rank_tol
    )

    # Random-split null distribution (optional).
    null_median = null_std = null_p95 = excess = z = None
    if n_null > 0:
        rng = stable_rng(null_seed)
        n_total = measured.S.shape[1]
        null_vals: list[float] = []
        for _ in range(n_null):
            perm = rng.permutation(n_total)
            random_mask = np.zeros(n_total, dtype=bool)
            random_mask[perm[: n_total // 2]] = True
            rd, _, _, _ = _split_rel_diff(
                measured, random_mask, method, parameter, report.rank_tol
            )
            if np.isfinite(rd):
                null_vals.append(rd)
        if null_vals:
            arr = np.asarray(null_vals, dtype=float)
            null_median = float(np.median(arr))
            null_std = float(arr.std())
            null_p95 = float(np.percentile(arr, 95))
            if np.isfinite(difference):
                excess = float(difference - null_median)
                z = float((difference - arr.mean()) / max(arr.std(), 1e-12))

    if n_null > 0 and excess is not None:
        passed = bool(overlap_rank > 0 and excess <= threshold)
    else:
        passed = bool(overlap_rank > 0 and difference <= threshold)

    weak_action = (
        weak_measurement.identified_action
        if weak_measurement is not None
        else np.zeros((report.d, report.d))
    )
    strong_action = (
        strong_measurement.identified_action
        if strong_measurement is not None
        else np.zeros((report.d, report.d))
    )

    return LinearityResult(
        weak_action=weak_action,
        strong_action=strong_action,
        relative_difference=difference,
        weak_guides=tuple(np.asarray(measured.guide_names)[weak]),
        strong_guides=tuple(np.asarray(measured.guide_names)[strong]),
        passed=passed,
        threshold=float(threshold),
        overlap_rank=overlap_rank,
        null_median=null_median,
        null_std=null_std,
        null_p95=null_p95,
        excess_above_null=excess,
        z_score=z,
        n_null=n_null,
    )
