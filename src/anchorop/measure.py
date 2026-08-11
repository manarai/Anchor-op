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
from .types import (
    AnchorOpError,
    HeldOutPredictionResult,
    LinearityResult,
    MeasuredOperator,
    ProgramBasis,
    TargetResponseAtlas,
    DoseResponseResult,
)


@dataclass(frozen=True)
class GuideResponse:
    """Guide-level response and input encoding retained for a measurement."""

    guide: str
    target: str
    response: np.ndarray
    input_vector: np.ndarray
    efficiency: float
    n_cells: int
    calibrated: bool = True
    efficiency_provenance: str = "same_assay_count_like"


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
    zero — the "dropout-driven pseudo-perfect knockdown" pathology. Pair with
    ``min_control_detection_rate`` in :func:`build_guide_responses` to filter
    information-limited targets before estimation. See MANUSCRIPT.md §2.3 and
    §3.2 and ``tutorial/02_efficiency_estimators.ipynb`` for the full estimator
    regime and when to use which.
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
    """Return the raw detection-probability shift `Pr[X_ctrl>0] − Pr[X_pert>0]`.

    NOT an unbiased estimator of `κ = 1 − E[X_pert]/E[X_ctrl]`. Under a Poisson
    model with `λ_pert = (1−κ)·λ_ctrl`, this quantity equals
    `exp(−(1−κ)λ_ctrl) − exp(−λ_ctrl)`, which is proportional to `κ` only in the
    low-`λ_ctrl` limit and saturates to 0 at high baseline. Preserved for
    backward compatibility and as a bounded regularizer against the dropout-
    driven 1.0-spike; prefer :func:`estimate_knockdown_efficiency_poisson_mle`
    for a `κ` estimate that remains valid across the baseline range.
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


def estimate_knockdown_efficiency_poisson_mle(
    expression: np.ndarray,
    *,
    target_index: int,
    perturbed_mask: np.ndarray,
    control_mask: np.ndarray,
    batches: np.ndarray | None = None,
    detection_floor: float = 1e-6,
    detection_ceiling: float = 1.0 - 1e-6,
) -> float:
    """Estimate `κ = 1 − λ_pert/λ_ctrl` via Poisson MLE from detection rates.

    Under the Poisson observation model `X ~ Poisson(λ)`,
    `Pr[X > 0] = 1 − exp(−λ)`, so `λ̂ = −log(1 − detection_rate)` is the moment
    estimator of `λ` from the binary indicator `X > 0`. This estimator
    - is unbiased for `κ` at low-to-moderate baseline expression (the regime
      where :func:`estimate_knockdown_efficiency` spikes to 1.0 due to dropout);
    - degrades gracefully at very low expression (where few cells detect) and
      near saturation (where `1 − detection_rate` approaches zero);
    - falls back to :func:`estimate_knockdown_efficiency` when control
      detection is outside `[detection_floor, detection_ceiling]`, i.e. when
      the detection moment is uninformative.

    Under zero-inflation independent of `λ`, this estimator is conservative:
    it attributes some structural zeros to Poisson, overstating `λ` and thus
    understating `κ`. The bias direction is deliberate — better to report a
    smaller `κ̂` than to spike to 1.0 on a lncRNA with 3 nonzero cells.
    """
    target_values = expression[:, target_index]
    if batches is None:
        ctrl_vals = target_values[control_mask]
        ctrl_det = float((ctrl_vals > 0).mean())
        ctrl_mean = float(ctrl_vals.mean())
    else:
        pert_batches = batches[perturbed_mask]
        total = 0
        det_sum = 0.0
        mean_sum = 0.0
        for batch in np.unique(pert_batches):
            n_perturbed = int(np.sum(pert_batches == batch))
            batch_controls = control_mask & (batches == batch)
            if not np.any(batch_controls):
                raise AnchorOpError(f"No matched control cells for batch {batch!r}.")
            batch_ctrl_vals = target_values[batch_controls]
            det_sum += n_perturbed * float((batch_ctrl_vals > 0).mean())
            mean_sum += n_perturbed * float(batch_ctrl_vals.mean())
            total += n_perturbed
        ctrl_det = det_sum / total
        ctrl_mean = mean_sum / total
    pert_vals = target_values[perturbed_mask]
    pert_det = float((pert_vals > 0).mean())
    pert_mean = float(pert_vals.mean())

    if ctrl_det < detection_floor or ctrl_det > detection_ceiling:
        if ctrl_mean <= 1e-8:
            return 0.0
        return float(np.clip(1.0 - pert_mean / ctrl_mean, 0.0, 1.0))

    lam_ctrl = -np.log(1.0 - ctrl_det)
    lam_pert = -np.log(max(1e-12, 1.0 - pert_det))
    if lam_ctrl <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - lam_pert / lam_ctrl, 0.0, 1.0))


_EFFICIENCY_ESTIMATORS = {
    "mean_ratio": estimate_knockdown_efficiency,
    "detection_rate": estimate_knockdown_efficiency_detection_rate,
    "poisson_mle": estimate_knockdown_efficiency_poisson_mle,
}


def _looks_pre_scaled(
    X: np.ndarray, *, negative_fraction_threshold: float = 0.02
) -> bool:
    """Return True when ``X`` looks like pre-scaled residuals (has meaningful
    negative values) rather than counts / normalized counts.

    A single stray negative from numerical noise should not trip this; a real
    residual matrix will have a substantial negative mass. Two percent of
    entries is far above numerical-noise thresholds and far below the ~50%
    negatives a z-score matrix typically has.
    """
    sample = X
    if sample.size > 200_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(sample.size, size=200_000, replace=False)
        sample = sample.ravel()[idx]
    negative_fraction = float((sample < 0).mean())
    return negative_fraction >= negative_fraction_threshold


def _resolve_estimator(name: str, X: np.ndarray) -> str:
    """Turn an ``efficiency_estimator`` argument into a concrete estimator key.

    Explicit choices pass through unchanged. ``"auto"`` inspects ``X`` and
    routes to ``"detection_rate"`` on pre-scaled residual matrices (where
    ``mean_ratio`` is undefined because controls are centered near zero) and
    to ``"mean_ratio"`` on count-like matrices (where the mean_ratio MLE is
    unbiased under Poisson and Poisson-with-dropout observation).
    """
    if name != "auto":
        return name
    if _looks_pre_scaled(X):
        return "detection_rate"
    return "mean_ratio"


def _require_paired_assay_alignment(
    response_obs: Any,
    calibration_obs: Any,
    *,
    guide_key: str,
    target_key: str | None,
    batch_key: str | None,
) -> None:
    """Refuse a raw/response pairing unless cells and key annotations match exactly."""
    if len(response_obs) != len(calibration_obs):
        raise AnchorOpError(
            "calibration_adata must have exactly the same number of cells as the response adata."
        )
    response_ids = tuple(map(str, response_obs.index))
    calibration_ids = tuple(map(str, calibration_obs.index))
    if response_ids != calibration_ids:
        raise AnchorOpError(
            "calibration_adata must have identical cell identifiers in identical order; "
            "do not pair independently filtered or reordered assays."
        )
    keys = [guide_key]
    if target_key is not None:
        keys.append(target_key)
    if batch_key is not None:
        keys.append(batch_key)
    for key in keys:
        if key not in response_obs.columns or key not in calibration_obs.columns:
            raise AnchorOpError(
                f"Paired calibration requires obs[{key!r}] in both response and calibration assays."
            )
        response_values = response_obs[key].astype(str).to_numpy()
        calibration_values = calibration_obs[key].astype(str).to_numpy()
        if not np.array_equal(response_values, calibration_values):
            raise AnchorOpError(
                f"Paired calibration requires identical obs[{key!r}] values in row order."
            )


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
    min_control_detection_rate: float = 0.05,
    loading_tol: float = 1e-8,
    efficiency_estimator: str = "auto",
    calibration_adata: Any | None = None,
    allow_proxy_efficiency: bool = False,
) -> tuple[list[GuideResponse], dict[str, str]]:
    """Estimate guide-level ``Δz`` and perturbation inputs from an AnnData-like input.

    The target transcript is used only to estimate guide efficacy. Its program
    encoding is the corresponding loading row, not a one-hot program vector.

    ``efficiency_estimator`` selects how per-guide efficiency is computed:

    - ``"auto"`` (default): inspect the expression matrix and route by format.
      Count-like data (all-nonnegative, or with a dropout-shaped zero mass) →
      ``"mean_ratio"``, which is the unbiased MLE of ``κ = 1 − E[X_pert]/
      E[X_ctrl]`` under Poisson and Poisson-with-dropout observation. Pre-
      scaled residuals (contain meaningful negative values, e.g. z-scored
      Perturb-seq h5ads such as Replogle 2022 essential-gene) → ``"detection_
      rate"``, which is a signed distributional-shift statistic on that data
      class (see below).
    - ``"mean_ratio"``: classical ``1 − mean_pert / mean_ctrl``. Sample-moment
      MLE under Poisson; still unbiased under independent zero-inflation
      (scRNA-seq dropout) because the dropout fraction cancels in the ratio.
      Undefined on data where the control mean is not bounded away from zero
      (pre-scaled residuals). Combine with ``min_control_detection_rate`` to
      drop information-limited targets rather than silently spiking to 0 or 1.
    - ``"poisson_mle"``: ``1 − λ̂_pert / λ̂_ctrl`` with ``λ̂ = −log(1 −
      detection)``. Equivalent to mean_ratio under pure Poisson; biased
      downward under independent zero-inflation. Included for completeness.
    - ``"detection_rate"``: the raw shift ``Pr[X_ctrl>0] − Pr[X_pert>0]``. On
      count data this is NOT an unbiased estimator of ``κ`` — it is a bounded
      shift diagnostic that scales with baseline expression. On pre-scaled
      residuals (where controls have mean ≈ 0 by construction), it recovers a
      valid signed distributional-shift statistic monotone in the perturbation
      shift ``Δ``: analytically ``0.5 − Φ(Δ/σ_ctrl)`` under a Gaussian
      approximation. This is what makes it the ``auto`` choice on pre-scaled
      data.

    ``min_control_detection_rate`` (default 0.05) drops any target whose
    control fraction of positive values is below the threshold. On count data
    this filters information-limited targets whose few nonzero controls make
    any estimator dominated by discretization noise.

    For signed normalized/residual response matrices, provide an exactly
    cell-aligned ``calibration_adata`` holding raw/count-like target expression.
    The response geometry is calculated from ``adata`` while κ is estimated only
    from ``calibration_adata``. Without it, this routine refuses inverse-model
    inputs unless ``allow_proxy_efficiency=True`` is set explicitly; proxy
    outputs are labelled uncalibrated and must not be used for biological
    operator claims.
    """
    if min_cells_per_guide < 1:
        raise AnchorOpError("min_cells_per_guide must be positive.")
    if not 0 <= min_knockdown_efficiency <= 1:
        raise AnchorOpError("min_knockdown_efficiency must lie in [0, 1].")
    if not 0 <= min_control_detection_rate <= 1:
        raise AnchorOpError("min_control_detection_rate must lie in [0, 1].")
    if efficiency_estimator != "auto" and efficiency_estimator not in _EFFICIENCY_ESTIMATORS:
        raise AnchorOpError(
            f"efficiency_estimator must be 'auto' or one of {sorted(_EFFICIENCY_ESTIMATORS)}; "
            f"got {efficiency_estimator!r}."
        )
    X, obs, gene_names = expression_and_metadata(adata)
    response_is_residual = _looks_pre_scaled(X)
    calibration_X = X
    calibration_obs = obs
    calibration_gene_names = gene_names
    calibrated = not response_is_residual
    efficiency_provenance = "same_assay_count_like"
    if calibration_adata is not None:
        calibration_X, calibration_obs, calibration_gene_names = expression_and_metadata(
            calibration_adata
        )
        _require_paired_assay_alignment(
            obs,
            calibration_obs,
            guide_key=guide_key,
            target_key=target_key,
            batch_key=batch_key,
        )
        if _looks_pre_scaled(calibration_X):
            raise AnchorOpError(
                "calibration_adata appears to contain signed normalized/residual values; "
                "raw/count-like target expression is required for quantitative κ calibration."
            )
        calibrated = True
        efficiency_provenance = "paired_raw_count_assay"
    elif response_is_residual and not allow_proxy_efficiency:
        raise AnchorOpError(
            "The response matrix appears signed normalized/residual-like and cannot calibrate κ. "
            "Provide an exactly paired raw/count-like calibration_adata, or set "
            "allow_proxy_efficiency=True only for explicitly uncalibrated exploratory outputs."
        )
    elif response_is_residual:
        calibrated = False
        efficiency_provenance = "uncalibrated_signed_response_proxy"

    resolved_estimator = _resolve_estimator(efficiency_estimator, calibration_X)
    if not calibrated:
        # A residual-space sign/detection statistic is descriptive only, never κ.
        resolved_estimator = "detection_rate"
    efficiency_fn = _EFFICIENCY_ESTIMATORS[resolved_estimator]
    if resolved_estimator == "detection_rate":
        min_control_detection_rate = 0.0
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
    response_gene_index = {gene: index for index, gene in enumerate(gene_names)}
    calibration_gene_index = {gene: index for index, gene in enumerate(calibration_gene_names)}
    z = project_expression(X, basis, gene_names=gene_names)
    responses: list[GuideResponse] = []

    for guide, target in targets_by_guide.items():
        perturbed = guides == guide
        n_cells = int(np.sum(perturbed))
        if n_cells < min_cells_per_guide:
            dropped[guide] = f"fewer_than_{min_cells_per_guide}_guide_positive_cells"
            continue
        if target not in response_gene_index:
            dropped[guide] = "target_gene_absent_from_response_matrix"
            continue
        if target not in calibration_gene_index:
            dropped[guide] = "target_gene_absent_from_calibration_matrix"
            continue
        target_col = calibration_X[:, calibration_gene_index[target]]
        ctrl_detection = float((target_col[controls] > 0).mean())
        if ctrl_detection < min_control_detection_rate:
            dropped[guide] = (
                f"control_detection_below_{min_control_detection_rate}_"
                f"info_limited_target"
            )
            continue
        try:
            matched_z = _matched_mean(
                z,
                perturbed_mask=perturbed,
                control_mask=controls,
                batches=batches,
            )
            efficiency = efficiency_fn(
                calibration_X,
                target_index=calibration_gene_index[target],
                perturbed_mask=perturbed,
                control_mask=controls,
                batches=batches,
            )
        except AnchorOpError as error:
            dropped[guide] = f"unmatched_controls: {error}"
            continue
        if calibrated and efficiency < min_knockdown_efficiency:
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
                calibrated=calibrated,
                efficiency_provenance=efficiency_provenance,
            )
        )
    if not responses:
        raise AnchorOpError("No informative perturbation guides remain after required filtering.")
    return responses, dropped


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Return a stable cosine similarity, or NaN for a zero-norm vector."""
    scale = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / scale) if scale > np.finfo(float).eps else float("nan")


def build_target_response_atlas(
    responses: Sequence[GuideResponse],
    *,
    minimum_guides: int = 1,
) -> TargetResponseAtlas:
    """Robustly aggregate guide-level response geometry at the target level.

    The returned atlas is deliberately a **descriptive** biological artifact.
    It does not invert a response matrix and does not assert that guide labels or
    proxy scores are scalar perturbation doses.  For each target, coordinatewise
    medians reduce the leverage of a discordant guide, while mean pairwise cosine
    exposes whether the guides agree on a shared response direction.
    """
    if minimum_guides < 1:
        raise AnchorOpError("minimum_guides must be positive.")
    groups: dict[str, list[GuideResponse]] = {}
    for record in responses:
        groups.setdefault(str(record.target), []).append(record)
    target_names: list[str] = []
    vectors: list[np.ndarray] = []
    n_guides: dict[str, int] = {}
    n_cells: dict[str, int] = {}
    concordance: dict[str, float] = {}
    guide_names: dict[str, tuple[str, ...]] = {}
    for target in sorted(groups):
        records = groups[target]
        if len(records) < minimum_guides:
            continue
        response_matrix = np.vstack([record.response for record in records])
        if not np.isfinite(response_matrix).all():
            raise AnchorOpError(f"Non-finite response encountered for target {target!r}.")
        pairwise = [
            _cosine(response_matrix[left], response_matrix[right])
            for left in range(len(records))
            for right in range(left + 1, len(records))
        ]
        target_names.append(target)
        vectors.append(np.median(response_matrix, axis=0))
        n_guides[target] = len(records)
        n_cells[target] = int(sum(record.n_cells for record in records))
        concordance[target] = float(np.nanmean(pairwise)) if pairwise else float("nan")
        guide_names[target] = tuple(record.guide for record in records)
    if not vectors:
        raise AnchorOpError("No targets satisfy the target-response atlas guide-count requirement.")
    return TargetResponseAtlas(
        target_names=tuple(target_names),
        responses=np.vstack(vectors),
        n_guides=n_guides,
        n_cells=n_cells,
        mean_pairwise_cosine=concordance,
        guide_names_by_target=guide_names,
        response_representation="matched_program_response",
        calibrated=all(record.calibrated for record in responses),
        notes=(
            "Target vectors are coordinatewise medians of retained guide responses.",
            "This atlas is descriptive and is not a biological operator or Jacobian estimate.",
        ),
    )


def within_target_dose_response(
    responses: Sequence[GuideResponse],
    *,
    minimum_guides: int = 3,
) -> DoseResponseResult:
    """Test whether calibrated guide strengths track a shared target response.

    This diagnostic is intentionally unavailable for residual-space proxy
    efficiencies. A response-atlas should be used instead when raw/count-like
    calibration is not paired to the response assay.
    """
    if minimum_guides < 3:
        raise AnchorOpError("minimum_guides must be at least three for a dose-response check.")
    if any(not record.calibrated for record in responses):
        raise AnchorOpError(
            "Within-target dose-response requires raw/count-calibrated guide efficiencies; "
            "uncalibrated proxy responses cannot establish a guide-dose relationship."
        )
    groups: dict[str, list[GuideResponse]] = {}
    for record in responses:
        groups.setdefault(str(record.target), []).append(record)
    names: list[str] = []
    r_squared: dict[str, float] = {}
    mean_cosine: dict[str, float] = {}
    n_guides: dict[str, int] = {}
    for target in sorted(groups):
        records = groups[target]
        if len(records) < minimum_guides:
            continue
        efficiencies = np.asarray([record.efficiency for record in records], dtype=float)
        norms = np.asarray([np.linalg.norm(record.response) for record in records], dtype=float)
        if np.ptp(efficiencies) <= np.finfo(float).eps or np.ptp(norms) <= np.finfo(float).eps:
            score = float("nan")
        else:
            coefficients = np.polyfit(efficiencies, norms, deg=1)
            fitted = np.polyval(coefficients, efficiencies)
            total = float(np.sum((norms - norms.mean()) ** 2))
            score = float(1.0 - np.sum((norms - fitted) ** 2) / total)
        vectors = np.vstack([record.response for record in records])
        pairwise = [
            _cosine(vectors[left], vectors[right])
            for left in range(len(records))
            for right in range(left + 1, len(records))
        ]
        names.append(target)
        r_squared[target] = score
        mean_cosine[target] = float(np.nanmean(pairwise)) if pairwise else float("nan")
        n_guides[target] = len(records)
    return DoseResponseResult(
        target_names=tuple(names),
        r_squared_by_target=r_squared,
        response_cosine_by_target=mean_cosine,
        n_guides_by_target=n_guides,
        notes=(
            "R-squared describes guide κ versus response-norm association within target.",
            "It is a necessary assay/model diagnostic, not by itself an operator-validity result.",
        ),
    )


def _aggregate_replicate_guides(responses: Sequence[GuideResponse]) -> list[GuideResponse]:
    """Collapse same-target guide replicates using robust medians before inversion."""
    groups: dict[str, list[GuideResponse]] = {}
    for record in responses:
        groups.setdefault(str(record.target), []).append(record)
    aggregated: list[GuideResponse] = []
    for target in sorted(groups):
        records = groups[target]
        # Preserve historical guide-level identity where no replicate guide
        # exists; collapse only genuinely duplicated target interventions.
        if len(records) == 1:
            aggregated.append(records[0])
            continue
        aggregated.append(
            GuideResponse(
                guide=target,
                target=target,
                response=np.median(np.vstack([record.response for record in records]), axis=0),
                input_vector=np.median(np.vstack([record.input_vector for record in records]), axis=0),
                efficiency=float(np.median([record.efficiency for record in records])),
                n_cells=int(sum(record.n_cells for record in records)),
                calibrated=all(record.calibrated for record in records),
                efficiency_provenance="target_median_of_" + "+".join(
                    sorted({record.efficiency_provenance for record in records})
                ),
            )
        )
    return aggregated


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
    guide_targets: Mapping[str, str] | None = None,
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
        guide_targets={} if guide_targets is None else guide_targets,
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
    min_control_detection_rate: float = 0.05,
    loading_tol: float = 1e-8,
    reg: str = "tsvd",
    reg_param: str | float | int = "path",
    rank_tol: float = 1e-2,
    bootstrap: int = 0,
    bootstrap_seed: int | None = 0,
    state_label: str | None = None,
    efficiency_estimator: str = "auto",
    calibration_adata: Any | None = None,
    aggregate_replicate_guides: bool = True,
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

    ``efficiency_estimator`` defaults to ``"auto"`` and estimates κ from
    count-like expression. A signed normalized/residual response matrix cannot
    provide a quantitative κ. For such data, pass an exactly cell-aligned raw
    or count-like ``calibration_adata``; otherwise this inverse-model function
    refuses to return an operator. Use :func:`build_target_response_atlas` for
    a descriptive response result when only normalized/residual data exist.

    ``aggregate_replicate_guides=True`` (default) robustly collapses guides
    assigned to the same target using coordinatewise median responses and
    inputs before inversion. This prevents several collinear guides for one
    target from receiving disproportionate leverage in the operator fit.
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
        min_control_detection_rate=min_control_detection_rate,
        loading_tol=loading_tol,
        efficiency_estimator=efficiency_estimator,
        calibration_adata=calibration_adata,
    )
    if any(not record.calibrated for record in responses):
        raise AnchorOpError(
            "Uncalibrated response proxies cannot be inverted into a biological operator. "
            "Provide paired raw/count calibration data or use build_target_response_atlas."
        )
    calibration_probe = adata if calibration_adata is None else calibration_adata
    X_probe, _, _ = expression_and_metadata(calibration_probe)
    resolved_estimator = _resolve_estimator(efficiency_estimator, X_probe)
    if efficiency_estimator == "auto":
        estimator_note = (
            f"Knockdown efficiency was estimated per target via '{resolved_estimator}' "
            f"(auto-routed from data format)."
        )
    else:
        estimator_note = (
            f"Knockdown efficiency was estimated per target via '{resolved_estimator}'."
        )
    original_guide_targets = {record.guide: record.target for record in responses}
    if aggregate_replicate_guides:
        responses = _aggregate_replicate_guides(responses)
    guide_names = tuple(record.guide for record in responses)
    S = np.column_stack([record.response for record in responses])
    U = np.column_stack([record.input_vector for record in responses])
    efficiencies = {record.guide: record.efficiency for record in responses}
    return measure_from_sensitivity(
        S,
        U,
        guide_names=guide_names,
        guide_efficiencies=efficiencies,
        guide_targets=original_guide_targets,
        dropped_guides=dropped,
        reg=reg,
        reg_param=reg_param,
        rank_tol=rank_tol,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
        state_label=state_label,
        notes=(
            "Guide-level S and U were estimated from matched perturbation and control means.",
            "Replicate guides were robustly aggregated by target before inversion."
            if aggregate_replicate_guides
            else "Guide-level inputs were retained without target aggregation.",
            "κ calibration used a paired raw/count-like assay."
            if calibration_adata is not None
            else "κ calibration used the same count-like assay as response geometry.",
            estimator_note,
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

    # `passed` always uses the preregistered raw-diff criterion. When null
    # correction is requested (n_null > 0), the null_median / null_std /
    # z_score / excess_above_null fields are informative diagnostics for
    # interpreting the failure, NOT a substitute pass/fail criterion — the
    # 0.25 threshold in PREREGISTRATION.md is locked to the raw statistic and
    # does not transfer to a different quantity. See MANUSCRIPT.md §3.5.
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


def held_out_prediction_check(
    measured: MeasuredOperator,
    *,
    n_folds: int = 5,
    seed: int | None = 0,
    n_permutation_null: int = 0,
    null_seed: int | None = 100,
) -> HeldOutPredictionResult:
    """Out-of-sample linearity diagnostic: fit ``A`` on train guides, evaluate residual on held-out.

    From the identity ``J·S_g = -U_g`` (which holds for every guide g under
    the additive-input linear settled-state model), fit
    ``A = J·P_X = -U_train · S_train⁺`` on 4/5 of the guides and evaluate the
    pooled residual ``ρ = ||A·S_test + U_test||_F / ||U_test||_F`` on the
    held-out fifth. Under perfect linearity + noise-free measurement,
    ``ρ → 0``; under a naive zero-predictor, ``ρ = 1``.

    Unlike :func:`linearity_check`'s median-split diagnostic, this metric is
    **algebraically invariant to global rescaling of** ``U``. Under
    per-column κ rescaling it is near-invariant only in the zero-predictor
    regime; outside that regime it is not invariant. Both diagnostics are
    noise-limited at published Perturb-seq scale (d≈30, n≈200, per-guide σ≈0.27)
    — interpret observed values against a matched-scale synthetic linear
    positive control per MANUSCRIPT.md §3.5–§3.6 and Fig. S10. The two
    diagnostics are complementary: bin-split rel_diff surfaces gross model
    disagreement in projected coordinates; held-out ρ tests whether the
    fitted operator generalizes beyond the training guides.

    ``n_permutation_null > 0`` triggers a shuffled-U↔S null distribution
    (each permutation shuffles the U column labels, refits, and reports ρ).
    The z-score against this null quantifies how much better the fit does
    than a random U↔S correspondence on the same S and U matrices.
    """
    report = measured.report
    if report is None:
        raise AnchorOpError("A report is required for the held-out prediction check.")
    S = measured.S
    U = measured.U
    d, m = S.shape
    if n_folds < 2:
        raise AnchorOpError("n_folds must be at least 2.")
    if m < n_folds:
        raise AnchorOpError(
            f"n_folds ({n_folds}) exceeds guide count ({m}); reduce n_folds."
        )
    rank_tol = report.rank_tol
    method = report.regularization_method

    rng = stable_rng(seed)
    fold_ids = rng.integers(0, n_folds, size=m)

    residual_sq_sum = 0.0
    target_sq_sum = 0.0
    per_fold: list[float] = []
    n_folds_used = 0
    for k in range(n_folds):
        train = fold_ids != k
        test = fold_ids == k
        if int(train.sum()) < d or int(test.sum()) == 0:
            continue
        S_pinv, _selected, _path, _proj = regularized_pseudoinverse(
            S[:, train], method=method, parameter="path", rank_tol=rank_tol
        )
        A = -U[:, train] @ S_pinv
        pred = A @ S[:, test]
        residual = pred + U[:, test]
        residual_norm = float(np.linalg.norm(residual))
        target_norm = float(np.linalg.norm(U[:, test]))
        residual_sq_sum += residual_norm ** 2
        target_sq_sum += target_norm ** 2
        per_fold.append(residual_norm / max(target_norm, 1e-12))
        n_folds_used += 1

    if n_folds_used == 0:
        raise AnchorOpError(
            "No fold satisfied n_train >= d and n_test > 0; increase guide count or decrease d."
        )
    rho_pooled = float(np.sqrt(residual_sq_sum) / max(np.sqrt(target_sq_sum), 1e-12))

    null_median = null_std = z = None
    if n_permutation_null > 0:
        null_rng = stable_rng(null_seed)
        null_vals: list[float] = []
        for _ in range(n_permutation_null):
            perm = null_rng.permutation(m)
            U_perm = U[:, perm]
            fold_ids_p = null_rng.integers(0, n_folds, size=m)
            r_sq = 0.0
            t_sq = 0.0
            any_fold = False
            for k in range(n_folds):
                train_p = fold_ids_p != k
                test_p = fold_ids_p == k
                if int(train_p.sum()) < d or int(test_p.sum()) == 0:
                    continue
                S_pinv, *_ = regularized_pseudoinverse(
                    S[:, train_p], method=method, parameter="path", rank_tol=rank_tol
                )
                A_p = -U_perm[:, train_p] @ S_pinv
                pred_p = A_p @ S[:, test_p]
                res_p = pred_p + U_perm[:, test_p]
                r_sq += float(np.linalg.norm(res_p)) ** 2
                t_sq += float(np.linalg.norm(U_perm[:, test_p])) ** 2
                any_fold = True
            if any_fold and t_sq > 0:
                null_vals.append(float(np.sqrt(r_sq) / np.sqrt(t_sq)))
        if null_vals:
            arr = np.asarray(null_vals, dtype=float)
            null_median = float(np.median(arr))
            null_std = float(arr.std())
            z = float((rho_pooled - arr.mean()) / max(arr.std(), 1e-12))

    return HeldOutPredictionResult(
        rho_pooled=rho_pooled,
        rho_per_fold=tuple(per_fold),
        n_folds_used=n_folds_used,
        n_folds_requested=n_folds,
        null_median=null_median,
        null_std=null_std,
        z_score=z,
        n_permutation_null=n_permutation_null,
    )
