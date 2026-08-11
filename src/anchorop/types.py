"""Public data structures for :mod:`anchorop`.

The central safety invariant is structural: a matrix action has no public access
without an :class:`AnchorReport`, and a full Jacobian is unavailable unless the
measured response subspace spans the requested program space.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class AnchorOpError(ValueError):
    """Base exception for actionable input or analysis errors."""


class IdentifiabilityError(AnchorOpError):
    """Raised when an operation would overstate what a measurement identifies."""


class DimensionGuardError(AnchorOpError):
    """Raised when a gene-level or excessively large program space is requested."""


@dataclass(frozen=True)
class ProgramBasis:
    """A control-derived program loading matrix.

    Parameters
    ----------
    loadings:
        Gene-by-program loading matrix with shape ``(n_genes, d)``.
    gene_names:
        Gene names aligned to the first axis of ``loadings``.
    method:
        Basis provenance, for example ``"nmf"``, ``"cnmf"``, or ``"external"``.
    control_count:
        Number of control cells used to fit or validate the basis.
    seed_concordance:
        Optional mean matched-component concordance across a cNMF seed ensemble.
    metadata:
        Additional reproducibility metadata.
    """

    loadings: np.ndarray
    gene_names: tuple[str, ...]
    method: str
    control_count: int
    seed_concordance: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        loadings = np.asarray(self.loadings, dtype=float)
        if loadings.ndim != 2:
            raise AnchorOpError(
                "ProgramBasis.loadings must be a two-dimensional gene-by-program matrix."
            )
        if loadings.shape[0] != len(self.gene_names):
            raise AnchorOpError("ProgramBasis.gene_names must align with the loading matrix rows.")
        if loadings.shape[1] < 1:
            raise AnchorOpError("ProgramBasis must contain at least one program.")
        if not np.isfinite(loadings).all():
            raise AnchorOpError("ProgramBasis.loadings contains non-finite values.")
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "gene_names", tuple(map(str, self.gene_names)))

    @property
    def d(self) -> int:
        """Number of programs."""
        return int(self.loadings.shape[1])

    @property
    def n_genes(self) -> int:
        """Number of genes represented by the basis."""
        return int(self.loadings.shape[0])


@dataclass(frozen=True)
class RegularizationPathEntry:
    """Diagnostics for one pseudo-inverse setting."""

    method: str
    parameter: float | int
    effective_rank: int
    retained_mask: np.ndarray
    singular_values: np.ndarray
    filter_factors: np.ndarray
    condition_number: float

    def __post_init__(self) -> None:
        for name in ("retained_mask", "singular_values", "filter_factors"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1:
                raise AnchorOpError(f"RegularizationPathEntry.{name} must be one-dimensional.")
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class AnchorReport:
    """Mandatory identifiability and numerical-stability disclosure.

    ``response_projector`` projects onto ``range(S)``. The measured action is
    identified only on that domain. ``input_projector`` is retained separately
    because it describes the perturbation directions, not the identified domain.
    """

    d: int
    input_subspace_dim: int
    response_subspace_dim: int
    effective_response_rank: int
    n_guides_input: int
    n_guides_retained: int
    retained_guides: tuple[str, ...]
    dropped_guides: Mapping[str, str]
    input_projector: np.ndarray
    response_projector: np.ndarray
    singular_values: np.ndarray
    retained_singular_directions: np.ndarray
    condition_number: float
    regularization_method: str
    selected_regularization: float | int
    regularization_path: tuple[RegularizationPathEntry, ...]
    full_domain_identified: bool
    rank_tol: float | None = None
    guide_efficiencies: Mapping[str, float] = field(default_factory=dict)
    guide_targets: Mapping[str, str] = field(default_factory=dict)
    bootstrap_covariance: np.ndarray | None = None
    bootstrap_actions: np.ndarray | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.d < 1:
            raise AnchorOpError("AnchorReport.d must be positive.")
        for name in ("input_projector", "response_projector"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (self.d, self.d):
                raise AnchorOpError(f"AnchorReport.{name} must have shape (d, d).")
            object.__setattr__(self, name, value.copy())
        singular_values = np.asarray(self.singular_values, dtype=float)
        retained = np.asarray(self.retained_singular_directions, dtype=bool)
        if singular_values.ndim != 1 or retained.shape != singular_values.shape:
            raise AnchorOpError(
                "Singular values and retained-direction flags must be aligned vectors."
            )
        object.__setattr__(self, "singular_values", singular_values.copy())
        object.__setattr__(self, "retained_singular_directions", retained.copy())
        if self.bootstrap_covariance is not None:
            covariance = np.asarray(self.bootstrap_covariance, dtype=float)
            if covariance.shape != (self.d * self.d, self.d * self.d):
                raise AnchorOpError("bootstrap_covariance must have shape (d*d, d*d).")
            object.__setattr__(self, "bootstrap_covariance", covariance.copy())
        if self.bootstrap_actions is not None:
            actions = np.asarray(self.bootstrap_actions, dtype=float)
            if actions.ndim != 3 or actions.shape[1:] != (self.d, self.d):
                raise AnchorOpError("bootstrap_actions must have shape (n_bootstrap, d, d).")
            object.__setattr__(self, "bootstrap_actions", actions.copy())
        object.__setattr__(self, "retained_guides", tuple(map(str, self.retained_guides)))
        object.__setattr__(self, "dropped_guides", dict(self.dropped_guides))
        object.__setattr__(self, "guide_efficiencies", dict(self.guide_efficiencies))
        object.__setattr__(
            self, "guide_targets", {str(key): str(value) for key, value in self.guide_targets.items()}
        )
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def unconstrained_dimension(self) -> int:
        """Program-space dimensions outside the effective response domain."""
        return self.d - self.effective_response_rank

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary, retaining numerical disclosures."""
        return {
            "d": self.d,
            "input_subspace_dim": self.input_subspace_dim,
            "response_subspace_dim": self.response_subspace_dim,
            "effective_response_rank": self.effective_response_rank,
            "unconstrained_dimension": self.unconstrained_dimension,
            "n_guides_input": self.n_guides_input,
            "n_guides_retained": self.n_guides_retained,
            "retained_guides": list(self.retained_guides),
            "dropped_guides": dict(self.dropped_guides),
            "condition_number": float(self.condition_number),
            "regularization_method": self.regularization_method,
            "selected_regularization": self.selected_regularization,
            "full_domain_identified": self.full_domain_identified,
            "rank_tol": self.rank_tol,
            "singular_values": self.singular_values.tolist(),
            "retained_singular_directions": self.retained_singular_directions.tolist(),
            "guide_efficiencies": dict(self.guide_efficiencies),
            "guide_targets": dict(self.guide_targets),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MeasuredOperator:
    """A measured Jacobian action with its inseparable identifiability report."""

    _identified_action: np.ndarray
    S: np.ndarray
    U: np.ndarray
    report: AnchorReport | None
    guide_names: tuple[str, ...]
    state_label: str | None = None

    def __post_init__(self) -> None:
        action = np.asarray(self._identified_action, dtype=float)
        S = np.asarray(self.S, dtype=float)
        U = np.asarray(self.U, dtype=float)
        if action.ndim != 2 or action.shape[0] != action.shape[1]:
            raise AnchorOpError("The identified action must be a square matrix.")
        d = action.shape[0]
        if S.ndim != 2 or U.ndim != 2 or S.shape != U.shape or S.shape[0] != d:
            raise AnchorOpError("S and U must be aligned d-by-m matrices for the measured action.")
        if len(self.guide_names) != S.shape[1]:
            raise AnchorOpError("guide_names must align with S and U columns.")
        if self.report is not None and self.report.d != d:
            raise AnchorOpError("AnchorReport.d must match the identified action dimension.")
        object.__setattr__(self, "_identified_action", action.copy())
        object.__setattr__(self, "S", S.copy())
        object.__setattr__(self, "U", U.copy())
        object.__setattr__(self, "guide_names", tuple(map(str, self.guide_names)))

    def _require_report(self) -> AnchorReport:
        if self.report is None:
            raise IdentifiabilityError(
                "No matrix action may be accessed without an AnchorReport. "
                "Construct the measurement through measure_operator()."
            )
        return self.report

    @property
    def identified_action(self) -> np.ndarray:
        """Return the experimentally identified map ``J P_X`` with its report verified."""
        self._require_report()
        return self._identified_action.copy()

    @property
    def J(self) -> np.ndarray:
        """Return a full Jacobian only after full effective-domain identification.

        A partial zero-extended action has no generally interpretable spectrum, so
        this property rejects it rather than allowing silent overinterpretation.
        """
        report = self._require_report()
        if not report.full_domain_identified:
            raise IdentifiabilityError(
                "A full Jacobian is not identified: only J P_X is measured. "
                "Use .identified_action and projected comparison metrics, or obtain full effective rank."
            )
        return self._identified_action.copy()

    @property
    def response_projector(self) -> np.ndarray:
        """Projector onto the experimentally identified response/domain subspace."""
        return self._require_report().response_projector.copy()

    @property
    def input_projector(self) -> np.ndarray:
        """Projector onto the actuated perturbation-input subspace."""
        return self._require_report().input_projector.copy()

    @property
    def M(self) -> np.ndarray:
        """Backward-compatible alias for the identified response-domain projector.

        Earlier design text called this space ``M``. New code should use
        ``response_projector`` to avoid confusing it with ``range(U)``.
        """
        return self.response_projector


@dataclass(frozen=True)
class ComparisonResult:
    """Results for one inferred operator versus one measured action."""

    method: str
    metrics: Mapping[str, float]
    null_metrics: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metrics", {key: float(value) for key, value in self.metrics.items()}
        )
        object.__setattr__(
            self,
            "null_metrics",
            {
                key: np.asarray(value, dtype=float).copy()
                for key, value in self.null_metrics.items()
            },
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ArchetypeResult:
    """Simplex-constrained archetypal decomposition result."""

    mode: str
    archetypes: np.ndarray
    weights: np.ndarray
    reconstruction_error: float
    selected_k: int
    candidate_errors: Mapping[int, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        archetypes = np.asarray(self.archetypes, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if archetypes.ndim < 2 or weights.ndim != 2:
            raise AnchorOpError("Archetypes and weights have incompatible dimensions.")
        if archetypes.shape[0] != weights.shape[1]:
            raise AnchorOpError("The number of archetypes must equal the weight columns.")
        object.__setattr__(self, "archetypes", archetypes.copy())
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(
            self, "candidate_errors", {int(k): float(v) for k, v in self.candidate_errors.items()}
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TransferResult:
    """Out-of-state projection diagnostics for operator archetypes."""

    weights: np.ndarray
    transfer_error: float
    refit_error: float
    error_ratio: float

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 2:
            raise AnchorOpError("Transfer weights must be a two-dimensional simplex matrix.")
        object.__setattr__(self, "weights", weights.copy())


@dataclass(frozen=True)
class LinearityResult:
    """Weak/strong knockdown-bin diagnostic for the local linearity assumption.

    ``passed`` uses **only** the preregistered raw-diff criterion
    (``relative_difference ≤ threshold``). The 0.25 threshold in
    ``PREREGISTRATION.md`` is locked to the raw statistic and does not
    transfer to derived quantities.

    When a random-split null is supplied (``n_null > 0`` passed to
    :func:`anchorop.linearity_check`), the ``null_median``, ``null_std``,
    ``null_p95``, ``excess_above_null``, and ``z_score`` fields characterize
    the bin-composition floor of the diagnostic itself: different guide
    subsets sample different columns of J and identify different sub-operators
    even under perfect linearity, so the naive raw-diff has an irreducible
    floor. These fields are **informative diagnostics** for interpreting a
    raw-diff failure — not a substitute pass/fail criterion. See
    ``MANUSCRIPT.md`` §3.5 for the framing and dynamic-range discussion.

    ``excess_above_null`` is ``relative_difference - null_median``. It is the
    part of the observed disagreement that a random 50/50 split of the same
    guide set would *not* reproduce.
    """

    weak_action: np.ndarray
    strong_action: np.ndarray
    relative_difference: float
    weak_guides: tuple[str, ...]
    strong_guides: tuple[str, ...]
    passed: bool
    threshold: float
    overlap_rank: int
    null_median: float | None = None
    null_std: float | None = None
    null_p95: float | None = None
    excess_above_null: float | None = None
    z_score: float | None = None
    n_null: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "weak_action", np.asarray(self.weak_action, dtype=float).copy())
        object.__setattr__(
            self, "strong_action", np.asarray(self.strong_action, dtype=float).copy()
        )
        object.__setattr__(self, "weak_guides", tuple(map(str, self.weak_guides)))
        object.__setattr__(self, "strong_guides", tuple(map(str, self.strong_guides)))


@dataclass(frozen=True)
class HeldOutPredictionResult:
    """Out-of-sample linearity diagnostic invariant to κ column scaling.

    From the identity ``J·S_g = -U_g`` for every guide g under the additive-
    input linear settled-state model, fit ``A = J·P_X`` on train guides and
    evaluate the residual ``ρ = ||A·S_test + U_test||_F / ||U_test||_F`` on
    held-out guides (k-fold CV). Under perfect linearity + noise-free
    measurement, ``ρ → 0``; under a naive zero-predictor, ``ρ = 1``.

    Unlike :class:`LinearityResult`'s median-split diagnostic, this metric is
    algebraically invariant to global rescaling of ``U``. It is near-invariant
    to per-column κ rescaling only in the zero-predictor regime; outside that
    regime it is not invariant. Both diagnostics are noise-limited at
    published Perturb-seq scale (d≈30, n≈200, per-guide σ≈0.27) — interpret
    observed values against a matched-scale synthetic linear positive control
    per MANUSCRIPT.md §3.5–§3.6 and Fig. S10.

    When ``n_permutation_null > 0`` is passed, a shuffled-U↔S null
    distribution is reported to calibrate ``ρ`` against random column
    permutations of the same S and U matrices.
    """

    rho_pooled: float
    rho_per_fold: tuple[float, ...]
    n_folds_used: int
    n_folds_requested: int
    null_median: float | None = None
    null_std: float | None = None
    z_score: float | None = None
    n_permutation_null: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho_per_fold", tuple(map(float, self.rho_per_fold)))


@dataclass(frozen=True)
class TargetResponseAtlas:
    """A descriptive, target-level response artifact for biological screens.

    This object is intentionally **not** a Jacobian or inverse-operator estimate.
    It records robust consensus response vectors across guides that share a target,
    together with guide-concordance diagnostics.  It is the appropriate public
    output when the screen supplies response geometry but not a paired raw/count
    assay that can calibrate target knockdown strengths.
    """

    target_names: tuple[str, ...]
    responses: np.ndarray
    n_guides: Mapping[str, int]
    n_cells: Mapping[str, int]
    mean_pairwise_cosine: Mapping[str, float]
    guide_names_by_target: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    response_representation: str = "descriptive"
    calibrated: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        responses = np.asarray(self.responses, dtype=float)
        if responses.ndim != 2:
            raise AnchorOpError("TargetResponseAtlas.responses must be target-by-program.")
        names = tuple(map(str, self.target_names))
        if responses.shape[0] != len(names):
            raise AnchorOpError("TargetResponseAtlas.target_names must align with response rows.")
        if len(set(names)) != len(names):
            raise AnchorOpError("TargetResponseAtlas.target_names must be unique.")
        if not np.isfinite(responses).all():
            raise AnchorOpError("TargetResponseAtlas.responses contains non-finite values.")
        object.__setattr__(self, "target_names", names)
        object.__setattr__(self, "responses", responses.copy())
        object.__setattr__(self, "n_guides", {str(key): int(value) for key, value in self.n_guides.items()})
        object.__setattr__(self, "n_cells", {str(key): int(value) for key, value in self.n_cells.items()})
        object.__setattr__(
            self,
            "mean_pairwise_cosine",
            {str(key): float(value) for key, value in self.mean_pairwise_cosine.items()},
        )
        object.__setattr__(
            self,
            "guide_names_by_target",
            {
                str(key): tuple(map(str, value))
                for key, value in self.guide_names_by_target.items()
            },
        )
        object.__setattr__(self, "notes", tuple(map(str, self.notes)))

    @property
    def d(self) -> int:
        """Number of response-coordinate dimensions."""
        return int(self.responses.shape[1])


@dataclass(frozen=True)
class DoseResponseResult:
    """Within-target guide efficiency-versus-response diagnostic.

    A target contributes only when it has at least three independently calibrated
    guides.  ``r_squared_by_target`` describes a simple linear relationship
    between calibrated raw-count knockdown strength and response magnitude; it
    is an assay/model diagnostic, not evidence for a system-wide operator.
    """

    target_names: tuple[str, ...]
    r_squared_by_target: Mapping[str, float]
    response_cosine_by_target: Mapping[str, float]
    n_guides_by_target: Mapping[str, int]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(map(str, self.target_names))
        object.__setattr__(self, "target_names", names)
        object.__setattr__(
            self,
            "r_squared_by_target",
            {str(key): float(value) for key, value in self.r_squared_by_target.items()},
        )
        object.__setattr__(
            self,
            "response_cosine_by_target",
            {str(key): float(value) for key, value in self.response_cosine_by_target.items()},
        )
        object.__setattr__(
            self,
            "n_guides_by_target",
            {str(key): int(value) for key, value in self.n_guides_by_target.items()},
        )
        object.__setattr__(self, "notes", tuple(map(str, self.notes)))
