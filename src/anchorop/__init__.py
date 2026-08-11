"""anchor-op: measured, identifiable dynamical actions from Perturb-seq.

`S` is measured from guide-level responses. The package estimates `J P_X`, a
regularized action on an explicitly reported response subspace; it does not
silently return an unqualified cellular Jacobian.
"""

from . import analyses, plotting
from .replogle import load_replogle_h5ad
from .archetypes import fit_archetypes, spectral_summary, transfer_test
from .compare import (
    compare,
    comparison_table,
    hyperbolicity_sign,
    spectral_abscissa_difference,
    spectral_wasserstein,
)
from .identifiability import regularization_path, regularized_pseudoinverse
from .io import load_operator, validate_operator
from .measure import (
    build_guide_responses,
    build_target_response_atlas,
    within_target_dose_response,
    estimate_knockdown_efficiency,
    estimate_knockdown_efficiency_detection_rate,
    estimate_knockdown_efficiency_poisson_mle,
    held_out_prediction_check,
    linearity_check,
    measure_from_sensitivity,
    measure_operator,
)
from .programs import fit_programs, make_program_basis, project_expression
from .types import (
    AnchorOpError,
    AnchorReport,
    ArchetypeResult,
    ComparisonResult,
    DimensionGuardError,
    HeldOutPredictionResult,
    IdentifiabilityError,
    LinearityResult,
    MeasuredOperator,
    ProgramBasis,
    TransferResult,
    TargetResponseAtlas,
    DoseResponseResult,
)

__all__ = [
    "analyses",
    "plotting",
    "AnchorOpError",
    "AnchorReport",
    "ArchetypeResult",
    "ComparisonResult",
    "DimensionGuardError",
    "HeldOutPredictionResult",
    "IdentifiabilityError",
    "LinearityResult",
    "MeasuredOperator",
    "ProgramBasis",
    "TransferResult",
    "TargetResponseAtlas",
    "DoseResponseResult",
    "build_guide_responses",
    "build_target_response_atlas",
    "compare",
    "comparison_table",
    "estimate_knockdown_efficiency",
    "estimate_knockdown_efficiency_detection_rate",
    "estimate_knockdown_efficiency_poisson_mle",
    "fit_archetypes",
    "fit_programs",
    "held_out_prediction_check",
    "hyperbolicity_sign",
    "linearity_check",
    "load_operator",
    "load_replogle_h5ad",
    "make_program_basis",
    "measure_from_sensitivity",
    "measure_operator",
    "project_expression",
    "regularization_path",
    "regularized_pseudoinverse",
    "spectral_abscissa_difference",
    "spectral_summary",
    "spectral_wasserstein",
    "transfer_test",
    "within_target_dose_response",
    "validate_operator",
]

__version__ = "0.1.0"
