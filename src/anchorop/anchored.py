"""Phase-gated anchored inference interface.

This module intentionally does not fit a constrained biological Jacobian until a
pre-registered Phase 2 benchmark establishes above-null agreement on held-out
perturbations. Building anchoring before that evidence would steer the project
toward a desired outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import AnchorOpError


def verify_phase2_gate(evidence_file: str | Path) -> dict[str, Any]:
    """Validate an immutable Phase 2 evidence record before anchoring is enabled.

    The JSON record must state a preregistration reference, a benchmark commit,
    held-out perturbation performance, and an above-null decision. This function
    validates provenance only; it does not decide scientific significance.
    """
    path = Path(evidence_file)
    if not path.exists():
        raise AnchorOpError(
            "Anchored inference is gated on Phase 2. No evidence record was found; run and freeze the benchmark first."
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AnchorOpError(f"Phase 2 evidence file is not valid JSON: {path}") from error
    required = {
        "preregistration_commit",
        "benchmark_commit",
        "heldout_metric",
        "null_metric",
        "above_null",
    }
    missing = sorted(required - set(record))
    if missing:
        raise AnchorOpError(
            f"Phase 2 evidence record is incomplete; missing: {', '.join(missing)}."
        )
    if record["above_null"] is not True:
        raise AnchorOpError(
            "Phase 2 did not establish above-null signal. Per the preregistered design, anchored inference remains disabled."
        )
    return record


def fit_anchored_operator(
    *args: Any, evidence_file: str | Path | None = None, **kwargs: Any
) -> None:
    """Reserve the public API while refusing premature anchored inference.

    Once a completed, independently reviewed Phase 2 evidence record is present,
    this function will be implemented through a separate release and validation
    protocol. It intentionally cannot be activated by a convenience flag.
    """
    if evidence_file is None:
        raise AnchorOpError(
            "fit_anchored_operator is unavailable before the Phase 2 evidence gate. "
            "Provide an audited evidence_file only after the benchmark is complete."
        )
    verify_phase2_gate(evidence_file)
    raise NotImplementedError(
        "The Phase 2 gate is recorded, but constrained inference is deferred to a separately validated Phase 4 release."
    )
