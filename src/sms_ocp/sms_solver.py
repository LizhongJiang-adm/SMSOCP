"""Public solve entry points for complete and single SMS-IA problems."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.nlp import ExplicitNlp, solve_nlp
from sms_ocp.optimal_control_problem import OptimalControlProblem
from sms_ocp.sms_algorithm import (
    SMSAlgorithmOptions,
    SMSAlgorithmResult,
    run_sms_algorithm,
)
from sms_ocp.sms_ia_transcription import SMSIAOptions, SMSIATranscription
from sms_ocp.sms_kkt import SMSKKTOptions


def solve_sms_ocp(
    ocp: OptimalControlProblem,
    shooting_grid: ArrayLike,
    initial_guess: ArrayLike,
    *,
    sms_ia_options: SMSIAOptions | None = None,
    algorithm_options: SMSAlgorithmOptions | None = None,
    kkt_options: SMSKKTOptions | None = None,
    integrator_options: Mapping[str, object] | None = None,
    solver_options: Mapping[str, object] | None = None,
) -> SMSAlgorithmResult:
    """Transcribe an OCP and run the complete SMS algorithm."""
    start = perf_counter()
    transcription = SMSIATranscription(
        ocp,
        shooting_grid,
        sms_ia_options=sms_ia_options,
        integrator_options=integrator_options,
    )
    transcription.initialize()
    transcription_seconds = perf_counter() - start
    result = run_sms_algorithm(
        transcription,
        initial_guess,
        algorithm_options=algorithm_options,
        kkt_options=kkt_options,
        solver_options=solver_options,
    )
    timing = {
        "transcription_seconds": transcription_seconds,
        **result.timing,
    }
    timing["total_seconds"] += transcription_seconds
    return replace(result, timing=timing)


@dataclass(frozen=True)
class SMSIASolveResult:
    """Store one transcribed NLP and its numerical solution."""

    transcription: SMSIATranscription
    nlp: ExplicitNlp
    decision_vector: np.ndarray
    objective: float
    constraint_values: np.ndarray
    solver_stats: dict[str, object]


def solve_sms_ia_once(
    ocp: OptimalControlProblem,
    shooting_grid: ArrayLike,
    initial_guess: ArrayLike,
    *,
    sms_ia_options: SMSIAOptions | None = None,
    integrator_options: Mapping[str, object] | None = None,
    solver_options: Mapping[str, object] | None = None,
) -> SMSIASolveResult:
    """Transcribe and solve one fixed-checking-interval SMS-IA NLP."""
    transcription = SMSIATranscription(
        ocp,
        shooting_grid,
        sms_ia_options=sms_ia_options,
        integrator_options=integrator_options,
    )
    transcription.initialize()
    nlp = transcription.build_sms_ia_nlp()
    nlp_result = solve_nlp(
        nlp,
        initial_guess,
        name="sms_ia_once",
        solver_options=solver_options,
    )

    return SMSIASolveResult(
        transcription=transcription,
        nlp=nlp_result.nlp,
        decision_vector=nlp_result.decision_vector,
        objective=nlp_result.objective,
        constraint_values=nlp_result.constraint_values,
        solver_stats=dict(nlp_result.solver_stats),
    )
