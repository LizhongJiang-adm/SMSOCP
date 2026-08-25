"""Run the complete SMS Phase-I, Phase-II, and KKT algorithm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.nlp import ExplicitNlp, NlpSolveResult, solve_nlp
from sms_ocp.sms_ia_checking_intervals import (
    CheckingIntervalUpdate,
)
from sms_ocp.sms_ia_transcription import (
    SMSIAConstraintRowInfo,
    SMSIATranscription,
)
from sms_ocp.sms_kkt import (
    SMSKKTCheck,
    SMSKKTChecker,
    SMSKKTOptions,
)


SMSAlgorithmStatus = Literal[
    "phase_one_failed",
    "phase_two_failed",
    "kkt_not_satisfied",
    "kkt_satisfied",
]

_DEFAULT_IPOPT_CONSTR_VIOL_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class SMSAlgorithmOptions:
    """Configure the SMS Phase-I and Phase-II refinement loops.

    NLP tolerance compensation shifts SMS upper bounds by zero, one, or
    ``num_intervals + 1`` times ``ipopt.constr_viol_tol`` for ``none``,
    ``basic``, and ``cumulative``, respectively.
    """

    rho_lower_bound: float = -1e-3
    max_phase_one_refinements: int = 15
    max_phase_two_refinements: int = 15
    sms_upper_bound_active_tolerance: float = 1e-6
    initial_rho_margin: float = 1e-4
    print_progress: bool = True
    print_ipopt_output: bool = False
    nlp_tolerance_compensation: Literal[
        "none",
        "basic",
        "cumulative",
    ] = "basic"

    def __post_init__(self) -> None:
        rho_lower_bound = float(self.rho_lower_bound)
        if np.isnan(rho_lower_bound) or rho_lower_bound > 0.0:
            raise ValueError(
                "rho_lower_bound must be nonpositive and not NaN."
            )
        object.__setattr__(
            self,
            "rho_lower_bound",
            rho_lower_bound,
        )

        for name in (
            "sms_upper_bound_active_tolerance",
            "initial_rho_margin",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and nonnegative."
                )
            object.__setattr__(self, name, value)

        for name in (
            "max_phase_one_refinements",
            "max_phase_two_refinements",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a nonnegative integer."
                )
            object.__setattr__(self, name, int(value))

        if self.nlp_tolerance_compensation not in (
            "none",
            "basic",
            "cumulative",
        ):
            raise ValueError(
                "nlp_tolerance_compensation must be "
                "'none', 'basic', or 'cumulative'."
            )


@dataclass(frozen=True, slots=True)
class SMSAlgorithmResult:
    """Store the completed portion of one SMS algorithm run."""

    transcription: SMSIATranscription
    phase_one: NlpSolveResult
    initial_rho: float
    phase_one_rho: float
    phase_two: NlpSolveResult | None
    kkt_check: SMSKKTCheck | None
    phase_one_refinement_count: int
    phase_two_refinement_count: int
    status: SMSAlgorithmStatus
    timing: Mapping[str, float]

    @property
    def is_success(self) -> bool:
        """Return whether Phase II satisfies the sampled KKT check."""
        return self.status == "kkt_satisfied"


def run_sms_algorithm(
    transcription: SMSIATranscription,
    initial_guess: ArrayLike,
    *,
    algorithm_options: SMSAlgorithmOptions | None = None,
    kkt_options: SMSKKTOptions | None = None,
    solver_options: Mapping[str, object] | None = None,
) -> SMSAlgorithmResult:
    """Run the complete SMS algorithm on an initialized transcription."""
    if not transcription.is_initialized:
        raise RuntimeError(
            "Initialize the SMS-IA transcription before "
            "running the SMS algorithm."
        )

    total_start = perf_counter()
    options = (
        SMSAlgorithmOptions()
        if algorithm_options is None
        else algorithm_options
    )
    timing: dict[str, float] = {
        "phase_one_assembly_seconds": 0.0,
        "phase_one_update_seconds": 0.0,
        "phase_two_assembly_seconds": 0.0,
        "phase_two_update_seconds": 0.0,
        "kkt_build_seconds": 0.0,
        "kkt_check_seconds": 0.0,
    }

    point = np.asarray(initial_guess, dtype=float).reshape(-1)
    decision_size = transcription.decision_layout.size
    if point.size != decision_size:
        raise ValueError(
            f"initial_guess must contain {decision_size} entries."
        )
    if not np.isfinite(point).all():
        raise ValueError(
            "initial_guess must contain only finite values."
        )
    point = np.minimum(
        np.maximum(
            point,
            transcription.decision_lower_bounds,
        ),
        transcription.decision_upper_bounds,
    )

    resolved_solver_options: dict[str, object] = {
        "ipopt.constr_viol_tol": (
            _DEFAULT_IPOPT_CONSTR_VIOL_TOL
        ),
    }
    resolved_solver_options.update(solver_options or {})
    if not options.print_ipopt_output:
        resolved_solver_options.update(
            {
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "print_time": False,
            }
        )
    constraint_violation_tolerance = float(
        resolved_solver_options["ipopt.constr_viol_tol"]
    )
    if (
        not np.isfinite(constraint_violation_tolerance)
        or constraint_violation_tolerance <= 0.0
    ):
        raise ValueError(
            "ipopt.constr_viol_tol must be finite and positive."
        )
    shift_multiplier = {
        "none": 0,
        "basic": 1,
        "cumulative": transcription.num_intervals + 1,
    }[options.nlp_tolerance_compensation]
    sms_shift = shift_multiplier * constraint_violation_tolerance

    initial_sms_function = ca.Function(
        "sms_algorithm_initial_sms_values",
        [transcription.decision_vector],
        [transcription.sms_upper_bound_constraint_vector],
    )
    initial_sms_values = np.asarray(
        initial_sms_function(point),
        dtype=float,
    ).reshape(-1)
    initial_rho = max(
        options.rho_lower_bound,
        float(np.max(initial_sms_values))
        + constraint_violation_tolerance
        + options.initial_rho_margin,
    )
    phase_one_guess = np.zeros(
        transcription.sms_fr_decision_layout.size
    )
    phase_one_guess[:decision_size] = point
    rho_block = transcription.sms_fr_decision_layout.block("rho")
    phase_one_guess[
        rho_block.start:rho_block.stop
    ] = initial_rho

    phase_one_refinement_count = 0
    if options.print_progress:
        print("\nSMS Algorithm Progress", flush=True)
        print(
            f"{'Phase':<8} | {'SMS iter':>8} | "
            f"{'Checking intervals':>18} | "
            f"{'NLP status':<28} | {'Success':^7} | "
            f"{'Objective':>14} | {'IPOPT iter':>10} | "
            f"{'Build [s]':>9} | {'Solve [s]':>9} | "
            f"{'KKT [s]':>9} | {'KKT stat. residual':>18} | "
            f"{'Total [s]':>9}",
            flush=True,
        )
        print(
            "-+-".join(
                "-" * width
                for width in (8, 8, 18, 28, 7, 14, 10, 9, 9, 9, 18, 9)
            ),
            flush=True,
        )

    for phase_one_iteration in range(
        options.max_phase_one_refinements + 1
    ):
        iteration_prefix = (
            f"phase_one_iteration_{phase_one_iteration}"
        )
        start = perf_counter()
        phase_one_nlp = transcription.build_sms_fr_nlp(
            rho_lower_bound=options.rho_lower_bound,
        )
        phase_one_nlp = _shift_sms_upper_bounds(
            phase_one_nlp,
            transcription.sms_upper_bound_constraint_slice,
            sms_shift,
        )
        assembly_seconds = perf_counter() - start
        timing["phase_one_assembly_seconds"] += assembly_seconds
        timing[f"{iteration_prefix}_assembly_seconds"] = (
            assembly_seconds
        )
        phase_one = solve_nlp(
            phase_one_nlp,
            phase_one_guess,
            name=(
                "sms_algorithm_phase_one_"
                f"{phase_one_iteration}"
            ),
            solver_options=resolved_solver_options,
        )
        timing[f"{iteration_prefix}_solver_build_seconds"] = (
            phase_one.solver_build_seconds
        )
        timing[f"{iteration_prefix}_solve_seconds"] = (
            phase_one.solve_seconds
        )
        if options.print_progress:
            _print_nlp_progress_row(
                "Phase I",
                phase_one_iteration + 1,
                len(transcription.sms_ia_constraint_rows),
                phase_one,
            )
        phase_one_rho = float(
            phase_one.decision_vector[rho_block.start]
        )

        if not phase_one.is_accepted:
            timing["total_seconds"] = (
                perf_counter() - total_start
            )
            return SMSAlgorithmResult(
                transcription=transcription,
                phase_one=phase_one,
                initial_rho=initial_rho,
                phase_one_rho=phase_one_rho,
                phase_two=None,
                kkt_check=None,
                phase_one_refinement_count=(
                    phase_one_refinement_count
                ),
                phase_two_refinement_count=0,
                status="phase_one_failed",
                timing=timing,
            )

        if phase_one_rho <= 0.0:
            break

        if (
            phase_one_iteration
            >= options.max_phase_one_refinements
        ):
            timing["total_seconds"] = (
                perf_counter() - total_start
            )
            return SMSAlgorithmResult(
                transcription=transcription,
                phase_one=phase_one,
                initial_rho=initial_rho,
                phase_one_rho=phase_one_rho,
                phase_two=None,
                kkt_check=None,
                phase_one_refinement_count=(
                    phase_one_refinement_count
                ),
                phase_two_refinement_count=0,
                status="phase_one_failed",
                timing=timing,
            )

        active_rows = tuple(
            row
            for row in transcription.sms_ia_constraint_rows
            if (
                phase_one.constraint_values[
                    row.nlp_row_index
                ]
                >= phase_one.nlp.ubg[row.nlp_row_index]
                - options.sms_upper_bound_active_tolerance
            )
        )
        updates = _build_active_interval_bisection_updates(
            transcription,
            active_rows,
        )
        if not updates:
            timing["total_seconds"] = (
                perf_counter() - total_start
            )
            return SMSAlgorithmResult(
                transcription=transcription,
                phase_one=phase_one,
                initial_rho=initial_rho,
                phase_one_rho=phase_one_rho,
                phase_two=None,
                kkt_check=None,
                phase_one_refinement_count=(
                    phase_one_refinement_count
                ),
                phase_two_refinement_count=0,
                status="phase_one_failed",
                timing=timing,
            )

        start = perf_counter()
        transcription.update_checking_intervals(updates)
        update_seconds = perf_counter() - start
        timing["phase_one_update_seconds"] += update_seconds
        timing[f"{iteration_prefix}_update_seconds"] = (
            update_seconds
        )
        phase_one_guess = phase_one.decision_vector
        phase_one_refinement_count += 1

    start = perf_counter()
    kkt_checker = SMSKKTChecker(
        transcription,
        options=kkt_options,
    )
    timing["kkt_build_seconds"] = perf_counter() - start

    phase_two_guess = phase_one.decision_vector[
        :decision_size
    ]
    phase_two_refinement_count = 0
    kkt_check: SMSKKTCheck | None = None

    for phase_two_iteration in range(
        options.max_phase_two_refinements + 1
    ):
        iteration_prefix = (
            f"phase_two_iteration_{phase_two_iteration}"
        )
        start = perf_counter()
        phase_two_nlp = transcription.build_sms_ia_nlp()
        phase_two_nlp = _shift_sms_upper_bounds(
            phase_two_nlp,
            transcription.sms_upper_bound_constraint_slice,
            sms_shift,
        )
        assembly_seconds = perf_counter() - start
        timing["phase_two_assembly_seconds"] += assembly_seconds
        timing[f"{iteration_prefix}_assembly_seconds"] = (
            assembly_seconds
        )
        phase_two = solve_nlp(
            phase_two_nlp,
            phase_two_guess,
            name=(
                "sms_algorithm_phase_two_"
                f"{phase_two_iteration}"
            ),
            solver_options=resolved_solver_options,
        )
        timing[f"{iteration_prefix}_solver_build_seconds"] = (
            phase_two.solver_build_seconds
        )
        timing[f"{iteration_prefix}_solve_seconds"] = (
            phase_two.solve_seconds
        )
        if not phase_two.is_accepted:
            if options.print_progress:
                _print_nlp_progress_row(
                    "Phase II",
                    phase_two_iteration + 1,
                    len(transcription.sms_ia_constraint_rows),
                    phase_two,
                )
            timing["total_seconds"] = (
                perf_counter() - total_start
            )
            return SMSAlgorithmResult(
                transcription=transcription,
                phase_one=phase_one,
                initial_rho=initial_rho,
                phase_one_rho=phase_one_rho,
                phase_two=phase_two,
                kkt_check=kkt_check,
                phase_one_refinement_count=(
                    phase_one_refinement_count
                ),
                phase_two_refinement_count=(
                    phase_two_refinement_count
                ),
                status="phase_two_failed",
                timing=timing,
            )

        start = perf_counter()
        kkt_check = kkt_checker.check(
            phase_two.decision_vector
        )
        kkt_check_seconds = perf_counter() - start
        timing["kkt_check_seconds"] += kkt_check_seconds
        timing[f"{iteration_prefix}_kkt_check_seconds"] = (
            kkt_check_seconds
        )
        if options.print_progress:
            _print_nlp_progress_row(
                "Phase II",
                phase_two_iteration + 1,
                len(transcription.sms_ia_constraint_rows),
                phase_two,
                kkt_check_seconds=kkt_check_seconds,
                stationarity_residual=(
                    kkt_check.stationarity_residual
                ),
            )
        if kkt_check.is_satisfied:
            timing["total_seconds"] = (
                perf_counter() - total_start
            )
            return SMSAlgorithmResult(
                transcription=transcription,
                phase_one=phase_one,
                initial_rho=initial_rho,
                phase_one_rho=phase_one_rho,
                phase_two=phase_two,
                kkt_check=kkt_check,
                phase_one_refinement_count=(
                    phase_one_refinement_count
                ),
                phase_two_refinement_count=(
                    phase_two_refinement_count
                ),
                status="kkt_satisfied",
                timing=timing,
            )

        if (
            phase_two_iteration
            >= options.max_phase_two_refinements
        ):
            timing["total_seconds"] = (
                perf_counter() - total_start
            )
            return SMSAlgorithmResult(
                transcription=transcription,
                phase_one=phase_one,
                initial_rho=initial_rho,
                phase_one_rho=phase_one_rho,
                phase_two=phase_two,
                kkt_check=kkt_check,
                phase_one_refinement_count=(
                    phase_one_refinement_count
                ),
                phase_two_refinement_count=(
                    phase_two_refinement_count
                ),
                status="kkt_not_satisfied",
                timing=timing,
            )

        active_rows = tuple(
            row
            for row in transcription.sms_ia_constraint_rows
            if (
                phase_two.constraint_values[
                    row.nlp_row_index
                ]
                >= phase_two.nlp.ubg[row.nlp_row_index]
                - options.sms_upper_bound_active_tolerance
            )
        )
        updates = _build_active_interval_bisection_updates(
            transcription,
            active_rows,
        )
        if not updates:
            timing["total_seconds"] = (
                perf_counter() - total_start
            )
            return SMSAlgorithmResult(
                transcription=transcription,
                phase_one=phase_one,
                initial_rho=initial_rho,
                phase_one_rho=phase_one_rho,
                phase_two=phase_two,
                kkt_check=kkt_check,
                phase_one_refinement_count=(
                    phase_one_refinement_count
                ),
                phase_two_refinement_count=(
                    phase_two_refinement_count
                ),
                status="kkt_not_satisfied",
                timing=timing,
            )

        start = perf_counter()
        transcription.update_checking_intervals(updates)
        update_seconds = perf_counter() - start
        timing["phase_two_update_seconds"] += update_seconds
        timing[f"{iteration_prefix}_update_seconds"] = (
            update_seconds
        )
        phase_two_guess = phase_two.decision_vector
        phase_two_refinement_count += 1

    raise RuntimeError("Unreachable SMS algorithm loop exit.")


def _print_nlp_progress_row(
    phase: str,
    outer_iteration: int,
    checking_interval_count: int,
    result: NlpSolveResult,
    *,
    kkt_check_seconds: float | None = None,
    stationarity_residual: float | None = None,
) -> None:
    """Print one completed outer NLP solve in the progress table."""
    status = str(result.solver_stats.get("return_status", "unknown"))
    ipopt_iterations = result.solver_stats.get("iter_count", "-")
    kkt_seconds_text = (
        "-"
        if kkt_check_seconds is None
        else f"{kkt_check_seconds:.3f}"
    )
    stationarity_text = (
        "-"
        if stationarity_residual is None
        else f"{stationarity_residual:.6e}"
    )
    total_seconds = (
        result.solver_build_seconds
        + result.solve_seconds
        + (0.0 if kkt_check_seconds is None else kkt_check_seconds)
    )
    print(
        f"{phase:<8} | {outer_iteration:>8} | "
        f"{checking_interval_count:>18} | "
        f"{status:<28} | "
        f"{'yes' if result.solver_success else 'no':^7} | "
        f"{result.objective:>14.6e} | "
        f"{str(ipopt_iterations):>10} | "
        f"{result.solver_build_seconds:>9.3f} | "
        f"{result.solve_seconds:>9.3f} | "
        f"{kkt_seconds_text:>9} | "
        f"{stationarity_text:>18} | "
        f"{total_seconds:>9.3f}",
        flush=True,
    )


def _build_active_interval_bisection_updates(
    transcription: SMSIATranscription,
    active_rows: Sequence[SMSIAConstraintRowInfo],
) -> tuple[CheckingIntervalUpdate, ...]:
    """Bisect checking intervals associated with active SMS rows."""
    active_indices_by_block: dict[
        tuple[int, int],
        set[int],
    ] = {}
    for row in active_rows:
        key = (
            row.inequality_index,
            row.shooting_interval_index,
        )
        active_indices_by_block.setdefault(
            key,
            set(),
        ).add(
            row.inequality_checking_interval_index
        )

    updates: list[CheckingIntervalUpdate] = []
    for (
        inequality_index,
        shooting_interval_index,
    ), active_interval_indices in (
        active_indices_by_block.items()
    ):
        shooting_left = transcription.shooting_grid[
            shooting_interval_index
        ]
        shooting_right = transcription.shooting_grid[
            shooting_interval_index + 1
        ]
        current_intervals = transcription.checking_intervals[
            inequality_index
        ].intervals
        updated_local_intervals: list[
            tuple[float, float]
        ] = []

        for interval_index, (left, right) in enumerate(
            current_intervals
        ):
            if not shooting_left <= left < shooting_right:
                continue

            if interval_index in active_interval_indices:
                midpoint = 0.5 * (left + right)
                updated_local_intervals.extend(
                    (
                        (left, midpoint),
                        (midpoint, right),
                    )
                )
            else:
                updated_local_intervals.append(
                    (left, right)
                )

        updates.append(
            CheckingIntervalUpdate(
                inequality_name=(
                    transcription.scalar_inequality_names[
                        inequality_index
                    ]
                ),
                shooting_interval_index=(
                    shooting_interval_index
                ),
                intervals=tuple(updated_local_intervals),
            )
        )

    return tuple(updates)


def _shift_sms_upper_bounds(
    nlp: ExplicitNlp,
    sms_constraint_slice: slice,
    shift: float,
) -> ExplicitNlp:
    """Tighten only the SMS upper bounds submitted to the NLP solver."""
    shifted_upper_bounds = nlp.ubg.copy()
    shifted_upper_bounds[sms_constraint_slice] -= shift
    return replace(nlp, ubg=shifted_upper_bounds)
