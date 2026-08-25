"""Assemble and evaluate numerical SMS KKT checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import lsq_linear

from sms_ocp.integrator import build_interval_integrator
from sms_ocp.sms_ia_transcription import SMSIATranscription
from sms_ocp.sms_kkt_path_gradients import (
    SMSKKTPathGradientEvaluator,
)
from sms_ocp.sms_kkt_path_active_point_selection import (
    PathGradientPointStrategy,
)
from sms_ocp.sms_kkt_path_sampling import (
    SMSKKTPathSampler,
    SampledPathConstraintPoint,
)
from sms_ocp.utils import CasadiExpr


@dataclass(frozen=True, slots=True)
class SMSKKTOptions:
    """Configure sampled SMS KKT checks."""

    samples_per_atomic_interval: int = 41
    active_tolerance: float = 1e-6
    active_point_sample_stride: int = 3
    path_gradient_point_strategy: (
        PathGradientPointStrategy
    ) = "sparse_active"
    stationarity_tolerance: float = 1e-6
    feasibility_tolerance: float = 1e-6
    path_constraint_scale_overrides: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        sample_count = self.samples_per_atomic_interval
        if (
            isinstance(sample_count, (bool, np.bool_))
            or not isinstance(sample_count, (int, np.integer))
            or sample_count < 2
        ):
            raise ValueError(
                "samples_per_atomic_interval must be "
                "an integer greater than or equal to 2."
            )
        object.__setattr__(
            self,
            "samples_per_atomic_interval",
            int(sample_count),
        )

        sample_stride = self.active_point_sample_stride
        if (
            isinstance(sample_stride, (bool, np.bool_))
            or not isinstance(sample_stride, (int, np.integer))
            or sample_stride < 1
        ):
            raise ValueError(
                "active_point_sample_stride must be "
                "a positive integer."
            )
        object.__setattr__(
            self,
            "active_point_sample_stride",
            int(sample_stride),
        )

        if self.path_gradient_point_strategy not in (
            "maximum",
            "sparse_active",
        ):
            raise ValueError(
                "path_gradient_point_strategy must be "
                "'maximum' or 'sparse_active'."
            )

        for name, value, strictly_positive in (
            ("active_tolerance", self.active_tolerance, True),
            (
                "stationarity_tolerance",
                self.stationarity_tolerance,
                False,
            ),
            (
                "feasibility_tolerance",
                self.feasibility_tolerance,
                False,
            ),
        ):
            value = float(value)
            if (
                not np.isfinite(value)
                or value < 0.0
                or (strictly_positive and value == 0.0)
            ):
                qualifier = "positive" if strictly_positive else "nonnegative"
                raise ValueError(
                    f"{name} must be finite and {qualifier}."
                )
            object.__setattr__(self, name, value)

        overrides = dict(
            self.path_constraint_scale_overrides or {}
        )
        for name, scale in overrides.items():
            if not isinstance(name, str):
                raise TypeError(
                    "Path-constraint scale override names "
                    "must be strings."
                )
            scale = float(scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"Path-constraint scale for {name!r} "
                    "must be finite and positive."
                )
            overrides[name] = scale
        object.__setattr__(
            self,
            "path_constraint_scale_overrides",
            overrides,
        )


@dataclass(frozen=True, slots=True)
class SMSKKTFiniteEvaluation:
    """Numerical finite KKT quantities evaluated at one decision point."""

    objective_gradient: np.ndarray
    equality_values: np.ndarray
    equality_jacobian: np.ndarray
    finite_inequality_values: np.ndarray
    finite_inequality_jacobian: np.ndarray


@dataclass(frozen=True, slots=True)
class SMSKKTActiveSet:
    """Record the source of every active KKT row."""

    finite_inequality_indices: tuple[int, ...]
    fixed_bound_indices: tuple[int, ...]
    lower_bound_indices: tuple[int, ...]
    upper_bound_indices: tuple[int, ...]
    active_path_points: tuple[SampledPathConstraintPoint, ...]


@dataclass(frozen=True, slots=True)
class SMSKKTMultipliers:
    """KKT multipliers separated by their source constraints."""

    finite_equalities: np.ndarray
    fixed_bounds: np.ndarray
    finite_inequalities: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    path_inequalities: np.ndarray


@dataclass(frozen=True, slots=True)
class SMSKKTCheck:
    """Numerical KKT residuals, active rows, and fitted multipliers."""

    is_satisfied: bool
    stationarity_residual: float
    stationarity_residual_squared: float
    max_equality_residual: float
    max_finite_inequality_violation: float
    max_decision_bound_violation: float
    max_path_value: float
    min_inequality_multiplier: float
    stationarity_vector: np.ndarray
    finite_evaluation: SMSKKTFiniteEvaluation
    path_maxima: tuple[SampledPathConstraintPoint, ...]
    active_set: SMSKKTActiveSet
    multipliers: SMSKKTMultipliers
    timing: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _DecisionBoundRows:
    """Canonical equality and active inequality rows from variable bounds."""

    fixed_indices: tuple[int, ...]
    fixed_values: np.ndarray
    fixed_jacobian: np.ndarray
    lower_indices: tuple[int, ...]
    lower_values: np.ndarray
    lower_jacobian: np.ndarray
    upper_indices: tuple[int, ...]
    upper_values: np.ndarray
    upper_jacobian: np.ndarray
    max_violation: float


@dataclass(frozen=True, slots=True)
class _MultiplierFit:
    """Fit constrained least-squares KKT multipliers."""

    equality_multipliers: np.ndarray
    inequality_multipliers: np.ndarray
    stationarity_vector: np.ndarray
    stationarity_residual: float


@dataclass(frozen=True, slots=True)
class SMSKKTFiniteReference:
    """Long-lived finite expressions used by SMS KKT checks.

    SMS-IA upper-bound constraints are deliberately excluded.
    """

    decision_variables: ca.MX
    objective_expression: ca.MX
    equality_expressions: ca.MX
    finite_inequality_expressions: ca.MX
    decision_lower_bounds: np.ndarray
    decision_upper_bounds: np.ndarray
    evaluator: ca.Function

    def evaluate(
        self,
        decision_point: ArrayLike,
    ) -> SMSKKTFiniteEvaluation:
        """Evaluate the fixed finite KKT expressions and derivatives."""
        point = np.asarray(decision_point, dtype=float).reshape(-1)
        if point.size != self.decision_variables.numel():
            raise ValueError(
                "decision_point must contain "
                f"{self.decision_variables.numel()} entries."
            )
        if not np.isfinite(point).all():
            raise ValueError(
                "decision_point must contain only finite values."
            )

        (
            objective_gradient,
            equality_values,
            equality_jacobian,
            finite_inequality_values,
            finite_inequality_jacobian,
        ) = self.evaluator(point)

        return SMSKKTFiniteEvaluation(
            objective_gradient=np.asarray(
                objective_gradient,
                dtype=float,
            ).reshape(-1),
            equality_values=np.asarray(
                equality_values,
                dtype=float,
            ).reshape(-1),
            equality_jacobian=np.asarray(
                equality_jacobian,
                dtype=float,
            ),
            finite_inequality_values=np.asarray(
                finite_inequality_values,
                dtype=float,
            ).reshape(-1),
            finite_inequality_jacobian=np.asarray(
                finite_inequality_jacobian,
                dtype=float,
            ),
        )


def build_sms_kkt_finite_reference(
    transcription: SMSIATranscription,
) -> SMSKKTFiniteReference:
    """Build the checking-interval-independent finite KKT reference."""
    if not transcription.is_initialized:
        raise RuntimeError(
            "Initialize the SMS-IA transcription before building "
            "its KKT finite reference."
        )

    (
        defect_vector,
        lagrange_cost,
    ) = _build_finite_kkt_defects_and_lagrange_cost(
        transcription
    )

    finite_constraint_vector = ca.vertcat(
        defect_vector,
        transcription.node_constraint_vector,
    )
    finite_constraint_lower_bounds = np.concatenate(
        (
            transcription.defect_constraint_lower_bounds,
            transcription.node_constraint_lower_bounds,
        )
    )
    finite_constraint_upper_bounds = np.concatenate(
        (
            transcription.defect_constraint_upper_bounds,
            transcription.node_constraint_upper_bounds,
        )
    )
    (
        equality_expressions,
        finite_inequality_expressions,
    ) = _canonicalize_bounded_constraint_rows(
        finite_constraint_vector,
        finite_constraint_lower_bounds,
        finite_constraint_upper_bounds,
    )

    decision_variables = transcription.decision_vector
    objective_expression = (
        transcription.static_objective_expr
        + lagrange_cost
    )
    objective_gradient = ca.gradient(
        objective_expression,
        decision_variables,
    )
    equality_jacobian = ca.jacobian(
        equality_expressions,
        decision_variables,
    )
    finite_inequality_jacobian = ca.jacobian(
        finite_inequality_expressions,
        decision_variables,
    )
    evaluator = ca.Function(
        "sms_kkt_finite_reference",
        [decision_variables],
        [
            objective_gradient,
            equality_expressions,
            equality_jacobian,
            finite_inequality_expressions,
            finite_inequality_jacobian,
        ],
        ["decision_variables"],
        [
            "objective_gradient",
            "equality_values",
            "equality_jacobian",
            "finite_inequality_values",
            "finite_inequality_jacobian",
        ],
    )

    return SMSKKTFiniteReference(
        decision_variables=decision_variables,
        objective_expression=objective_expression,
        equality_expressions=equality_expressions,
        finite_inequality_expressions=(
            finite_inequality_expressions
        ),
        decision_lower_bounds=(
            transcription.decision_lower_bounds.copy()
        ),
        decision_upper_bounds=(
            transcription.decision_upper_bounds.copy()
        ),
        evaluator=evaluator,
    )


class SMSKKTChecker:
    """Reuse fixed KKT graphs while checking successive NLP points."""

    def __init__(
        self,
        transcription: SMSIATranscription,
        *,
        options: SMSKKTOptions | None = None,
    ) -> None:
        self.transcription = transcription
        self.options = SMSKKTOptions() if options is None else options
        self.finite_reference = build_sms_kkt_finite_reference(
            transcription
        )
        overrides = dict(
            self.options.path_constraint_scale_overrides or {}
        )
        valid_names = set(
            transcription.scalar_inequality_names
        )
        unknown_names = set(overrides).difference(valid_names)
        if unknown_names:
            names = ", ".join(
                repr(name)
                for name in sorted(unknown_names)
            )
            raise ValueError(
                "Unknown path-constraint scale override names: "
                f"{names}."
            )
        self.path_constraint_scales = np.asarray(
            [
                overrides.get(name, 1.0)
                for name in transcription.scalar_inequality_names
            ],
            dtype=float,
        )
        self.path_sampler = SMSKKTPathSampler(
            transcription,
            samples_per_atomic_interval=(
                self.options.samples_per_atomic_interval
            ),
            active_tolerance=self.options.active_tolerance,
            active_point_sample_stride=(
                self.options.active_point_sample_stride
            ),
            path_gradient_point_strategy=(
                self.options.path_gradient_point_strategy
            ),
            path_constraint_scales=self.path_constraint_scales,
        )
        self.path_gradient_evaluator = (
            SMSKKTPathGradientEvaluator(transcription)
        )

    def check(
        self,
        decision_point: ArrayLike,
    ) -> SMSKKTCheck:
        """Check the original OCP KKT conditions at one NLP point."""
        total_start = perf_counter()
        timing: dict[str, float] = {}
        point = np.asarray(decision_point, dtype=float).reshape(-1)

        stage_start = perf_counter()
        finite_evaluation = self.finite_reference.evaluate(point)
        timing["finite_evaluation_seconds"] = (
            perf_counter() - stage_start
        )

        options = self.options
        active_finite_indices_array = np.flatnonzero(
            finite_evaluation.finite_inequality_values
            >= -options.active_tolerance
        )
        active_finite_indices = tuple(
            int(index)
            for index in active_finite_indices_array
        )
        active_finite_values = (
            finite_evaluation.finite_inequality_values[
                active_finite_indices_array
            ]
        )
        active_finite_jacobian = (
            finite_evaluation.finite_inequality_jacobian[
                active_finite_indices_array,
                :,
            ]
        )

        bound_rows = _build_decision_bound_rows(
            point,
            self.finite_reference.decision_lower_bounds,
            self.finite_reference.decision_upper_bounds,
            active_tolerance=options.active_tolerance,
        )
        equality_values = np.concatenate(
            (
                finite_evaluation.equality_values,
                bound_rows.fixed_values,
            )
        )
        equality_jacobian = np.vstack(
            (
                finite_evaluation.equality_jacobian,
                bound_rows.fixed_jacobian,
            )
        )

        stage_start = perf_counter()
        path_sampling = self.path_sampler.compute(point)
        timing["path_sampling_seconds"] = (
            perf_counter() - stage_start
        )
        path_maxima = path_sampling.path_maxima
        active_path_points = path_sampling.active_path_points

        stage_start = perf_counter()
        path_gradient_evaluation = (
            self.path_gradient_evaluator.evaluate(
                point,
                active_path_points,
            )
        )
        timing["path_gradient_seconds"] = (
            perf_counter() - stage_start
        )
        active_path_scales = np.asarray(
            [
                self.path_constraint_scales[
                    path_point.inequality_index
                ]
                for path_point in active_path_points
            ],
            dtype=float,
        )
        if active_path_scales.size:
            active_path_jacobian = (
                path_gradient_evaluation.jacobian
                / active_path_scales[:, None]
            )
        else:
            active_path_jacobian = (
                path_gradient_evaluation.jacobian
            )

        active_inequality_jacobian = np.vstack(
            (
                active_finite_jacobian,
                bound_rows.lower_jacobian,
                bound_rows.upper_jacobian,
                active_path_jacobian,
            )
        )

        stage_start = perf_counter()
        multiplier_fit = _fit_kkt_multipliers(
            finite_evaluation.objective_gradient,
            equality_jacobian,
            active_inequality_jacobian,
        )
        timing["multiplier_fit_seconds"] = (
            perf_counter() - stage_start
        )

        number_of_finite_equalities = (
            finite_evaluation.equality_values.size
        )
        finite_equality_multipliers = (
            multiplier_fit.equality_multipliers[
                :number_of_finite_equalities
            ].copy()
        )
        fixed_bound_multipliers = (
            multiplier_fit.equality_multipliers[
                number_of_finite_equalities:
            ].copy()
        )

        number_of_active_finite = active_finite_values.size
        number_of_active_lower = bound_rows.lower_values.size
        number_of_active_upper = bound_rows.upper_values.size
        finite_stop = number_of_active_finite
        lower_stop = finite_stop + number_of_active_lower
        upper_stop = lower_stop + number_of_active_upper
        inequality_multipliers = (
            multiplier_fit.inequality_multipliers
        )
        finite_inequality_multipliers = (
            inequality_multipliers[:finite_stop].copy()
        )
        lower_bound_multipliers = (
            inequality_multipliers[
                finite_stop:lower_stop
            ].copy()
        )
        upper_bound_multipliers = (
            inequality_multipliers[
                lower_stop:upper_stop
            ].copy()
        )
        path_multipliers = (
            inequality_multipliers[upper_stop:].copy()
        )

        max_equality_residual = (
            float(np.max(np.abs(equality_values)))
            if equality_values.size
            else 0.0
        )
        max_finite_inequality_violation = (
            max(
                0.0,
                float(
                    np.max(
                        finite_evaluation.finite_inequality_values
                    )
                ),
            )
            if finite_evaluation.finite_inequality_values.size
            else 0.0
        )
        max_path_value = (
            max(maximum.value for maximum in path_maxima)
            if path_maxima
            else -np.inf
        )
        min_inequality_multiplier = (
            float(np.min(inequality_multipliers))
            if inequality_multipliers.size
            else np.inf
        )
        stationarity_residual_squared = float(
            multiplier_fit.stationarity_vector
            @ multiplier_fit.stationarity_vector
        )
        # Phase I and Phase II own primal-feasibility decisions. The sampled
        # values above still define the active set and remain available as
        # diagnostics, while this checker stops the algorithm on stationarity.
        is_satisfied = (
            multiplier_fit.stationarity_residual
            <= options.stationarity_tolerance
        )
        timing["total_seconds"] = perf_counter() - total_start

        return SMSKKTCheck(
            is_satisfied=is_satisfied,
            stationarity_residual=(
                multiplier_fit.stationarity_residual
            ),
            stationarity_residual_squared=(
                stationarity_residual_squared
            ),
            max_equality_residual=max_equality_residual,
            max_finite_inequality_violation=(
                max_finite_inequality_violation
            ),
            max_decision_bound_violation=(
                bound_rows.max_violation
            ),
            max_path_value=float(max_path_value),
            min_inequality_multiplier=(
                min_inequality_multiplier
            ),
            stationarity_vector=(
                multiplier_fit.stationarity_vector
            ),
            finite_evaluation=finite_evaluation,
            path_maxima=path_maxima,
            active_set=SMSKKTActiveSet(
                finite_inequality_indices=(
                    active_finite_indices
                ),
                fixed_bound_indices=bound_rows.fixed_indices,
                lower_bound_indices=bound_rows.lower_indices,
                upper_bound_indices=bound_rows.upper_indices,
                active_path_points=active_path_points,
            ),
            multipliers=SMSKKTMultipliers(
                finite_equalities=(
                    finite_equality_multipliers
                ),
                fixed_bounds=fixed_bound_multipliers,
                finite_inequalities=(
                    finite_inequality_multipliers
                ),
                lower_bounds=lower_bound_multipliers,
                upper_bounds=upper_bound_multipliers,
                path_inequalities=path_multipliers,
            ),
            timing=timing,
        )


def _build_finite_kkt_defects_and_lagrange_cost(
    transcription: SMSIATranscription,
) -> tuple[ca.MX, ca.MX]:
    """Build defects and Lagrange cost without SMS quadratures."""
    ocp = transcription.ocp
    model = ocp.model
    quadrature_expression: CasadiExpr = (
        ca.SX.zeros(0, 1)
        if ocp.lagrange_term_expr is None
        else ocp.lagrange_term_expr
    )
    interval_integrator = build_interval_integrator(
        model,
        quadrature_expression,
        output_points=(1.0,),
        name="sms_kkt_finite_interval",
        options=transcription.integrator_options,
    )

    horizon_duration = transcription.terminal_time - ocp.t0
    defect_expressions: list[ca.MX] = []
    lagrange_cost = ca.MX(0)

    for interval_index in range(transcription.num_intervals):
        normalized_start = float(
            transcription.shooting_grid[interval_index]
        )
        normalized_duration = float(
            transcription.shooting_grid[interval_index + 1]
            - transcription.shooting_grid[interval_index]
        )
        start_time = (
            ocp.t0
            + normalized_start * horizon_duration
        )
        duration = normalized_duration * horizon_duration
        integrator_inputs: dict[
            str,
            CasadiExpr | float,
        ] = {
            "x0": transcription.state_nodes[interval_index],
            "start_time": start_time,
            "duration": duration,
        }
        if model.nu > 0:
            integrator_inputs["u"] = (
                transcription.control_intervals[interval_index]
            )
        if model.np > 0:
            assert transcription.parameter_vector is not None
            integrator_inputs["p"] = (
                transcription.parameter_vector
            )

        result = interval_integrator(**integrator_inputs)
        defect_expressions.append(
            result["state"][:, -1]
            - transcription.state_nodes[interval_index + 1]
        )
        if ocp.lagrange_term_expr is not None:
            lagrange_cost += result["integrals"][0, -1]

    return (
        ca.vertcat(*defect_expressions),
        lagrange_cost,
    )


def _canonicalize_bounded_constraint_rows(
    expressions: ca.MX,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[ca.MX, ca.MX]:
    """Convert bounded rows into equalities and ``<= 0`` inequalities."""
    expression_vector = ca.vec(expressions)
    lower = np.asarray(lower_bounds, dtype=float).reshape(-1)
    upper = np.asarray(upper_bounds, dtype=float).reshape(-1)
    if (
        lower.size != expression_vector.numel()
        or upper.size != expression_vector.numel()
    ):
        raise ValueError(
            "Constraint bounds must match the number "
            "of constraint rows."
        )

    equality_parts: list[ca.MX] = []
    inequality_parts: list[ca.MX] = []
    for row_index, (
        lower_bound,
        upper_bound,
    ) in enumerate(zip(lower, upper, strict=True)):
        expression = expression_vector[row_index]
        if (
            np.isfinite(lower_bound)
            and np.isfinite(upper_bound)
            and lower_bound == upper_bound
        ):
            equality_parts.append(
                expression - lower_bound
            )
            continue
        if np.isfinite(upper_bound):
            inequality_parts.append(
                expression - upper_bound
            )
        if np.isfinite(lower_bound):
            inequality_parts.append(
                lower_bound - expression
            )

    equality_expressions = (
        ca.vertcat(*equality_parts)
        if equality_parts
        else ca.MX.zeros(0, 1)
    )
    inequality_expressions = (
        ca.vertcat(*inequality_parts)
        if inequality_parts
        else ca.MX.zeros(0, 1)
    )
    return (
        equality_expressions,
        inequality_expressions,
    )


def _build_decision_bound_rows(
    decision_point: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    active_tolerance: float,
) -> _DecisionBoundRows:
    """Build fixed and epsilon-active decision-bound rows."""
    point = np.asarray(decision_point, dtype=float).reshape(-1)
    lower = np.asarray(lower_bounds, dtype=float).reshape(-1)
    upper = np.asarray(upper_bounds, dtype=float).reshape(-1)
    if lower.size != point.size or upper.size != point.size:
        raise ValueError(
            "Decision bounds must match decision_point."
        )

    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    fixed_mask = finite_lower & finite_upper & (lower == upper)
    lower_mask = finite_lower & ~fixed_mask
    upper_mask = finite_upper & ~fixed_mask

    fixed_indices_array = np.flatnonzero(fixed_mask)
    lower_canonical_values = lower - point
    upper_canonical_values = point - upper
    active_lower_indices_array = np.flatnonzero(
        lower_mask
        & (lower_canonical_values >= -active_tolerance)
    )
    active_upper_indices_array = np.flatnonzero(
        upper_mask
        & (upper_canonical_values >= -active_tolerance)
    )

    def coordinate_rows(
        indices: np.ndarray,
        sign: float,
    ) -> np.ndarray:
        rows = np.zeros((indices.size, point.size))
        rows[np.arange(indices.size), indices] = sign
        return rows

    fixed_values = (
        point[fixed_indices_array]
        - lower[fixed_indices_array]
    )
    lower_values = lower_canonical_values[
        active_lower_indices_array
    ]
    upper_values = upper_canonical_values[
        active_upper_indices_array
    ]
    all_bound_violations = np.concatenate(
        (
            np.abs(fixed_values),
            lower_canonical_values[lower_mask],
            upper_canonical_values[upper_mask],
        )
    )
    max_violation = (
        max(0.0, float(np.max(all_bound_violations)))
        if all_bound_violations.size
        else 0.0
    )

    return _DecisionBoundRows(
        fixed_indices=tuple(
            int(index)
            for index in fixed_indices_array
        ),
        fixed_values=fixed_values,
        fixed_jacobian=coordinate_rows(
            fixed_indices_array,
            1.0,
        ),
        lower_indices=tuple(
            int(index)
            for index in active_lower_indices_array
        ),
        lower_values=lower_values,
        lower_jacobian=coordinate_rows(
            active_lower_indices_array,
            -1.0,
        ),
        upper_indices=tuple(
            int(index)
            for index in active_upper_indices_array
        ),
        upper_values=upper_values,
        upper_jacobian=coordinate_rows(
            active_upper_indices_array,
            1.0,
        ),
        max_violation=max_violation,
    )


def _fit_kkt_multipliers(
    objective_gradient: np.ndarray,
    equality_jacobian: np.ndarray,
    active_inequality_jacobian: np.ndarray,
) -> _MultiplierFit:
    """Fit KKT multipliers with a rank-deficiency-robust method.

    This enhanced fit first uses a rank-revealing SVD to remove the free
    equality-multiplier subspace.  It then solves the remaining projected
    problem for nonnegative inequality multipliers with column-scaled BVLS,
    and finally recovers one minimum-norm set of equality multipliers.
    Consequently, redundant equality rows or active inequalities already
    represented by the equality gradients do not make the bounded
    least-squares problem singular.
    """
    gradient = np.asarray(
        objective_gradient,
        dtype=float,
    ).reshape(-1)
    equality_rows = np.asarray(
        equality_jacobian,
        dtype=float,
    ).reshape(-1, gradient.size)
    inequality_rows = np.asarray(
        active_inequality_jacobian,
        dtype=float,
    ).reshape(-1, gradient.size)
    number_of_equalities = equality_rows.shape[0]
    number_of_inequalities = inequality_rows.shape[0]

    # The columns of E.T describe every stationarity direction that free
    # equality multipliers can cancel.  SVD supplies an orthonormal basis for
    # this column space even when the equality gradients are rank deficient.
    equality_columns = equality_rows.T
    if number_of_equalities:
        left_vectors, singular_values, _ = np.linalg.svd(
            equality_columns,
            full_matrices=False,
        )
        rank_tolerance = (
            np.finfo(float).eps
            * max(equality_columns.shape)
            * singular_values[0]
            if singular_values.size
            else 0.0
        )
        equality_rank = int(
            np.count_nonzero(singular_values > rank_tolerance)
        )
        equality_basis = left_vectors[:, :equality_rank]
    else:
        equality_basis = np.empty((gradient.size, 0))

    # Projecting onto the orthogonal complement of range(E.T) eliminates the
    # unrestricted equality multipliers.  The remaining least-squares problem
    # therefore contains only multipliers that must satisfy nonnegativity.
    projected_gradient = (
        gradient
        - equality_basis @ (equality_basis.T @ gradient)
    )
    inequality_columns = inequality_rows.T
    projected_inequality_columns = (
        inequality_columns
        - equality_basis
        @ (equality_basis.T @ inequality_columns)
    )

    inequality_multipliers = np.zeros(number_of_inequalities)
    if number_of_inequalities:
        projected_column_norms = np.linalg.norm(
            projected_inequality_columns,
            axis=0,
        )
        original_column_norms = np.linalg.norm(
            inequality_columns,
            axis=0,
        )

        # A projected column at round-off level is already generated by the
        # equality gradients.  Its nonnegative multiplier may safely be set
        # to zero because a free equality multiplier can absorb its effect.
        redundancy_tolerances = (
            np.finfo(float).eps
            * max(gradient.size, number_of_equalities, 1)
            * np.maximum(original_column_norms, 1.0)
        )
        retained_indices = np.flatnonzero(
            projected_column_norms > redundancy_tolerances
        )

        if retained_indices.size:
            retained_columns = projected_inequality_columns[
                :,
                retained_indices,
            ]
            retained_norms = projected_column_norms[
                retained_indices
            ]

            # Positive column scaling preserves multiplier signs while keeping
            # the BVLS subproblem insensitive to different gradient units.
            scaled_columns = retained_columns / retained_norms
            result = lsq_linear(
                scaled_columns,
                -projected_gradient,
                bounds=(0.0, np.inf),
                method="bvls",
                max_iter=1000,
            )

            # KKT acceptance depends on the residual reconstructed below, not
            # on BVLS's termination flag.  A finite feasible iterate therefore
            # remains a valid multiplier candidate and gives a conservative
            # "not satisfied" result if its stationarity residual is too large.
            if not np.isfinite(result.x).all():
                raise RuntimeError(
                    "Could not fit SMS KKT inequality multipliers: "
                    "BVLS returned non-finite values."
                )
            inequality_multipliers[retained_indices] = (
                result.x / retained_norms
            )

    # With the nonnegative multipliers fixed, recover one minimum-norm vector
    # of free equality multipliers.  np.linalg.lstsq deliberately supports a
    # rank-deficient E.T and preserves the original multiplier array length.
    if number_of_equalities:
        equality_multipliers = np.linalg.lstsq(
            equality_columns,
            -gradient
            - inequality_columns @ inequality_multipliers,
            rcond=None,
        )[0]
    else:
        equality_multipliers = np.empty(0)

    # Always evaluate stationarity in the original, unprojected coordinates;
    # this is the residual used by the mathematical KKT acceptance test.
    stationarity_vector = (
        gradient
        + equality_columns @ equality_multipliers
        + inequality_columns @ inequality_multipliers
    )
    return _MultiplierFit(
        equality_multipliers=equality_multipliers,
        inequality_multipliers=inequality_multipliers,
        stationarity_vector=stationarity_vector,
        stationarity_residual=float(
            np.linalg.norm(stationarity_vector)
        ),
    )
