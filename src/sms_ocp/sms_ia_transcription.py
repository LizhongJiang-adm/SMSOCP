from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Mapping, Sequence, TypeAlias

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.integrator import build_interval_integrator
from sms_ocp.decision_layout import (
    DecisionBlock,
    DecisionLayout,
    build_multiple_shooting_decision_layout,
)
from sms_ocp.nlp import ExplicitNlp
from sms_ocp.optimal_control_problem import Constraint, OptimalControlProblem
from sms_ocp.sms_ia_checking_intervals import (
    CheckingIntervalUpdate,
    InequalityCheckingIntervals,
    ShootingIntervalPlan,
    build_shooting_interval_plans,
    resolve_initial_checking_intervals,
    resolve_updated_checking_intervals,
)
from sms_ocp.sms_ia_inequality_path_constraints import (
    ScalarPathInequality,
    build_sms_ia_growth_expr,
    scalarize_sms_ia_path_constraints,
)
from sms_ocp.utils import CasadiExpr, normalize_grid_points


_NlpBound: TypeAlias = float | np.ndarray
_NlpConstraintBlock: TypeAlias = tuple[
    CasadiExpr,
    _NlpBound,
    _NlpBound,
]


@dataclass(frozen=True)
class SMSIAOptions:
    """Configure the initial SMS-IA path-constraint approximation."""

    default_smoothing_parameter: float = 1e-3
    smoothing_overrides: Mapping[str, float] | None = None
    checking_interval_overrides: Mapping[
        str,
        Sequence[tuple[float, float]],
    ] | None = None


@dataclass(frozen=True, slots=True)
class SMSIAConstraintRowInfo:
    """Map one scalar SMS-IA NLP constraint row to its source interval.

    ``inequality_checking_interval_index`` is global within the scalar
    inequality rather than local to one shooting interval.
    """

    nlp_row_index: int
    inequality_index: int
    shooting_interval_index: int
    inequality_checking_interval_index: int


@dataclass(frozen=True, slots=True)
class _ShootingIntervalTranscriptionBlock:
    """Store expressions produced by one shooting-interval integration."""

    plan: ShootingIntervalPlan
    defect_constraint_block: _NlpConstraintBlock
    lagrange_cost_contribution: ca.MX
    sms_upper_bound_constraint_blocks: tuple[
        _NlpConstraintBlock,
        ...,
    ]


class SMSIATranscription:
    """Transcribe an optimal-control problem with multiple shooting."""

    decision_layout: DecisionLayout
    decision_vector: ca.MX
    scalar_inequalities: tuple[ScalarPathInequality, ...]
    scalar_inequality_names: tuple[str, ...]
    _scalar_inequality_index_by_name: dict[str, int]
    node_constraint_blocks: list[_NlpConstraintBlock]
    node_constraint_vector: ca.MX
    node_constraint_lower_bounds: np.ndarray
    node_constraint_upper_bounds: np.ndarray
    decision_lower_bounds: np.ndarray
    decision_upper_bounds: np.ndarray
    static_objective_expr: ca.MX
    checking_intervals: tuple[InequalityCheckingIntervals, ...]
    shooting_interval_blocks: tuple[
        _ShootingIntervalTranscriptionBlock,
        ...,
    ]
    defect_constraint_blocks: list[_NlpConstraintBlock]
    defect_constraint_vector: ca.MX
    defect_constraint_lower_bounds: np.ndarray
    defect_constraint_upper_bounds: np.ndarray
    sms_upper_bound_constraint_blocks: list[_NlpConstraintBlock]
    sms_upper_bound_constraint_vector: ca.MX
    sms_upper_bound_constraint_lower_bounds: np.ndarray
    sms_upper_bound_constraint_upper_bounds: np.ndarray
    lagrange_cost: ca.MX
    constraint_vector: ca.MX
    constraint_lower_bounds: np.ndarray
    constraint_upper_bounds: np.ndarray
    sms_upper_bound_constraint_slice: slice
    sms_ia_constraint_rows: tuple[SMSIAConstraintRowInfo, ...]
    rho_symbol: ca.MX
    sms_fr_decision_layout: DecisionLayout
    sms_fr_decision_vector: ca.MX

    def __init__(
        self,
        ocp: OptimalControlProblem,
        shooting_grid: ArrayLike,
        *,
        sms_ia_options: SMSIAOptions | None = None,
        integrator_options: Mapping[str, object] | None = None,
    ) -> None:
        self.ocp = ocp
        self.shooting_grid = normalize_grid_points(shooting_grid)
        if not (
            np.isclose(
                self.shooting_grid[0],
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
            and np.isclose(
                self.shooting_grid[-1],
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "shooting_grid must span normalized time from 0 to 1."
            )
        self.num_intervals = int(self.shooting_grid.size - 1)
        self.sms_ia_options = (
            SMSIAOptions()
            if sms_ia_options is None
            else sms_ia_options
        )
        self.integrator_options = dict(integrator_options or {})

        self._create_decision_variables()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Return whether the transcription has been initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Build the long-lived and initial interval expressions once."""
        if self._initialized:
            return

        options = self.sms_ia_options
        scalar_inequalities = scalarize_sms_ia_path_constraints(
            self.ocp.path_constraints,
            default_smoothing_parameter=(
                options.default_smoothing_parameter
            ),
            smoothing_overrides=options.smoothing_overrides,
        )

        valid_names = {
            inequality.name
            for inequality in scalar_inequalities
        }
        unknown_smoothing_names = set(
            options.smoothing_overrides or {}
        ).difference(valid_names)
        if unknown_smoothing_names:
            names = ", ".join(
                repr(name)
                for name in sorted(unknown_smoothing_names)
            )
            raise ValueError(
                f"Unknown smoothing override names: {names}."
            )

        self._prepare_interval_integration(scalar_inequalities)
        self.node_constraint_blocks = (
            self._build_node_constraint_blocks()
        )
        (
            self.node_constraint_vector,
            self.node_constraint_lower_bounds,
            self.node_constraint_upper_bounds,
        ) = self._pack_constraint_blocks(
            self.node_constraint_blocks
        )
        self._build_decision_bounds()
        self._build_static_objective_expr()
        self.checking_intervals = resolve_initial_checking_intervals(
            self.shooting_grid,
            self.scalar_inequalities,
            options.checking_interval_overrides,
        )
        interval_plans = build_shooting_interval_plans(
            self.shooting_grid,
            self.checking_intervals,
        )
        shooting_interval_blocks = tuple(
            self._build_shooting_interval_transcription_block(
                shooting_interval_index,
                plan,
            )
            for shooting_interval_index, plan in enumerate(
                interval_plans
            )
        )
        self.shooting_interval_blocks = shooting_interval_blocks
        self._assemble_constraint_vectors()
        self._initialized = True

    def update_checking_intervals(
        self,
        updates: Sequence[CheckingIntervalUpdate],
    ) -> None:
        """Replace local intervals and rebuild affected shooting blocks."""
        if len(updates) == 0:
            return
        if not self._initialized:
            raise RuntimeError(
                "SMSIATranscription must be initialized before "
                "checking intervals can be updated."
            )

        (
            updated_checking_intervals,
            changed_shooting_interval_indices,
        ) = resolve_updated_checking_intervals(
            self.shooting_grid,
            self.checking_intervals,
            updates,
            inequality_index_by_name=(
                self._scalar_inequality_index_by_name
            ),
        )
        if not changed_shooting_interval_indices:
            return

        resolved_interval_plans = build_shooting_interval_plans(
            self.shooting_grid,
            updated_checking_intervals,
        )
        updated_shooting_interval_blocks = list(
            self.shooting_interval_blocks
        )

        for shooting_interval_index, resolved_plan in enumerate(
            resolved_interval_plans
        ):
            old_block = self.shooting_interval_blocks[
                shooting_interval_index
            ]

            if (
                shooting_interval_index
                in changed_shooting_interval_indices
            ):
                updated_shooting_interval_blocks[
                    shooting_interval_index
                ] = self._build_shooting_interval_transcription_block(
                    shooting_interval_index,
                    resolved_plan,
                )
                continue

            if resolved_plan == old_block.plan:
                continue

            assert (
                resolved_plan.output_points
                == old_block.plan.output_points
            ), (
                f"Shooting interval {shooting_interval_index} was not "
                "marked as changed, but its integrator output points "
                f"changed: old={old_block.plan.output_points}, "
                f"updated={resolved_plan.output_points}."
            )

            old_check_geometry = tuple(
                (
                    check.inequality_index,
                    check.left_point_index,
                    check.right_point_index,
                )
                for check in old_block.plan.checking_intervals
            )
            updated_check_geometry = tuple(
                (
                    check.inequality_index,
                    check.left_point_index,
                    check.right_point_index,
                )
                for check in resolved_plan.checking_intervals
            )
            assert updated_check_geometry == old_check_geometry, (
                f"Shooting interval {shooting_interval_index} was not "
                "marked as changed, but its SMS-IA checking-interval "
                f"routing changed: old={old_check_geometry}, "
                f"updated={updated_check_geometry}. Only "
                "inequality_checking_interval_index changes are "
                "allowed for an unchanged shooting interval."
            )

            updated_shooting_interval_blocks[
                shooting_interval_index
            ] = replace(
                old_block,
                plan=resolved_plan,
            )

        self.checking_intervals = updated_checking_intervals
        self.shooting_interval_blocks = tuple(
            updated_shooting_interval_blocks
        )
        self._assemble_constraint_vectors()

    def build_sms_ia_nlp(self) -> ExplicitNlp:
        """Assemble the Phase-II SMS-IA nonlinear program."""
        if not self._initialized:
            warnings.warn(
                "SMSIATranscription is not initialized. Call initialize() "
                "after fully defining the OCP and before building an NLP. "
                "Initializing automatically now.",
                UserWarning,
                stacklevel=2,
            )
            self.initialize()

        objective_expr = (
            self.static_objective_expr
            + self.lagrange_cost
        )
        return ExplicitNlp(
            x=self.decision_vector,
            f=objective_expr,
            g=self.constraint_vector,
            lbx=self.decision_lower_bounds,
            ubx=self.decision_upper_bounds,
            lbg=self.constraint_lower_bounds,
            ubg=self.constraint_upper_bounds,
        )

    def build_sms_fr_nlp(
        self,
        *,
        rho_lower_bound: float = -np.inf,
    ) -> ExplicitNlp:
        """Assemble the Phase-I SMS feasibility-restoration NLP."""
        rho_lower_bound = float(rho_lower_bound)
        if np.isnan(rho_lower_bound) or rho_lower_bound > 0.0:
            raise ValueError(
                "rho_lower_bound must be nonpositive and not NaN."
            )

        if not self._initialized:
            warnings.warn(
                "SMSIATranscription is not initialized. Call initialize() "
                "after fully defining the OCP and before building an NLP. "
                "Initializing automatically now.",
                UserWarning,
                stacklevel=2,
            )
            self.initialize()

        if self.sms_upper_bound_constraint_vector.numel() == 0:
            raise ValueError(
                "SMS-FR requires at least one SMS-IA "
                "upper-bound constraint."
            )

        constraint_vector = ca.vertcat(
            self.defect_constraint_vector,
            self.node_constraint_vector,
            self.sms_upper_bound_constraint_vector - self.rho_symbol,
        )
        return ExplicitNlp(
            x=self.sms_fr_decision_vector,
            f=self.rho_symbol,
            g=constraint_vector,
            lbx=np.concatenate(
                (self.decision_lower_bounds, [rho_lower_bound])
            ),
            ubx=np.concatenate(
                (self.decision_upper_bounds, [np.inf])
            ),
            lbg=self.constraint_lower_bounds,
            ubg=self.constraint_upper_bounds,
        )

    def _create_decision_variables(self) -> None:
        self.decision_layout: DecisionLayout = (
            build_multiple_shooting_decision_layout(
                self.ocp,
                self.num_intervals,
            )
        )
        self.decision_symbols = {
            block.name: ca.MX.sym(block.name, *block.shape)
            for block in self.decision_layout.blocks
        }

        self.state_matrix = self.decision_symbols["x"]
        self.state_nodes = tuple(ca.horzsplit(self.state_matrix))

        if self.decision_layout.has_block("u"):
            self.control_matrix = self.decision_symbols["u"]
            self.control_intervals = tuple(
                ca.horzsplit(self.control_matrix)
            )
        else:
            self.control_matrix = None
            self.control_intervals = ()

        if self.decision_layout.has_block("p"):
            self.parameter_vector = self.decision_symbols["p"]
        else:
            self.parameter_vector = None

        if self.decision_layout.has_block("T"):
            self.terminal_time = self.decision_symbols["T"]
        else:
            self.terminal_time = float(self.ocp.tf_bounds[0])

        self.decision_vector = ca.vertcat(
            *(
                ca.reshape(
                    self.decision_symbols[block.name],
                    -1,
                    1,
                )
                for block in self.decision_layout.blocks
            )
        )
        self.rho_symbol = ca.MX.sym("rho")
        rho_start = self.decision_layout.size
        self.sms_fr_decision_layout = DecisionLayout(
            blocks=(
                *self.decision_layout.blocks,
                DecisionBlock(
                    name="rho",
                    shape=(1, 1),
                    start=rho_start,
                    stop=rho_start + 1,
                ),
            )
        )
        self.sms_fr_decision_vector = ca.vertcat(
            self.decision_vector,
            self.rho_symbol,
        )

    @property
    def rho_index(self) -> int:
        """Return rho's scalar index in the SMS-FR decision vector."""
        return self.sms_fr_decision_layout.block("rho").start

    @property
    def interval_plans(
        self,
    ) -> tuple[ShootingIntervalPlan, ...]:
        """Return the plans owned by the current shooting blocks."""
        return tuple(
            block.plan
            for block in self.shooting_interval_blocks
        )

    def _prepare_interval_integration(
        self,
        scalar_inequalities: Sequence[ScalarPathInequality],
    ) -> None:
        """Prepare fixed quadratures, functions, and the integrator cache."""
        self.scalar_inequalities = tuple(scalar_inequalities)
        self.scalar_inequality_names = tuple(
            inequality.name
            for inequality in self.scalar_inequalities
        )
        self._scalar_inequality_index_by_name = {
            name: index
            for index, name in enumerate(
                self.scalar_inequality_names
            )
        }
        assert (
            len(self._scalar_inequality_index_by_name)
            == len(self.scalar_inequality_names)
        )
        self._sms_functions = tuple(
            self.ocp.model.create_function(
                f"sms_inequality_{index}",
                inequality.expr,
                output_name="value",
            )
            for index, inequality in enumerate(self.scalar_inequalities)
        )

        quadrature_parts: list[CasadiExpr] = []
        if self.ocp.lagrange_term_expr is not None:
            quadrature_parts.append(self.ocp.lagrange_term_expr)
            self._sms_quadrature_offset = 1
        else:
            self._sms_quadrature_offset = 0
        quadrature_parts.append(
            build_sms_ia_growth_expr(
                self.ocp.model,
                self.scalar_inequalities,
            )
        )

        self._quadrature_expr = ca.vertcat(*quadrature_parts)
        self._integrator_cache: dict[tuple[float, ...], ca.Function] = {}

    def _get_interval_integrator(
        self,
        output_points: tuple[float, ...],
    ) -> ca.Function:
        """Return the cached integrator for one output-point pattern."""
        if output_points not in self._integrator_cache:
            integrator_index = len(self._integrator_cache)
            self._integrator_cache[output_points] = (
                build_interval_integrator(
                    self.ocp.model,
                    self._quadrature_expr,
                    output_points,
                    name=f"sms_interval_{integrator_index}",
                    options=self.integrator_options,
                )
            )
        return self._integrator_cache[output_points]

    def _build_shooting_interval_transcription_block(
        self,
        shooting_interval_index: int,
        plan: ShootingIntervalPlan,
    ) -> _ShootingIntervalTranscriptionBlock:
        """Build expressions produced by one shooting-interval integration."""
        model = self.ocp.model
        horizon_duration = self.terminal_time - self.ocp.t0
        x_k = self.state_nodes[shooting_interval_index]
        x_next = self.state_nodes[shooting_interval_index + 1]
        u_k = (
            self.control_intervals[shooting_interval_index]
            if model.nu > 0
            else None
        )

        normalized_start = float(
            self.shooting_grid[shooting_interval_index]
        )
        normalized_duration = float(
            self.shooting_grid[shooting_interval_index + 1]
            - self.shooting_grid[shooting_interval_index]
        )
        start_time = (
            self.ocp.t0
            + normalized_start * horizon_duration
        )
        duration = normalized_duration * horizon_duration

        integrator_inputs: dict[str, CasadiExpr | float] = {
            "x0": x_k,
            "start_time": start_time,
            "duration": duration,
        }
        if model.nu > 0:
            integrator_inputs["u"] = u_k
        if model.np > 0:
            assert self.parameter_vector is not None
            integrator_inputs["p"] = self.parameter_vector

        result = self._get_interval_integrator(plan.output_points)(
            **integrator_inputs
        )
        state_outputs = result["state"]
        integral_outputs = result["integrals"]
        defect_constraint_block: _NlpConstraintBlock = (
            state_outputs[:, -1] - x_next,
            0.0,
            0.0,
        )
        lagrange_cost_contribution = (
            integral_outputs[0, -1]
            if self.ocp.lagrange_term_expr is not None
            else ca.MX(0)
        )

        sms_integrals = integral_outputs[
            self._sms_quadrature_offset:,
            :,
        ]
        states_at_points = ca.horzcat(x_k, state_outputs)
        sms_integrals_at_points = ca.horzcat(
            ca.MX.zeros(len(self.scalar_inequalities), 1),
            sms_integrals,
        )
        local_points = (0.0, *plan.output_points)
        sms_upper_bound_constraint_blocks: list[
            _NlpConstraintBlock
        ] = []

        for check in plan.checking_intervals:
            inequality_index = check.inequality_index
            left_point_index = check.left_point_index
            right_point_index = check.right_point_index
            left_time = (
                start_time
                + local_points[left_point_index] * duration
            )
            model_inputs = model.function_inputs(
                t=left_time,
                x=states_at_points[:, left_point_index],
                u=u_k,
                p=self.parameter_vector,
            )
            left_value = self._sms_functions[inequality_index](
                *model_inputs.values()
            )
            upper_bound_expr = (
                left_value
                + sms_integrals_at_points[
                    inequality_index,
                    right_point_index,
                ]
                - sms_integrals_at_points[
                    inequality_index,
                    left_point_index,
                ]
            )
            sms_upper_bound_constraint_blocks.append(
                (
                    upper_bound_expr,
                    -np.inf,
                    0.0,
                )
            )

        return _ShootingIntervalTranscriptionBlock(
            plan=plan,
            defect_constraint_block=defect_constraint_block,
            lagrange_cost_contribution=lagrange_cost_contribution,
            sms_upper_bound_constraint_blocks=tuple(
                sms_upper_bound_constraint_blocks
            ),
        )

    def _build_node_constraint_blocks(
        self,
    ) -> list[_NlpConstraintBlock]:
        """Build grid-only, initial, and terminal constraint blocks."""
        grid_only_constraints = tuple(
            constraint
            for constraint in self.ocp.path_constraints
            if constraint.enforcement == "grid_only"
        )
        return (
            self._evaluate_constraints_at_nodes(
                grid_only_constraints,
                range(self.num_intervals + 1),
                function_prefix="grid_only",
            )
            + self._evaluate_constraints_at_nodes(
                self.ocp.initial_constraints,
                (0,),
                function_prefix="initial",
            )
            + self._evaluate_constraints_at_nodes(
                self.ocp.terminal_constraints,
                (self.num_intervals,),
                function_prefix="terminal",
            )
        )

    def _assemble_constraint_vectors(self) -> None:
        """Pack and concatenate the three constraint categories."""
        self.defect_constraint_blocks = [
            block.defect_constraint_block
            for block in self.shooting_interval_blocks
        ]
        self.sms_upper_bound_constraint_blocks = [
            constraint_block
            for shooting_block in self.shooting_interval_blocks
            for constraint_block in (
                shooting_block.sms_upper_bound_constraint_blocks
            )
        ]
        assert all(
            len(block.plan.checking_intervals)
            == len(block.sms_upper_bound_constraint_blocks)
            for block in self.shooting_interval_blocks
        ), (
            "Each shooting block must contain one SMS upper-bound "
            "constraint block per checking-interval plan."
        )
        sms_row_sources = [
            (
                check.inequality_index,
                shooting_interval_index,
                check.inequality_checking_interval_index,
            )
            for shooting_interval_index, block in enumerate(
                self.shooting_interval_blocks
            )
            for check in block.plan.checking_intervals
        ]
        self.lagrange_cost = ca.MX(0)
        for block in self.shooting_interval_blocks:
            self.lagrange_cost += block.lagrange_cost_contribution

        (
            self.defect_constraint_vector,
            self.defect_constraint_lower_bounds,
            self.defect_constraint_upper_bounds,
        ) = self._pack_constraint_blocks(
            self.defect_constraint_blocks
        )
        (
            self.sms_upper_bound_constraint_vector,
            self.sms_upper_bound_constraint_lower_bounds,
            self.sms_upper_bound_constraint_upper_bounds,
        ) = self._pack_constraint_blocks(
            self.sms_upper_bound_constraint_blocks
        )

        sms_start = int(
            self.defect_constraint_vector.numel()
            + self.node_constraint_vector.numel()
        )
        sms_size = int(
            self.sms_upper_bound_constraint_vector.numel()
        )
        assert sms_size == len(sms_row_sources), (
            "The number of assembled SMS upper-bound rows must match "
            "the number of checking-interval plan sources."
        )
        self.sms_upper_bound_constraint_slice = slice(
            sms_start,
            sms_start + sms_size,
        )
        self.sms_ia_constraint_rows = tuple(
            SMSIAConstraintRowInfo(
                nlp_row_index=sms_start + local_row_index,
                inequality_index=inequality_index,
                shooting_interval_index=shooting_interval_index,
                inequality_checking_interval_index=(
                    inequality_checking_interval_index
                ),
            )
            for local_row_index, (
                inequality_index,
                shooting_interval_index,
                inequality_checking_interval_index,
            ) in enumerate(sms_row_sources)
        )
        self.constraint_vector = ca.vertcat(
            self.defect_constraint_vector,
            self.node_constraint_vector,
            self.sms_upper_bound_constraint_vector,
        )
        self.constraint_lower_bounds = np.concatenate(
            (
                self.defect_constraint_lower_bounds,
                self.node_constraint_lower_bounds,
                self.sms_upper_bound_constraint_lower_bounds,
            )
        )
        self.constraint_upper_bounds = np.concatenate(
            (
                self.defect_constraint_upper_bounds,
                self.node_constraint_upper_bounds,
                self.sms_upper_bound_constraint_upper_bounds,
            )
        )

    def _build_static_objective_expr(self) -> None:
        """Build the checking-interval-independent objective terms."""
        model = self.ocp.model
        objective = ca.MX(0)

        if self.ocp.mayer_term_expr is not None:
            mayer_function = model.create_function(
                "mayer_term",
                self.ocp.mayer_term_expr,
                output_name="value",
            )
            terminal_control = (
                self.control_intervals[-1]
                if model.nu > 0
                else None
            )
            terminal_inputs = model.function_inputs(
                t=self.terminal_time,
                x=self.state_nodes[-1],
                u=terminal_control,
                p=self.parameter_vector,
            )
            objective += mayer_function(
                *terminal_inputs.values()
            )

        if self.ocp.min_time_weight > 0.0:
            objective += (
                self.ocp.min_time_weight
                * (self.terminal_time - self.ocp.t0)
            )

        self.static_objective_expr = objective

    def _build_decision_bounds(self) -> None:
        """Pack variable bounds in the decision-vector layout order."""
        block_bounds: dict[
            str,
            tuple[np.ndarray, np.ndarray],
        ] = {
            "x": (
                np.repeat(
                    self.ocp.x_lb,
                    self.num_intervals + 1,
                    axis=1,
                ),
                np.repeat(
                    self.ocp.x_ub,
                    self.num_intervals + 1,
                    axis=1,
                ),
            ),
        }

        if self.decision_layout.has_block("u"):
            block_bounds["u"] = (
                np.repeat(
                    self.ocp.u_lb,
                    self.num_intervals,
                    axis=1,
                ),
                np.repeat(
                    self.ocp.u_ub,
                    self.num_intervals,
                    axis=1,
                ),
            )

        if self.decision_layout.has_block("p"):
            block_bounds["p"] = (
                self.ocp.p_lb,
                self.ocp.p_ub,
            )

        if self.decision_layout.has_block("T"):
            block_bounds["T"] = (
                self.ocp.tf_bounds[:1].reshape(1, 1),
                self.ocp.tf_bounds[1:].reshape(1, 1),
            )

        self.decision_lower_bounds = np.concatenate(
            [
                block_bounds[block.name][0].reshape(
                    -1,
                    order="F",
                )
                for block in self.decision_layout.blocks
            ]
        )
        self.decision_upper_bounds = np.concatenate(
            [
                block_bounds[block.name][1].reshape(
                    -1,
                    order="F",
                )
                for block in self.decision_layout.blocks
            ]
        )

    def _evaluate_constraints_at_nodes(
        self,
        constraints: Sequence[Constraint],
        node_indices: Sequence[int],
        *,
        function_prefix: str,
    ) -> list[_NlpConstraintBlock]:
        """Evaluate model expressions at selected multiple-shooting nodes."""
        model = self.ocp.model
        horizon_duration = self.terminal_time - self.ocp.t0
        blocks: list[_NlpConstraintBlock] = []

        for constraint_index, constraint in enumerate(constraints):
            constraint_function = model.create_function(
                f"{function_prefix}_{constraint_index}",
                constraint.expr,
                output_name="value",
            )
            for node_index in node_indices:
                node_time = (
                    self.ocp.t0
                    + float(self.shooting_grid[node_index])
                    * horizon_duration
                )
                u_at_node = (
                    self.control_intervals[
                        min(node_index, self.num_intervals - 1)
                    ]
                    if model.nu > 0
                    else None
                )
                model_inputs = model.function_inputs(
                    t=node_time,
                    x=self.state_nodes[node_index],
                    u=u_at_node,
                    p=self.parameter_vector,
                )
                blocks.append(
                    (
                        constraint_function(*model_inputs.values()),
                        constraint.lb,
                        constraint.ub,
                    )
                )

        return blocks

    @staticmethod
    def _pack_constraint_blocks(
        blocks: Sequence[_NlpConstraintBlock],
    ) -> tuple[CasadiExpr, np.ndarray, np.ndarray]:
        """Pack expression blocks and broadcast their numerical bounds."""
        if not blocks:
            return (
                ca.MX.zeros(0, 1),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
            )

        expressions: list[CasadiExpr] = []
        lower_bounds: list[np.ndarray] = []
        upper_bounds: list[np.ndarray] = []
        for expression, lb, ub in blocks:
            expression_column = ca.vec(expression)
            size = int(expression_column.numel())
            expressions.append(expression_column)
            lower_bounds.append(
                np.broadcast_to(
                    np.asarray(lb, dtype=float),
                    (size, 1),
                ).reshape(-1).copy()
            )
            upper_bounds.append(
                np.broadcast_to(
                    np.asarray(ub, dtype=float),
                    (size, 1),
                ).reshape(-1).copy()
            )

        return (
            ca.vertcat(*expressions),
            np.concatenate(lower_bounds),
            np.concatenate(upper_bounds),
        )
