"""Scalar inequality path constraints used by the SMS-IA method."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping, Sequence

import casadi as ca

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.optimal_control_problem import Constraint
from sms_ocp.utils import CasadiExpr


PathInequalitySide = Literal["upper", "lower"]


@dataclass(frozen=True)
class ScalarPathInequality:
    """Represent one canonical scalar inequality ``expr <= 0``."""

    expr: CasadiExpr
    source_constraint_name: str
    component_index: int
    side: PathInequalitySide
    source_bound: float
    smoothing_parameter: float

    @property
    def name(self) -> str:
        return (
            f"{self.source_constraint_name}"
            f"[{self.component_index}].{self.side}"
        )


def scalarize_sms_ia_path_constraints(
    constraints: Sequence[Constraint],
    *,
    default_smoothing_parameter: float = 1e-3,
    smoothing_overrides: Mapping[str, float] | None = None,
) -> tuple[ScalarPathInequality, ...]:
    """Convert finite SMS-IA path-constraint bounds to scalar inequalities."""
    default_smoothing_parameter = float(default_smoothing_parameter)
    if (
        not isfinite(default_smoothing_parameter)
        or default_smoothing_parameter <= 0.0
    ):
        raise ValueError(
            "default_smoothing_parameter must be finite and positive."
        )

    overrides = dict(smoothing_overrides or {})
    scalar_inequalities: list[ScalarPathInequality] = []

    for constraint in constraints:
        if constraint.enforcement != "sms_ia":
            continue

        for component_index in range(constraint.expr.shape[0]):
            expr = constraint.expr[component_index]
            lb = float(constraint.lb[component_index, 0])
            ub = float(constraint.ub[component_index, 0])

            for side, bound, scalar_expr in (
                ("upper", ub, expr - ub),
                ("lower", lb, lb - expr),
            ):
                if not isfinite(bound):
                    continue

                name = f"{constraint.name}[{component_index}].{side}"
                smoothing_parameter = float(
                    overrides.get(name, default_smoothing_parameter)
                )
                if (
                    not isfinite(smoothing_parameter)
                    or smoothing_parameter <= 0.0
                ):
                    raise ValueError(
                        f"smoothing parameter for {name!r} "
                        "must be finite and positive."
                    )

                scalar_inequalities.append(
                    ScalarPathInequality(
                        expr=scalar_expr,
                        source_constraint_name=constraint.name,
                        component_index=component_index,
                        side=side,
                        source_bound=bound,
                        smoothing_parameter=smoothing_parameter,
                    )
                )

    return tuple(scalar_inequalities)


def build_sms_ia_growth_expr(
    model: BaseDynamicModel,
    scalar_inequalities: Sequence[ScalarPathInequality],
) -> CasadiExpr:
    """Build smoothed physical-time growth rates for constant ``u`` and ``p``."""
    if not scalar_inequalities:
        return ca.SX.zeros(0, 1)

    path_expr = ca.vertcat(
        *(inequality.expr for inequality in scalar_inequalities)
    )
    smoothing_parameters = ca.DM(
        [
            inequality.smoothing_parameter
            for inequality in scalar_inequalities
        ]
    )
    path_derivative = (
        ca.jacobian(path_expr, model.t_sym)
        + ca.jtimes(
            path_expr,
            model.x_sym,
            model.ode_expr,
        )
    )

    return 0.5 * (
        ca.sqrt(
            path_derivative**2
            + smoothing_parameters**2
        )
        + path_derivative
    )
