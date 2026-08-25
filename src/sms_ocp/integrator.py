"""CasADi integration over one normalized shooting interval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import casadi as ca
import numpy as np

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.utils import CasadiExpr, as_casadi_column_vector


def build_interval_integrator(
    model: BaseDynamicModel,
    quadrature_expr: CasadiExpr,
    output_points: Sequence[float],
    *,
    name: str = "interval_integrator",
    options: Mapping[str, object] | None = None,
) -> ca.Function:
    """Build an interval propagator, using CVODES by default.

    ``options["integrator_plugin"]`` is a package-level selector and is
    removed before passing the remaining options to CasADi.  The currently
    supported plugins are ``"cvodes"`` and the fixed-step RK4 plugin
    ``"rk"``.
    """
    output_grid = np.asarray(output_points, dtype=float).reshape(-1)
    if (
        output_grid.size == 0
        or not np.isfinite(output_grid).all()
        or output_grid[0] <= 0.0
        or output_grid[-1] > 1.0
        or np.any(np.diff(output_grid) <= 0.0)
    ):
        raise ValueError(
            "output_points must be finite, strictly increasing, "
            "and within (0, 1]."
        )

    quadrature_expr, n_quadratures = as_casadi_column_vector(
        quadrature_expr,
        "quadrature_expr",
    )
    quadrature_expr = ca.SX(quadrature_expr)

    local_time = ca.SX.sym("local_time")
    start_time = ca.SX.sym("start_time")
    duration = ca.SX.sym("duration")
    physical_time = start_time + duration * local_time

    parameters = ca.vertcat(
        model.u_sym,
        model.p_sym,
        start_time,
        duration,
    )
    dae = {
        "t": local_time,
        "x": model.x_sym,
        "p": parameters,
        "ode": duration
        * ca.substitute(
            model.ode_expr,
            model.t_sym,
            physical_time,
        ),
    }
    if n_quadratures > 0:
        dae["quad"] = duration * ca.substitute(
            quadrature_expr,
            model.t_sym,
            physical_time,
        )

    resolved_options = dict(options or {})
    integrator_plugin = str(
        resolved_options.pop("integrator_plugin", "cvodes")
    )
    if integrator_plugin not in {"cvodes", "rk"}:
        raise ValueError(
            "integrator_plugin must be 'cvodes' or 'rk'."
        )
    if integrator_plugin == "cvodes":
        resolved_options = {
            "quad_err_con": True,
            **resolved_options,
        }

    integrator = ca.integrator(
        f"{name}_raw",
        integrator_plugin,
        dae,
        0.0,
        output_grid.tolist(),
        resolved_options,
    )

    x0 = ca.MX.sym("x0", model.nx)
    function_inputs = [x0]
    input_names = ["x0"]
    parameter_values = []

    if model.nu > 0:
        u = ca.MX.sym("u", model.nu)
        function_inputs.append(u)
        input_names.append("u")
        parameter_values.append(u)

    if model.np > 0:
        p = ca.MX.sym("p", model.np)
        function_inputs.append(p)
        input_names.append("p")
        parameter_values.append(p)

    start_time_value = ca.MX.sym("start_time")
    duration_value = ca.MX.sym("duration")
    function_inputs.extend(
        [
            start_time_value,
            duration_value,
        ]
    )
    input_names.extend(
        [
            "start_time",
            "duration",
        ]
    )
    parameter_values.extend(
        [
            start_time_value,
            duration_value,
        ]
    )

    result = integrator(
        x0=x0,
        p=ca.vertcat(*parameter_values),
    )
    return ca.Function(
        name,
        function_inputs,
        [
            result["xf"],
            result["qf"],
        ],
        input_names,
        [
            "state",
            "integrals",
        ],
    )
