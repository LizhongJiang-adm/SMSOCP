import casadi as ca
import numpy as np
import pytest

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.optimal_control_problem import OptimalControlProblem
from sms_ocp.sms_solver import solve_sms_ia_once


class IntegratorModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = self.u_sym


def test_solve_sms_ia_once_solves_integrator_problem() -> None:
    model = IntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.set_objective(lagrange=model.u_sym**2)
    ocp.set_variable_bounds("u", lb=-3.0, ub=3.0)
    ocp.add_initial_constraint(model.x_sym, 0.0)
    ocp.add_terminal_constraint(model.x_sym, 1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.5,
        name="state_limit",
    )

    result = solve_sms_ia_once(
        ocp,
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0, 1.0, 1.0],
        solver_options={"ipopt.tol": 1e-9},
    )

    assert result.solver_stats["success"]
    assert result.objective == pytest.approx(1.0, abs=1e-7)
    states = result.transcription.decision_layout.extract(
        result.decision_vector,
        "x",
    )
    assert states[0, -1] == pytest.approx(1.0, abs=1e-8)
    assert np.max(
        np.maximum(
            result.constraint_values - result.nlp.ubg,
            result.nlp.lbg - result.constraint_values,
        )
    ) <= 1e-7
