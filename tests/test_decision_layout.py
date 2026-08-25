import casadi as ca
import numpy as np
import pytest

from sms_ocp import BaseDynamicModel, OptimalControlProblem, pack_initial_guess
from sms_ocp.decision_layout import build_multiple_shooting_decision_layout


class ParameterizedIntegratorModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 2)
        self.u_sym = ca.SX.sym("u")
        self.p_sym = ca.SX.sym("p", 2)
        self.ode_expr = ca.vertcat(
            self.x_sym[1],
            self.u_sym[0] + self.p_sym[0],
        )


def test_pack_initial_guess_matches_decision_layout() -> None:
    ocp = OptimalControlProblem(ParameterizedIntegratorModel())
    ocp.set_time_horizon(tf=(1.0, 2.0))
    shooting_grid = np.array([0.0, 0.5, 1.0])
    states = np.array([[0.0, 0.25, 0.5], [1.0, 1.0, 1.0]])
    controls = np.array([[0.2, 0.3]])
    parameters = np.array([4.0, 5.0])

    guess = pack_initial_guess(
        ocp,
        shooting_grid,
        states=states,
        controls=controls,
        parameters=parameters,
        terminal_time=1.5,
    )

    layout = build_multiple_shooting_decision_layout(ocp, 2)
    assert layout.extract(guess, "x") == pytest.approx(states)
    assert layout.extract(guess, "u") == pytest.approx(controls)
    assert layout.extract(guess, "p") == pytest.approx(
        parameters[:, None]
    )
    assert layout.extract(guess, "T").item() == pytest.approx(1.5)


def test_pack_initial_guess_rejects_wrong_matrix_shape() -> None:
    ocp = OptimalControlProblem(ParameterizedIntegratorModel())
    ocp.set_time_horizon(tf=(1.0, 2.0))

    with pytest.raises(ValueError, match="must have shape"):
        pack_initial_guess(
            ocp,
            [0.0, 0.5, 1.0],
            states=np.zeros((3, 2)),
            controls=np.zeros((1, 2)),
            parameters=np.zeros(2),
            terminal_time=1.5,
        )
