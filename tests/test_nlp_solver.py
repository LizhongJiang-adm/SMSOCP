import casadi as ca
import numpy as np
import pytest

from sms_ocp.nlp import ExplicitNlp, solve_nlp


def test_solve_nlp_solves_scalar_quadratic() -> None:
    x = ca.MX.sym("x")
    nlp = ExplicitNlp(
        x=x,
        f=(x - 2.0) ** 2,
        g=ca.MX.zeros(0, 1),
        lbx=np.array([-np.inf]),
        ubx=np.array([np.inf]),
        lbg=np.empty(0),
        ubg=np.empty(0),
    )

    result = solve_nlp(
        nlp,
        [0.0],
        name="test_scalar_quadratic",
        solver_options={"ipopt.tol": 1e-10},
    )

    assert result.is_accepted
    assert result.decision_vector[0] == pytest.approx(2.0, abs=1e-8)
    assert result.objective == pytest.approx(0.0, abs=1e-12)
    assert result.max_violation == 0.0
