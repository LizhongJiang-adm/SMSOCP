import casadi as ca
import numpy as np
import pytest

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.integrator import build_interval_integrator


class TimeDependentModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.u_sym = ca.SX.sym("u")
        self.p_sym = ca.SX.sym("p")
        self.ode_expr = self.u_sym + self.p_sym * self.t_sym


class StateOnlyModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.ode_expr = -self.x_sym


def test_integrates_states_and_quadratures_at_multiple_output_points() -> None:
    model = TimeDependentModel()
    output_points = np.array([0.25, 0.75, 1.0])
    integrator = build_interval_integrator(
        model,
        ca.vertcat(1.0, model.t_sym),
        output_points,
        options={
            "reltol": 1e-10,
            "abstol": 1e-12,
        },
    )

    result = integrator(
        x0=2.0,
        u=3.0,
        p=0.5,
        start_time=4.0,
        duration=2.0,
    )

    physical_durations = 2.0 * output_points
    physical_time_integrals = (
        4.0 * physical_durations
        + 0.5 * physical_durations**2
    )
    expected_state = (
        2.0
        + 3.0 * physical_durations
        + 0.5 * physical_time_integrals
    )
    expected_integrals = np.vstack(
        [
            physical_durations,
            physical_time_integrals,
        ]
    )

    np.testing.assert_allclose(
        result["state"].full(),
        expected_state.reshape(1, -1),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        result["integrals"].full(),
        expected_integrals,
        rtol=1e-9,
        atol=1e-11,
    )


def test_supports_state_propagation_without_quadratures() -> None:
    integrator = build_interval_integrator(
        StateOnlyModel(),
        ca.SX.zeros(0, 1),
        [1.0],
        name="state_only_interval",
        options={
            "reltol": 1e-10,
            "abstol": 1e-12,
        },
    )

    result = integrator(
        x0=1.0,
        start_time=0.0,
        duration=1.0,
    )

    np.testing.assert_allclose(
        result["state"].full(),
        [[np.exp(-1.0)]],
        rtol=1e-7,
    )
    assert result["integrals"].shape == (0, 1)


def test_supports_fixed_step_rk_plugin() -> None:
    model = TimeDependentModel()
    integrator = build_interval_integrator(
        model,
        ca.vertcat(1.0, model.t_sym),
        [0.5, 1.0],
        options={
            "integrator_plugin": "rk",
            "number_of_finite_elements": 8,
        },
    )

    result = integrator(
        x0=2.0,
        u=3.0,
        p=0.5,
        start_time=4.0,
        duration=2.0,
    )

    np.testing.assert_allclose(
        result["state"].full(),
        [[7.25, 13.0]],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["integrals"].full(),
        [[1.0, 2.0], [4.5, 10.0]],
        rtol=1e-12,
        atol=1e-12,
    )


def test_rejects_unknown_integrator_plugin() -> None:
    with pytest.raises(ValueError, match="integrator_plugin"):
        build_interval_integrator(
            StateOnlyModel(),
            ca.SX.zeros(0, 1),
            [1.0],
            options={"integrator_plugin": "unknown"},
        )


@pytest.mark.parametrize(
    "output_points",
    [
        [],
        [0.0],
        [0.5, 0.5],
        [1.1],
        [np.nan],
    ],
)
def test_rejects_invalid_output_points(
    output_points: list[float],
) -> None:
    with pytest.raises(ValueError, match="within"):
        build_interval_integrator(
            StateOnlyModel(),
            ca.SX.zeros(0, 1),
            output_points,
        )
