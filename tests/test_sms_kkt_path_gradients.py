import casadi as ca
import numpy as np

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.optimal_control_problem import OptimalControlProblem
from sms_ocp.sms_ia_transcription import SMSIATranscription
from sms_ocp.sms_kkt_path_gradients import (
    SMSKKTPathGradientEvaluator,
)
from sms_ocp.sms_kkt_path_sampling import (
    SMSKKTPathSampler,
)


class ControlledIntegratorModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = self.u_sym


class ParameterStateOnlyModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.p_sym = ca.SX.sym("p")
        self.ode_expr = self.p_sym


def test_matches_finite_differences_across_shooting_intervals() -> None:
    model = ControlledIntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=2.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=100.0,
        name="state_limit",
    )
    ocp.add_path_constraint(
        2.0 * model.x_sym + model.u_sym,
        ub=100.0,
        name="mixed_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
        integrator_options={
            "reltol": 1e-11,
            "abstol": 1e-13,
        },
    )
    transcription.initialize()
    sampler = SMSKKTPathSampler(
        transcription,
        samples_per_atomic_interval=3,
    )
    evaluator = SMSKKTPathGradientEvaluator(transcription)
    decision_point = np.array([0.0, 1.0, 3.0, 1.0, 2.0])
    maxima = sampler.compute(decision_point).path_maxima

    evaluation = evaluator.evaluate(
        decision_point,
        maxima,
    )

    np.testing.assert_allclose(
        evaluation.jacobian,
        [
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 3.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 0.0, 3.0],
        ],
        rtol=1e-8,
        atol=1e-9,
    )
    assert evaluation.points == maxima
    assert tuple(evaluator._state_integrator_cache) == ((1.0,),)
    assert len(evaluator._gradient_function_cache) == 1

    step = 1e-6
    finite_difference_jacobian = np.empty_like(
        evaluation.jacobian
    )
    for decision_index in range(decision_point.size):
        positive_point = decision_point.copy()
        negative_point = decision_point.copy()
        positive_point[decision_index] += step
        negative_point[decision_index] -= step
        positive_values = np.array(
            [
                maximum.value
                for maximum in (
                    sampler.compute(positive_point).path_maxima
                )
            ]
        )
        negative_values = np.array(
            [
                maximum.value
                for maximum in (
                    sampler.compute(negative_point).path_maxima
                )
            ]
        )
        finite_difference_jacobian[:, decision_index] = (
            positive_values - negative_values
        ) / (2.0 * step)

    np.testing.assert_allclose(
        evaluation.jacobian,
        finite_difference_jacobian,
        rtol=1e-6,
        atol=1e-7,
    )


def test_handles_left_endpoint_parameter_and_free_time_gradients() -> None:
    model = ParameterStateOnlyModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(t0=1.0, tf=(2.0, 4.0))
    ocp.add_path_constraint(
        -(model.t_sym + model.x_sym + model.p_sym),
        ub=100.0,
        name="decreasing_limit",
    )
    ocp.add_path_constraint(
        model.t_sym + model.x_sym + model.p_sym,
        ub=100.0,
        name="increasing_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
        integrator_options={
            "reltol": 1e-11,
            "abstol": 1e-13,
        },
    )
    transcription.initialize()
    sampler = SMSKKTPathSampler(
        transcription,
        samples_per_atomic_interval=3,
    )
    evaluator = SMSKKTPathGradientEvaluator(transcription)
    decision_point = np.array([1.0, 5.0, 2.0, 3.0])
    maxima = sampler.compute(decision_point).path_maxima

    evaluation = evaluator.evaluate(
        decision_point,
        maxima,
    )

    np.testing.assert_allclose(
        [maximum.normalized_time for maximum in maxima],
        [0.0, 1.0],
    )
    np.testing.assert_allclose(
        evaluation.jacobian,
        [
            [-1.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 3.0, 3.0],
        ],
        rtol=1e-8,
        atol=1e-9,
    )
    assert tuple(evaluator._state_integrator_cache) == ((1.0,),)
