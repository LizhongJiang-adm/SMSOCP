import casadi as ca
import numpy as np
import pytest

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.optimal_control_problem import OptimalControlProblem
from sms_ocp.sms_ia_checking_intervals import CheckingIntervalUpdate
from sms_ocp.sms_ia_transcription import SMSIAOptions, SMSIATranscription
from sms_ocp.sms_kkt import (
    build_sms_kkt_finite_reference,
)
from sms_ocp.sms_kkt_path_sampling import (
    SMSKKTPathSampler,
    SampledPathConstraintPoint,
)


class IntegratorModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = self.u_sym


class ParameterStateOnlyModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.p_sym = ca.SX.sym("p")
        self.ode_expr = self.p_sym


def test_builds_and_reuses_finite_kkt_reference() -> None:
    model = IntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.set_objective(
        mayer=model.x_sym**2,
        lagrange=model.u_sym**2,
    )
    ocp.set_variable_bounds("x", lb=-3.0, ub=3.0)
    ocp.set_variable_bounds("u", lb=-2.0, ub=2.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=10.0,
        name="sms_limit",
    )
    ocp.add_path_constraint(
        model.x_sym,
        lb=-1.0,
        ub=2.0,
        name="grid_band",
        enforcement="grid_only",
    )
    ocp.add_initial_constraint(
        model.x_sym,
        name="initial_state",
    )
    ocp.add_terminal_constraint(
        model.x_sym,
        lb=-np.inf,
        ub=1.5,
        name="terminal_upper",
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

    reference = build_sms_kkt_finite_reference(
        transcription
    )
    decision_point = np.array(
        [0.0, 0.5, 1.0, 1.0, 1.0]
    )
    evaluation = reference.evaluate(decision_point)

    np.testing.assert_allclose(
        evaluation.objective_gradient,
        [0.0, 0.0, 2.0, 1.0, 1.0],
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        evaluation.equality_values,
        [0.0, 0.0, 0.0],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        evaluation.equality_jacobian,
        [
            [1.0, -1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        evaluation.finite_inequality_values,
        [-2.0, -1.0, -1.5, -1.5, -1.0, -2.0, -0.5],
        atol=1e-10,
    )
    np.testing.assert_array_equal(
        evaluation.finite_inequality_jacobian,
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
    )
    np.testing.assert_array_equal(
        reference.decision_lower_bounds,
        [-3.0, -3.0, -3.0, -2.0, -2.0],
    )
    np.testing.assert_array_equal(
        reference.decision_upper_bounds,
        [3.0, 3.0, 3.0, 2.0, 2.0],
    )

    evaluator = reference.evaluator
    transcription.update_checking_intervals(
        (
            CheckingIntervalUpdate(
                inequality_name="sms_limit[0].upper",
                shooting_interval_index=0,
                intervals=((0.0, 0.25), (0.25, 0.5)),
            ),
        )
    )
    updated_evaluation = reference.evaluate(
        decision_point
    )

    assert reference.evaluator is evaluator
    np.testing.assert_allclose(
        updated_evaluation.objective_gradient,
        evaluation.objective_gradient,
    )
    np.testing.assert_allclose(
        updated_evaluation.equality_values,
        evaluation.equality_values,
    )
    np.testing.assert_allclose(
        updated_evaluation.equality_jacobian,
        evaluation.equality_jacobian,
    )
    np.testing.assert_allclose(
        updated_evaluation.finite_inequality_values,
        evaluation.finite_inequality_values,
    )
    np.testing.assert_allclose(
        updated_evaluation.finite_inequality_jacobian,
        evaluation.finite_inequality_jacobian,
    )


def test_finite_kkt_reference_requires_initialized_transcription() -> None:
    transcription = SMSIATranscription(
        OptimalControlProblem(IntegratorModel()),
        shooting_grid=[0.0, 1.0],
    )

    assert not transcription.is_initialized
    with pytest.raises(RuntimeError, match="Initialize"):
        build_sms_kkt_finite_reference(transcription)

    transcription.initialize()

    assert transcription.is_initialized


def test_samples_path_maxima_on_shared_atomic_grid() -> None:
    model = IntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        -(model.x_sym - 0.4) ** 2,
        ub=0.0,
        name="first_peak",
    )
    ocp.add_path_constraint(
        -(model.x_sym - 0.8) ** 2,
        ub=0.0,
        name="second_peak",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
        sms_ia_options=SMSIAOptions(
            checking_interval_overrides={
                "first_peak[0].upper": ((0.0, 0.5),),
                "second_peak[0].upper": ((0.25, 1.0),),
            },
        ),
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
    decision_point = np.array([0.0, 1.0, 1.0])

    maxima = sampler.compute(decision_point).path_maxima

    assert maxima == (
        SampledPathConstraintPoint(
            inequality_index=0,
            shooting_interval_index=0,
            inequality_checking_interval_index=0,
            shooting_interval_sample_index=3,
            normalized_time=0.375,
            time=0.375,
            value=pytest.approx(-0.000625),
        ),
        SampledPathConstraintPoint(
            inequality_index=1,
            shooting_interval_index=0,
            inequality_checking_interval_index=0,
            shooting_interval_sample_index=5,
            normalized_time=0.75,
            time=0.75,
            value=pytest.approx(-0.0025),
        ),
    )
    assert tuple(sampler._state_integrator_cache) == (
        (0.125, 0.25, 0.375, 0.5, 0.75, 1.0),
    )

    transcription.update_checking_intervals(
        (
            CheckingIntervalUpdate(
                inequality_name="first_peak[0].upper",
                shooting_interval_index=0,
                intervals=((0.0, 0.125), (0.125, 0.5)),
            ),
        )
    )
    updated_maxima = sampler.compute(decision_point).path_maxima

    assert len(updated_maxima) == 3
    assert tuple(
        maximum.inequality_checking_interval_index
        for maximum in updated_maxima
        if maximum.inequality_index == 0
    ) == (0, 1)
    assert len(sampler._state_integrator_cache) == 2


def test_path_sampler_supports_both_gradient_point_strategies() -> None:
    model = IntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=0.0,
        name="active_state",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
    )
    transcription.initialize()
    sparse_sampler = SMSKKTPathSampler(
        transcription,
        samples_per_atomic_interval=7,
        active_tolerance=1e-3,
        active_point_sample_stride=3,
        path_gradient_point_strategy="sparse_active",
    )
    maximum_sampler = SMSKKTPathSampler(
        transcription,
        samples_per_atomic_interval=7,
        active_tolerance=1e-3,
        path_gradient_point_strategy="maximum",
    )
    decision_point = [0.0, 0.0, 0.0]

    sparse_sampling = sparse_sampler.compute(decision_point)
    maximum_sampling = maximum_sampler.compute(decision_point)

    assert len(sparse_sampling.path_maxima) == 1
    np.testing.assert_allclose(
        [
            point.normalized_time
            for point in sparse_sampling.active_path_points
        ],
        [0.0, 0.5, 1.0],
    )
    np.testing.assert_allclose(
        [
            point.normalized_time
            for point in maximum_sampling.active_path_points
        ],
        [0.0],
    )


def test_path_sampler_merges_nearly_equal_constraint_boundaries() -> None:
    model = IntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="left_constraint",
    )
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="right_constraint",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
        sms_ia_options=SMSIAOptions(
            checking_interval_overrides={
                "left_constraint[0].upper": ((0.0, 0.3),),
                "right_constraint[0].upper": (
                    (0.1 + 0.2, 0.5),
                ),
            },
        ),
    )
    transcription.initialize()
    sampler = SMSKKTPathSampler(
        transcription,
        samples_per_atomic_interval=3,
    )

    maxima = sampler.compute([0.0, 1.0, 1.0]).path_maxima

    assert len(maxima) == 2
    np.testing.assert_allclose(
        [maximum.normalized_time for maximum in maxima],
        [0.3, 0.5],
    )
    assert tuple(sampler._state_integrator_cache) == (
        (0.15, 0.3, 0.4, 0.5),
    )


def test_path_sampler_uses_each_shooting_intervals_control() -> None:
    model = IntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.u_sym,
        ub=10.0,
        name="control_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()
    sampler = SMSKKTPathSampler(
        transcription,
        samples_per_atomic_interval=3,
    )

    maxima = sampler.compute(
        [0.0, 0.5, 1.5, 1.0, 2.0]
    ).path_maxima

    assert tuple(
        maximum.shooting_interval_index
        for maximum in maxima
    ) == (0, 1)
    np.testing.assert_allclose(
        [maximum.normalized_time for maximum in maxima],
        [0.0, 0.5],
    )
    np.testing.assert_allclose(
        [maximum.value for maximum in maxima],
        [-9.0, -8.0],
    )


def test_path_sampler_rejects_non_finite_path_values() -> None:
    model = IntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        ca.log(model.x_sym),
        ub=0.0,
        name="log_state",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
    )
    transcription.initialize()
    sampler = SMSKKTPathSampler(transcription)

    with pytest.raises(
        RuntimeError,
        match="non-finite values",
    ):
        sampler.compute([-1.0, -1.0, 0.0])


def test_path_sampler_supports_parameters_free_time_and_no_control() -> None:
    model = ParameterStateOnlyModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=(1.0, 3.0))
    ocp.add_path_constraint(
        model.t_sym + model.x_sym + model.p_sym,
        ub=100.0,
        name="combined_value",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
    )
    transcription.initialize()
    sampler = SMSKKTPathSampler(
        transcription,
        samples_per_atomic_interval=3,
    )

    maxima = sampler.compute(
        [1.0, 3.0, 2.0, 2.0]
    ).path_maxima

    assert len(maxima) == 1
    assert maxima[0].normalized_time == pytest.approx(1.0)
    assert maxima[0].time == pytest.approx(2.0)
    assert maxima[0].value == pytest.approx(-91.0)
