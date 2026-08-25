import casadi as ca
import numpy as np
import pytest

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.optimal_control_problem import OptimalControlProblem
from sms_ocp.sms_ia_transcription import SMSIATranscription
from sms_ocp.sms_kkt import (
    SMSKKTChecker,
    SMSKKTOptions,
    _fit_kkt_multipliers,
)


class FixedControlModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = 0.0 * self.u_sym


class ConstantStateModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.ode_expr = 0.0 * self.x_sym


def _build_scaled_path_kkt_checker() -> SMSKKTChecker:
    model = FixedControlModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.set_objective(mayer=-model.x_sym)
    ocp.set_variable_bounds("u", lb=0.0, ub=0.0)
    ocp.add_path_constraint(
        10.0 * model.x_sym,
        ub=0.0,
        name="state_limit",
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
    return SMSKKTChecker(
        transcription,
        options=SMSKKTOptions(
            samples_per_atomic_interval=3,
            stationarity_tolerance=1e-8,
            feasibility_tolerance=1e-8,
            path_constraint_scale_overrides={
                "state_limit[0].upper": 10.0,
            },
        ),
    )


def test_checker_fits_scaled_path_and_fixed_bound_multipliers() -> None:
    checker = _build_scaled_path_kkt_checker()

    result = checker.check([0.0, 0.0, 0.0])

    assert result.is_satisfied
    assert result.stationarity_residual == pytest.approx(
        0.0,
        abs=1e-10,
    )
    assert result.max_path_value == pytest.approx(0.0)
    assert result.active_set.fixed_bound_indices == (2,)
    assert result.active_set.lower_bound_indices == ()
    assert result.active_set.upper_bound_indices == ()
    assert len(result.active_set.active_path_points) == 1
    np.testing.assert_allclose(
        result.multipliers.finite_equalities,
        [-1.0],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.multipliers.fixed_bounds,
        [0.0],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.multipliers.path_inequalities,
        [1.0],
        atol=1e-8,
    )


def test_checker_reports_path_infeasibility_without_gating_stationarity() -> None:
    checker = _build_scaled_path_kkt_checker()
    checker.check([0.0, 0.0, 0.0])
    finite_evaluator = checker.finite_reference.evaluator
    path_gradient_cache_size = len(
        checker.path_gradient_evaluator._gradient_function_cache
    )

    result = checker.check([0.1, 0.1, 0.0])

    assert result.is_satisfied
    assert result.max_path_value == pytest.approx(1.0)
    assert checker.finite_reference.evaluator is finite_evaluator
    assert (
        len(checker.path_gradient_evaluator._gradient_function_cache)
        == path_gradient_cache_size
    )


def test_checker_handles_finite_constraints_without_sms_path_rows() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.set_variable_bounds("x", lb=0.0, ub=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=0.0,
        name="grid_limit",
        enforcement="grid_only",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
    )
    transcription.initialize()
    checker = SMSKKTChecker(
        transcription,
        options=SMSKKTOptions(
            samples_per_atomic_interval=3,
        ),
    )

    result = checker.check([0.0, 0.0])

    assert result.is_satisfied
    assert result.path_maxima == ()
    assert result.active_set.active_path_points == ()
    assert result.active_set.finite_inequality_indices == (0, 1)
    assert result.active_set.lower_bound_indices == (0, 1)
    assert result.active_set.upper_bound_indices == ()
    assert result.active_set.fixed_bound_indices == ()
    assert result.multipliers.path_inequalities.size == 0
    assert result.max_path_value == -np.inf


def test_checker_validates_named_path_constraint_scales() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=0.0,
        name="state_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
    )
    transcription.initialize()

    with pytest.raises(
        ValueError,
        match="Unknown path-constraint scale",
    ):
        SMSKKTChecker(
            transcription,
            options=SMSKKTOptions(
                path_constraint_scale_overrides={
                    "unknown": 1.0,
                },
            ),
        )

    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        SMSKKTOptions(
            path_constraint_scale_overrides={
                "state_limit[0].upper": 0.0,
            },
        )


def test_validates_path_gradient_point_options() -> None:
    options = SMSKKTOptions()

    assert options.active_point_sample_stride == 3
    assert options.path_gradient_point_strategy == "sparse_active"

    with pytest.raises(
        ValueError,
        match="active_point_sample_stride",
    ):
        SMSKKTOptions(active_point_sample_stride=0)

    with pytest.raises(
        ValueError,
        match="path_gradient_point_strategy",
    ):
        SMSKKTOptions(
            path_gradient_point_strategy="unknown",  # type: ignore[arg-type]
        )


def test_multiplier_fit_handles_rank_deficient_equalities() -> None:
    equality_jacobian = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )

    result = _fit_kkt_multipliers(
        objective_gradient=np.array([-1.0, 0.0]),
        equality_jacobian=equality_jacobian,
        active_inequality_jacobian=np.empty((0, 2)),
    )

    assert result.stationarity_residual == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert result.equality_multipliers.shape == (2,)
    np.testing.assert_allclose(
        equality_jacobian.T @ result.equality_multipliers,
        [1.0, 0.0],
        atol=1e-12,
    )


def test_multiplier_fit_removes_inequality_in_equality_span() -> None:
    result = _fit_kkt_multipliers(
        objective_gradient=np.array([0.0, -1.0]),
        equality_jacobian=np.array([[1.0, 0.0]]),
        active_inequality_jacobian=np.array(
            [
                [2.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )

    assert result.stationarity_residual == pytest.approx(
        0.0,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        result.inequality_multipliers,
        [0.0, 1.0],
        atol=1e-12,
    )


def test_multiplier_fit_preserves_opposite_inequality_directions() -> None:
    inequality_jacobian = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )

    result = _fit_kkt_multipliers(
        objective_gradient=np.array([1.0, 0.0]),
        equality_jacobian=np.empty((0, 2)),
        active_inequality_jacobian=inequality_jacobian,
    )

    assert result.stationarity_residual == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert np.all(result.inequality_multipliers >= 0.0)
    np.testing.assert_allclose(
        inequality_jacobian.T
        @ result.inequality_multipliers,
        [-1.0, 0.0],
        atol=1e-12,
    )
