from dataclasses import replace
from typing import Literal

import casadi as ca
import numpy as np
import pytest

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.optimal_control_problem import OptimalControlProblem
from sms_ocp.sms_algorithm import (
    SMSAlgorithmOptions,
)
from sms_ocp.sms_kkt import SMSKKTChecker, SMSKKTOptions
from sms_ocp.sms_solver import solve_sms_ocp


class IntegratorModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = self.u_sym


class ConstantStateModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.ode_expr = 0.0 * self.x_sym


class OscillatingStateModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.ode_expr = (
            2.0
            * np.pi
            * ca.cos(2.0 * np.pi * self.t_sym)
        )


def test_runs_phase_one_phase_two_and_kkt(capsys) -> None:
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

    result = solve_sms_ocp(
        ocp,
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0, 1.0, 1.0],
        solver_options={"ipopt.tol": 1e-10},
        kkt_options=SMSKKTOptions(
            samples_per_atomic_interval=5,
            active_tolerance=1e-5,
            stationarity_tolerance=1e-5,
            feasibility_tolerance=1e-7,
        ),
    )

    assert result.is_success
    assert result.status == "kkt_satisfied"
    assert result.phase_one.is_accepted
    assert result.phase_one_rho <= 0.0
    assert result.phase_one_refinement_count == 0
    assert result.phase_two_refinement_count == 0
    assert result.phase_two is not None
    assert result.phase_two.is_accepted
    assert result.phase_two.objective == pytest.approx(
        1.0,
        abs=1e-7,
    )
    assert result.kkt_check is not None
    assert result.kkt_check.is_satisfied
    for phase in ("phase_one", "phase_two"):
        prefix = f"{phase}_iteration_0"
        for component in (
            "assembly",
            "solver_build",
            "solve",
        ):
            assert (
                result.timing[f"{prefix}_{component}_seconds"]
                >= 0.0
            )
    assert (
        result.timing[
            "phase_two_iteration_0_kkt_check_seconds"
        ]
        >= 0.0
    )
    progress = capsys.readouterr().out
    assert "SMS Algorithm Progress" in progress
    assert "Phase I" in progress
    assert "Phase II" in progress
    assert "Checking intervals" in progress
    assert "NLP status" in progress
    assert "IPOPT iter" in progress
    assert "KKT [s]" in progress
    assert "KKT stat. residual" in progress
    phase_one_row = next(
        line for line in progress.splitlines()
        if line.startswith("Phase I ")
    )
    phase_one_columns = phase_one_row.split("|")
    assert phase_one_columns[2].strip() == "2"
    assert phase_one_columns[9].strip() == "-"
    assert phase_one_columns[10].strip() == "-"
    phase_two_row = next(
        line for line in progress.splitlines()
        if line.startswith("Phase II")
    )
    phase_two_columns = phase_two_row.split("|")
    assert float(phase_two_columns[9]) >= 0.0
    assert result.kkt_check is not None
    assert float(phase_two_columns[10]) == pytest.approx(
        result.kkt_check.stationarity_residual,
        rel=1e-6,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("compensation_mode", "shift_multiplier"),
    (
        ("none", 0),
        ("basic", 1),
        ("cumulative", 3),
    ),
)
def test_applies_selected_nlp_tolerance_compensation(
    compensation_mode: Literal[
        "none",
        "basic",
        "cumulative",
    ],
    shift_multiplier: int,
) -> None:
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
    constraint_tolerance = 1e-5
    expected_shift = shift_multiplier * constraint_tolerance

    result = solve_sms_ocp(
        ocp,
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0, 1.0, 1.0],
        solver_options={
            "ipopt.constr_viol_tol": constraint_tolerance,
            "ipopt.tol": 1e-10,
        },
        algorithm_options=SMSAlgorithmOptions(
            nlp_tolerance_compensation=compensation_mode,
        ),
        kkt_options=SMSKKTOptions(
            samples_per_atomic_interval=5,
            active_tolerance=1e-5,
            stationarity_tolerance=1e-5,
            feasibility_tolerance=1e-7,
        ),
    )

    sms_slice = (
        result.transcription.sms_upper_bound_constraint_slice
    )
    expected_upper_bounds = (
        result.transcription.constraint_upper_bounds.copy()
    )
    expected_upper_bounds[sms_slice] -= expected_shift
    np.testing.assert_array_equal(
        result.phase_one.nlp.ubg,
        expected_upper_bounds,
    )
    assert result.phase_two is not None
    np.testing.assert_array_equal(
        result.phase_two.nlp.ubg,
        expected_upper_bounds,
    )
    assert np.max(
        result.phase_one.constraint_values[sms_slice]
        + result.phase_one_rho
    ) <= 0.0
    assert np.max(
        result.phase_two.constraint_values[sms_slice]
    ) <= 0.0


def test_stops_after_unsuccessful_feasibility_restoration() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_initial_constraint(model.x_sym, 0.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=-1.0,
        name="impossible_state_limit",
    )

    result = solve_sms_ocp(
        ocp,
        [0.0, 1.0],
        [0.0, 0.0],
        algorithm_options=SMSAlgorithmOptions(
            max_phase_one_refinements=1,
        ),
        solver_options={"ipopt.tol": 1e-10},
    )

    assert result.status == "phase_one_failed"
    assert not result.is_success
    assert result.phase_one.solver_success
    assert result.phase_one.is_accepted
    assert result.phase_one_rho == pytest.approx(
        1.0,
        abs=1e-3,
    )
    assert result.phase_one_refinement_count == 1
    assert len(
        result.transcription.checking_intervals[0].intervals
    ) == 2
    assert result.phase_two is None
    assert result.kkt_check is None


def test_phase_one_bisects_active_intervals_until_rho_is_nonpositive() -> None:
    model = OscillatingStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_initial_constraint(model.x_sym, 0.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.1,
        name="state_limit",
    )

    result = solve_sms_ocp(
        ocp,
        [0.0, 1.0],
        [0.0, 0.0],
        algorithm_options=SMSAlgorithmOptions(
            max_phase_one_refinements=2,
        ),
        integrator_options={
            "abstol": 1e-10,
            "reltol": 1e-10,
        },
        solver_options={"ipopt.tol": 1e-10},
    )

    assert result.status == "kkt_satisfied"
    assert result.phase_one_refinement_count == 1
    assert result.phase_one_rho <= 0.0
    assert result.transcription.checking_intervals[
        0
    ].intervals == (
        (0.0, 0.5),
        (0.5, 1.0),
    )


def test_phase_two_bisects_active_intervals_after_failed_kkt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = IntegratorModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.set_objective(mayer=-model.x_sym)
    ocp.set_variable_bounds("u", lb=-2.0, ub=2.0)
    ocp.add_initial_constraint(model.x_sym, 0.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="state_limit",
    )

    original_check = SMSKKTChecker.check
    check_count = 0

    def force_one_refinement(
        checker: SMSKKTChecker,
        decision_point: np.ndarray,
    ):
        nonlocal check_count
        check = original_check(checker, decision_point)
        check_count += 1
        return replace(
            check,
            is_satisfied=check_count >= 2,
        )

    monkeypatch.setattr(
        SMSKKTChecker,
        "check",
        force_one_refinement,
    )

    result = solve_sms_ocp(
        ocp,
        [0.0, 1.0],
        [0.0, 0.0, 0.0],
        algorithm_options=SMSAlgorithmOptions(
            max_phase_two_refinements=1,
            sms_upper_bound_active_tolerance=1e-5,
        ),
        solver_options={"ipopt.tol": 1e-10},
    )

    assert result.status == "kkt_satisfied"
    assert result.phase_two_refinement_count == 1
    assert result.transcription.checking_intervals[
        0
    ].intervals == (
        (0.0, 0.5),
        (0.5, 1.0),
    )


def test_validates_initial_guess_before_solving() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="state_limit",
    )

    with pytest.raises(
        ValueError,
        match="initial_guess must contain 2 entries",
    ):
        solve_sms_ocp(
            ocp,
            [0.0, 1.0],
            [0.0],
        )


def test_validates_algorithm_tolerances() -> None:
    default_options = SMSAlgorithmOptions()
    assert default_options.rho_lower_bound == pytest.approx(-1e-3)
    assert default_options.nlp_tolerance_compensation == "basic"
    assert default_options.print_progress
    assert not default_options.print_ipopt_output
    assert SMSAlgorithmOptions(
        print_ipopt_output=True
    ).print_ipopt_output

    with pytest.raises(
        ValueError,
        match="sms_upper_bound_active_tolerance",
    ):
        SMSAlgorithmOptions(
            sms_upper_bound_active_tolerance=-1.0,
        )

    with pytest.raises(
        ValueError,
        match="nlp_tolerance_compensation",
    ):
        SMSAlgorithmOptions(
            nlp_tolerance_compensation="strong",
        )


def test_validates_algorithm_refinement_counts() -> None:
    with pytest.raises(
        ValueError,
        match="max_phase_one_refinements",
    ):
        SMSAlgorithmOptions(
            max_phase_one_refinements=-1,
        )


def test_rejects_invalid_ipopt_constraint_violation_tolerance() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="state_limit",
    )

    with pytest.raises(
        ValueError,
        match="ipopt.constr_viol_tol must be finite and positive",
    ):
        solve_sms_ocp(
            ocp,
            [0.0, 1.0],
            [0.0, 0.0],
            solver_options={"ipopt.constr_viol_tol": 0.0},
        )
