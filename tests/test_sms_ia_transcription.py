import warnings

import casadi as ca
import numpy as np
import pytest

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.nlp import ExplicitNlp
from sms_ocp.optimal_control_problem import OptimalControlProblem
from sms_ocp.sms_ia_checking_intervals import (
    CheckingIntervalUpdate,
    CheckingIntervalPlan,
    InequalityCheckingIntervals,
    ShootingIntervalPlan,
    build_shooting_interval_plans,
)
from sms_ocp.sms_ia_inequality_path_constraints import (
    build_sms_ia_growth_expr,
    scalarize_sms_ia_path_constraints,
)
from sms_ocp.sms_ia_transcription import (
    SMSIAConstraintRowInfo,
    SMSIAOptions,
    SMSIATranscription,
)


class ThreeStateModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 3)
        self.ode_expr = -self.x_sym


class GrowthModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.u_sym = ca.SX.sym("u")
        self.p_sym = ca.SX.sym("p")
        self.ode_expr = self.u_sym + self.p_sym


class ConstantStateModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.ode_expr = ca.SX.zeros(1, 1)


def test_builds_shooting_interval_plans() -> None:
    plans = build_shooting_interval_plans(
        shooting_grid=[0.0, 0.5, 1.0],
        checking_intervals=(
            InequalityCheckingIntervals(
                inequality_index=0,
                intervals=((0.125, 0.5), (0.5, 1.0)),
            ),
            InequalityCheckingIntervals(
                inequality_index=1,
                intervals=((0.25, 0.5),),
            ),
            InequalityCheckingIntervals(
                inequality_index=2,
                intervals=((0.5, 0.75), (0.75, 1.0)),
            ),
        ),
    )

    assert plans == (
        ShootingIntervalPlan(
            output_points=(0.25, 0.5, 1.0),
            checking_intervals=(
                CheckingIntervalPlan(0, 0, 1, 3),
                CheckingIntervalPlan(1, 0, 2, 3),
            ),
        ),
        ShootingIntervalPlan(
            output_points=(0.5, 1.0),
            checking_intervals=(
                CheckingIntervalPlan(0, 1, 0, 2),
                CheckingIntervalPlan(2, 0, 0, 1),
                CheckingIntervalPlan(2, 1, 1, 2),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("shooting_grid", "message"),
    [
        ([0.0, np.nan, 1.0], "only finite values"),
        ([0.0, np.inf, 1.0], "only finite values"),
        ([0.0, 1.0, 2.0], "normalized time from 0 to 1"),
        ([-1.0, 0.0, 1.0], "normalized time from 0 to 1"),
    ],
)
def test_rejects_invalid_normalized_shooting_grid(
    shooting_grid: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SMSIATranscription(
            OptimalControlProblem(ConstantStateModel()),
            shooting_grid,
        )


def test_builds_initial_checking_interval_overrides() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="overridden_limit",
    )
    ocp.add_path_constraint(
        model.x_sym,
        ub=2.0,
        name="default_limit",
    )
    shooting_node = 0.1 + 0.2

    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, shooting_node, 1.0],
        sms_ia_options=SMSIAOptions(
            smoothing_overrides={
                "overridden_limit[0].upper": 2e-3,
            },
            checking_interval_overrides={
                "overridden_limit[0].upper": (
                    (0.3, 1.0),
                ),
            },
        ),
    )

    transcription.initialize()
    nlp = transcription.build_sms_ia_nlp()

    assert tuple(
        inequality.smoothing_parameter
        for inequality in transcription.scalar_inequalities
    ) == (2e-3, 1e-3)
    assert transcription.checking_intervals == (
        InequalityCheckingIntervals(
            inequality_index=0,
            intervals=((shooting_node, 1.0),),
        ),
        InequalityCheckingIntervals(
            inequality_index=1,
            intervals=(
                (0.0, shooting_node),
                (shooting_node, 1.0),
            ),
        ),
    )
    assert tuple(
        len(plan.checking_intervals)
        for plan in transcription.interval_plans
    ) == (1, 2)
    assert len(transcription._integrator_cache) == 1
    assert nlp.x.shape == (3, 1)
    assert nlp.g.shape == (5, 1)
    np.testing.assert_array_equal(nlp.ubg, np.zeros(5))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            SMSIAOptions(
                smoothing_overrides={
                    "missing[0].upper": 1e-3,
                },
            ),
            "Unknown smoothing override names",
        ),
        (
            SMSIAOptions(
                checking_interval_overrides={
                    "missing[0].upper": ((0.0, 0.5),),
                },
            ),
            "Unknown checking-interval override names",
        ),
        (
            SMSIAOptions(
                checking_interval_overrides={
                    "state_limit[0].upper": ((0.25, 0.75),),
                },
            ),
            "must not cross a shooting node",
        ),
    ],
)
def test_initialize_rejects_invalid_initial_options(
    options: SMSIAOptions,
    message: str,
) -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="state_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
        sms_ia_options=options,
    )

    with pytest.raises(ValueError, match=message):
        transcription.initialize()


def test_updates_checking_intervals_by_name_and_rebuilds_only_affected_block(
) -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="first_limit",
    )
    ocp.add_path_constraint(
        model.x_sym,
        ub=2.0,
        name="second_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()

    current_checking_intervals = transcription.checking_intervals
    current_interval_plans = transcription.interval_plans
    current_shooting_blocks = transcription.shooting_interval_blocks
    scalar_inequalities = transcription.scalar_inequalities
    node_constraint_vector = transcription.node_constraint_vector
    decision_layout = transcription.decision_layout

    transcription.update_checking_intervals(
        (
            CheckingIntervalUpdate(
                inequality_name="first_limit[0].upper",
                shooting_interval_index=1,
                intervals=(
                    (0.5, 0.75),
                    (0.75, 1.0),
                ),
            ),
        )
    )

    assert transcription.scalar_inequality_names == (
        "first_limit[0].upper",
        "second_limit[0].upper",
    )
    assert transcription.checking_intervals == (
        InequalityCheckingIntervals(
            inequality_index=0,
            intervals=(
                (0.0, 0.5),
                (0.5, 0.75),
                (0.75, 1.0),
            ),
        ),
        InequalityCheckingIntervals(
            inequality_index=1,
            intervals=((0.0, 0.5), (0.5, 1.0)),
        ),
    )
    assert current_checking_intervals == (
        InequalityCheckingIntervals(
            inequality_index=0,
            intervals=((0.0, 0.5), (0.5, 1.0)),
        ),
        InequalityCheckingIntervals(
            inequality_index=1,
            intervals=((0.0, 0.5), (0.5, 1.0)),
        ),
    )
    assert transcription.interval_plans[0] is current_interval_plans[0]
    assert transcription.interval_plans[1] is not current_interval_plans[1]
    assert (
        transcription.shooting_interval_blocks[0]
        is current_shooting_blocks[0]
    )
    assert (
        transcription.shooting_interval_blocks[1]
        is not current_shooting_blocks[1]
    )
    assert transcription.scalar_inequalities is scalar_inequalities
    assert transcription.node_constraint_vector is node_constraint_vector
    assert transcription.decision_layout is decision_layout
    assert transcription.sms_upper_bound_constraint_slice == slice(2, 7)
    assert transcription.sms_ia_constraint_rows == (
        SMSIAConstraintRowInfo(2, 0, 0, 0),
        SMSIAConstraintRowInfo(3, 1, 0, 0),
        SMSIAConstraintRowInfo(4, 0, 1, 1),
        SMSIAConstraintRowInfo(5, 0, 1, 2),
        SMSIAConstraintRowInfo(6, 1, 1, 1),
    )
    assert transcription.build_sms_ia_nlp().g.shape == (7, 1)
    assert transcription.build_sms_fr_nlp().g.shape == (7, 1)


def test_accumulates_updates_for_one_inequality_across_shooting_intervals(
) -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="state_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()

    transcription.update_checking_intervals(
        (
            CheckingIntervalUpdate(
                inequality_name="state_limit[0].upper",
                shooting_interval_index=0,
                intervals=((0.0, 0.25), (0.25, 0.5)),
            ),
            CheckingIntervalUpdate(
                inequality_name="state_limit[0].upper",
                shooting_interval_index=1,
                intervals=((0.5, 0.75), (0.75, 1.0)),
            ),
        )
    )

    assert transcription.checking_intervals == (
        InequalityCheckingIntervals(
            inequality_index=0,
            intervals=(
                (0.0, 0.25),
                (0.25, 0.5),
                (0.5, 0.75),
                (0.75, 1.0),
            ),
        ),
    )
    assert transcription.sms_upper_bound_constraint_vector.shape == (4, 1)


def test_refreshes_later_global_checking_interval_indices_without_rebuilding(
) -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="state_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()

    old_second_block = transcription.shooting_interval_blocks[1]
    assert (
        old_second_block.plan.checking_intervals[
            0
        ].inequality_checking_interval_index
        == 1
    )

    transcription.update_checking_intervals(
        (
            CheckingIntervalUpdate(
                inequality_name="state_limit[0].upper",
                shooting_interval_index=0,
                intervals=((0.0, 0.25), (0.25, 0.5)),
            ),
        )
    )

    updated_second_block = transcription.shooting_interval_blocks[1]
    assert updated_second_block is not old_second_block
    assert (
        updated_second_block.defect_constraint_block
        is old_second_block.defect_constraint_block
    )
    assert (
        updated_second_block.lagrange_cost_contribution
        is old_second_block.lagrange_cost_contribution
    )
    assert (
        updated_second_block.sms_upper_bound_constraint_blocks
        is old_second_block.sms_upper_bound_constraint_blocks
    )
    assert (
        updated_second_block.plan.checking_intervals[
            0
        ].inequality_checking_interval_index
        == 2
    )
    assert transcription.sms_ia_constraint_rows == (
        SMSIAConstraintRowInfo(2, 0, 0, 0),
        SMSIAConstraintRowInfo(3, 0, 0, 1),
        SMSIAConstraintRowInfo(4, 0, 1, 2),
    )


def test_checking_interval_no_op_preserves_current_state() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="state_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()

    checking_intervals = transcription.checking_intervals
    interval_plans = transcription.interval_plans
    shooting_blocks = transcription.shooting_interval_blocks
    constraint_vector = transcription.constraint_vector

    transcription.update_checking_intervals(
        (
            CheckingIntervalUpdate(
                inequality_name="state_limit[0].upper",
                shooting_interval_index=0,
                intervals=((0.0, 0.5),),
            ),
        )
    )

    assert transcription.checking_intervals is checking_intervals
    assert all(
        current_plan is previous_plan
        for current_plan, previous_plan in zip(
            transcription.interval_plans,
            interval_plans,
            strict=True,
        )
    )
    assert transcription.shooting_interval_blocks is shooting_blocks
    assert transcription.constraint_vector is constraint_vector


@pytest.mark.parametrize(
    ("updates", "exception", "message"),
    [
        (
            (
                CheckingIntervalUpdate(
                    inequality_name="missing[0].upper",
                    shooting_interval_index=0,
                    intervals=((0.0, 0.5),),
                ),
            ),
            ValueError,
            "Unknown scalar inequality name",
        ),
        (
            (
                CheckingIntervalUpdate(
                    inequality_name="state_limit[0].upper",
                    shooting_interval_index=2,
                    intervals=((0.0, 0.5),),
                ),
            ),
            IndexError,
            "shooting_interval_index must be within",
        ),
        (
            (
                CheckingIntervalUpdate(
                    inequality_name="state_limit[0].upper",
                    shooting_interval_index=0,
                    intervals=((0.25, 0.75),),
                ),
            ),
            ValueError,
            "must lie within the shooting grid",
        ),
        (
            (
                CheckingIntervalUpdate(
                    inequality_name="state_limit[0].upper",
                    shooting_interval_index=0,
                    intervals=((0.0, 0.25), (0.25, 0.5)),
                ),
                CheckingIntervalUpdate(
                    inequality_name="state_limit[0].upper",
                    shooting_interval_index=0,
                    intervals=((0.0, 0.5),),
                ),
            ),
            ValueError,
            "may be updated at most once",
        ),
    ],
)
def test_rejects_invalid_checking_interval_updates_without_changing_state(
    updates: tuple[CheckingIntervalUpdate, ...],
    exception: type[Exception],
    message: str,
) -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="state_limit",
    )
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()

    checking_intervals = transcription.checking_intervals
    interval_plans = transcription.interval_plans
    shooting_blocks = transcription.shooting_interval_blocks

    with pytest.raises(exception, match=message):
        transcription.update_checking_intervals(updates)

    assert transcription.checking_intervals is checking_intervals
    assert all(
        current_plan is previous_plan
        for current_plan, previous_plan in zip(
            transcription.interval_plans,
            interval_plans,
            strict=True,
        )
    )
    assert transcription.shooting_interval_blocks is shooting_blocks


def test_requires_initialization_before_updating_checking_intervals() -> None:
    transcription = SMSIATranscription(
        OptimalControlProblem(ConstantStateModel()),
        shooting_grid=[0.0, 1.0],
    )

    with pytest.raises(RuntimeError, match="must be initialized"):
        transcription.update_checking_intervals(
            (
                CheckingIntervalUpdate(
                    inequality_name="missing[0].upper",
                    shooting_interval_index=0,
                    intervals=((0.0, 1.0),),
                ),
            )
        )


def test_creates_multiple_shooting_decision_variables() -> None:
    ocp = OptimalControlProblem(GrowthModel())

    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.4, 1.0],
    )

    assert tuple(
        block.name for block in transcription.decision_layout.blocks
    ) == ("x", "u", "p", "T")
    assert transcription.state_matrix.shape == (1, 3)
    assert len(transcription.state_nodes) == 3
    assert transcription.control_matrix.shape == (1, 2)
    assert len(transcription.control_intervals) == 2
    assert transcription.parameter_vector.shape == (1, 1)
    assert transcription.terminal_time.shape == (1, 1)
    assert transcription.decision_vector.shape == (7, 1)
    assert tuple(
        block.name
        for block in transcription.sms_fr_decision_layout.blocks
    ) == ("x", "u", "p", "T", "rho")
    assert transcription.sms_fr_decision_vector.shape == (8, 1)
    assert transcription.rho_index == 7

    values = np.arange(8, dtype=float)
    np.testing.assert_array_equal(
        transcription.sms_fr_decision_layout.extract(values, "x"),
        [[0.0, 1.0, 2.0]],
    )
    np.testing.assert_array_equal(
        transcription.sms_fr_decision_layout.extract(values, "u"),
        [[3.0, 4.0]],
    )
    np.testing.assert_array_equal(
        transcription.sms_fr_decision_layout.extract(values, "p"),
        [[5.0]],
    )
    np.testing.assert_array_equal(
        transcription.sms_fr_decision_layout.extract(values, "T"),
        [[6.0]],
    )
    np.testing.assert_array_equal(
        transcription.sms_fr_decision_layout.extract(values, "rho"),
        [[7.0]],
    )


def test_omits_absent_and_fixed_decision_blocks() -> None:
    ocp = OptimalControlProblem(ThreeStateModel())
    ocp.set_time_horizon(t0=1.0, tf=3.0)

    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )

    assert tuple(
        block.name for block in transcription.decision_layout.blocks
    ) == ("x",)
    assert not transcription.decision_layout.has_block("u")
    assert not transcription.decision_layout.has_block("p")
    assert not transcription.decision_layout.has_block("T")
    assert transcription.control_matrix is None
    assert transcription.control_intervals == ()
    assert transcription.parameter_vector is None
    assert transcription.terminal_time == 3.0
    assert transcription.decision_vector.shape == (9, 1)
    assert tuple(
        block.name
        for block in transcription.sms_fr_decision_layout.blocks
    ) == ("x", "rho")
    assert transcription.sms_fr_decision_vector.shape == (10, 1)
    assert transcription.rho_index == 9


def test_initializes_long_lived_transcription_content_once() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="sms_limit",
    )
    ocp.add_path_constraint(
        model.x_sym,
        lb=-2.0,
        ub=2.0,
        name="grid_limit",
        enforcement="grid_only",
    )
    ocp.add_initial_constraint(model.x_sym, name="initial_state")
    ocp.add_terminal_constraint(model.x_sym, name="terminal_state")

    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()

    assert len(transcription.scalar_inequalities) == 1
    assert len(transcription.node_constraint_blocks) == 5
    assert transcription.node_constraint_vector.shape == (5, 1)
    assert transcription.decision_lower_bounds.shape == (3,)
    assert transcription.decision_upper_bounds.shape == (3,)
    assert len(transcription.interval_plans) == 2
    assert transcription.defect_constraint_vector.shape == (2, 1)
    assert transcription.sms_upper_bound_constraint_vector.shape == (2, 1)
    assert transcription.constraint_vector.shape == (9, 1)
    assert len(transcription._integrator_cache) == 1

    scalar_inequalities = transcription.scalar_inequalities
    node_constraint_vector = transcription.node_constraint_vector
    integrator_cache = transcription._integrator_cache
    decision_lower_bounds = transcription.decision_lower_bounds
    static_objective_expr = transcription.static_objective_expr
    checking_intervals = transcription.checking_intervals
    shooting_interval_blocks = transcription.shooting_interval_blocks
    defect_constraint_vector = transcription.defect_constraint_vector
    lagrange_cost = transcription.lagrange_cost
    constraint_vector = transcription.constraint_vector

    transcription.initialize()
    transcription.build_sms_ia_nlp()

    assert transcription.scalar_inequalities is scalar_inequalities
    assert transcription.node_constraint_vector is node_constraint_vector
    assert transcription._integrator_cache is integrator_cache
    assert transcription.decision_lower_bounds is decision_lower_bounds
    assert transcription.static_objective_expr is static_objective_expr
    assert transcription.checking_intervals is checking_intervals
    assert (
        transcription.shooting_interval_blocks
        is shooting_interval_blocks
    )
    assert transcription.defect_constraint_vector is defect_constraint_vector
    assert transcription.lagrange_cost is lagrange_cost
    assert transcription.constraint_vector is constraint_vector


def test_build_sms_ia_nlp_warns_and_initializes() -> None:
    transcription = SMSIATranscription(
        OptimalControlProblem(ConstantStateModel()),
        shooting_grid=[0.0, 1.0],
    )

    with pytest.warns(
        UserWarning,
        match="Call initialize\\(\\).*Initializing automatically",
    ):
        nlp = transcription.build_sms_ia_nlp()

    assert transcription._initialized
    assert nlp.x is transcription.decision_vector
    assert nlp.g is transcription.constraint_vector
    assert nlp.x.shape == (transcription.decision_layout.size, 1)
    assert transcription.sms_fr_decision_vector.shape == (
        transcription.decision_layout.size + 1,
        1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rebuilt_nlp = transcription.build_sms_ia_nlp()

    assert rebuilt_nlp.x is nlp.x
    assert rebuilt_nlp.g is nlp.g
    assert rebuilt_nlp.lbx is nlp.lbx
    assert rebuilt_nlp.lbg is nlp.lbg


def test_builds_sms_fr_nlp_by_relaxing_only_sms_rows() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(
        model.x_sym,
        ub=1.0,
        name="sms_limit",
    )
    ocp.add_path_constraint(
        model.x_sym,
        lb=-2.0,
        ub=2.0,
        name="grid_limit",
        enforcement="grid_only",
    )
    ocp.add_initial_constraint(model.x_sym, name="initial_state")

    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()
    sms_ia_nlp = transcription.build_sms_ia_nlp()
    sms_fr_nlp = transcription.build_sms_fr_nlp()

    assert sms_fr_nlp.x is transcription.sms_fr_decision_vector
    assert sms_fr_nlp.f is transcription.rho_symbol
    assert sms_fr_nlp.x.shape == (
        transcription.decision_layout.size + 1,
        1,
    )
    assert sms_fr_nlp.g.shape == sms_ia_nlp.g.shape
    assert sms_fr_nlp.lbg is sms_ia_nlp.lbg
    assert sms_fr_nlp.ubg is sms_ia_nlp.ubg
    np.testing.assert_array_equal(
        sms_fr_nlp.lbx[:-1],
        sms_ia_nlp.lbx,
    )
    np.testing.assert_array_equal(
        sms_fr_nlp.ubx[:-1],
        sms_ia_nlp.ubx,
    )
    assert np.isneginf(sms_fr_nlp.lbx[-1])
    assert np.isposinf(sms_fr_nlp.ubx[-1])

    rho_jacobian = ca.Function(
        "sms_fr_rho_jacobian",
        [sms_fr_nlp.x],
        [ca.jacobian(sms_fr_nlp.g, transcription.rho_symbol)],
    )
    rho_derivatives = np.asarray(
        rho_jacobian(np.zeros(transcription.sms_fr_decision_layout.size))
    ).reshape(-1)
    expected_derivatives = np.zeros(int(sms_fr_nlp.g.numel()))
    expected_derivatives[
        transcription.sms_upper_bound_constraint_slice
    ] = -1.0
    np.testing.assert_array_equal(
        rho_derivatives,
        expected_derivatives,
    )

    bounded_sms_fr_nlp = transcription.build_sms_fr_nlp(
        rho_lower_bound=-1e-4,
    )
    assert bounded_sms_fr_nlp.lbx[-1] == pytest.approx(-1e-4)


def test_build_sms_fr_nlp_warns_and_initializes() -> None:
    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(model.x_sym, ub=1.0, name="sms_limit")
    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 1.0],
    )

    with pytest.warns(
        UserWarning,
        match="Call initialize\\(\\).*Initializing automatically",
    ):
        nlp = transcription.build_sms_fr_nlp()

    assert transcription._initialized
    assert nlp.x is transcription.sms_fr_decision_vector


@pytest.mark.parametrize(
    "rho_lower_bound",
    [np.nan, 1e-4, np.inf],
)
def test_build_sms_fr_nlp_rejects_invalid_rho_lower_bound(
    rho_lower_bound: float,
) -> None:
    transcription = SMSIATranscription(
        OptimalControlProblem(ConstantStateModel()),
        shooting_grid=[0.0, 1.0],
    )

    with pytest.raises(
        ValueError,
        match="must be nonpositive and not NaN",
    ):
        transcription.build_sms_fr_nlp(
            rho_lower_bound=rho_lower_bound,
        )

    assert not transcription._initialized


def test_build_sms_fr_nlp_requires_an_sms_upper_bound() -> None:
    transcription = SMSIATranscription(
        OptimalControlProblem(ConstantStateModel()),
        shooting_grid=[0.0, 1.0],
    )
    transcription.initialize()

    with pytest.raises(
        ValueError,
        match="requires at least one SMS-IA upper-bound constraint",
    ):
        transcription.build_sms_fr_nlp()


def test_assembles_objective_and_decision_bounds() -> None:
    model = GrowthModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(t0=1.0, tf=(2.0, 4.0))
    ocp.set_objective(
        mayer=(
            model.t_sym
            + model.x_sym
            + model.u_sym
            + model.p_sym
        ),
        lagrange=model.u_sym**2,
        min_time_weight=0.5,
    )
    ocp.set_variable_bounds("x", lb=-1.0, ub=1.0)
    ocp.set_variable_bounds("u", lb=-2.0, ub=2.0)
    ocp.set_variable_bounds("p", lb=0.0, ub=6.0)

    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()
    transcription.lagrange_cost = ca.MX(2.0)
    transcription.constraint_vector = ca.MX.zeros(0, 1)
    transcription.constraint_lower_bounds = np.empty(0)
    transcription.constraint_upper_bounds = np.empty(0)

    nlp = transcription.build_sms_ia_nlp()
    objective_function = ca.Function(
        "assembled_objective",
        [nlp.x],
        [nlp.f],
    )
    objective_value = objective_function(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0]
    )

    assert isinstance(nlp, ExplicitNlp)
    assert float(objective_value) == pytest.approx(17.0)
    assert nlp.g.shape == (0, 1)
    np.testing.assert_array_equal(
        nlp.lbx,
        [-1.0, -1.0, -1.0, -2.0, -2.0, 0.0, 2.0],
    )
    np.testing.assert_array_equal(
        nlp.ubx,
        [1.0, 1.0, 1.0, 2.0, 2.0, 6.0, 4.0],
    )
    np.testing.assert_array_equal(nlp.lbg, np.empty(0))
    np.testing.assert_array_equal(nlp.ubg, np.empty(0))


def test_scalarizes_sms_ia_bounds_in_stable_order() -> None:
    model = ThreeStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(
        model.x_sym,
        lb=[-np.inf, -2.0, -np.inf],
        ub=[1.0, 2.0, np.inf],
        name="state_band",
    )
    ocp.add_path_constraint(
        model.x_sym[2],
        lb=-1.0,
        ub=1.0,
        name="grid_state",
        enforcement="grid_only",
    )

    inequalities = scalarize_sms_ia_path_constraints(
        ocp.path_constraints,
        default_smoothing_parameter=1e-3,
        smoothing_overrides={
            "state_band[1].upper": 2e-3,
            "state_band[1].lower": 3e-3,
        },
    )

    assert tuple(item.name for item in inequalities) == (
        "state_band[0].upper",
        "state_band[1].upper",
        "state_band[1].lower",
    )
    assert tuple(item.source_constraint_name for item in inequalities) == (
        "state_band",
        "state_band",
        "state_band",
    )
    assert tuple(item.component_index for item in inequalities) == (0, 1, 1)
    assert tuple(item.side for item in inequalities) == (
        "upper",
        "upper",
        "lower",
    )
    assert tuple(item.source_bound for item in inequalities) == (1.0, 2.0, -2.0)
    assert tuple(item.smoothing_parameter for item in inequalities) == (
        1e-3,
        2e-3,
        3e-3,
    )

    expression = ca.vertcat(*(item.expr for item in inequalities))
    function = model.create_function("scalar_path_inequalities", expression)
    value = function(0.0, [3.0, 4.0, 5.0])

    np.testing.assert_array_equal(
        value.full().reshape(-1),
        np.array([2.0, 2.0, -6.0]),
    )


def test_scalarization_skips_constraints_without_finite_sms_ia_bounds() -> None:
    model = ThreeStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(
        model.x_sym,
        lb=-np.inf,
        ub=np.inf,
        name="unbounded_sms",
    )
    ocp.add_path_constraint(
        model.x_sym,
        lb=-1.0,
        ub=1.0,
        name="bounded_grid",
        enforcement="grid_only",
    )

    inequalities = scalarize_sms_ia_path_constraints(ocp.path_constraints)

    assert inequalities == ()


@pytest.mark.parametrize("parameter", [0.0, -1e-3, np.inf, np.nan])
def test_scalarization_rejects_invalid_smoothing_parameter(
    parameter: float,
) -> None:
    model = ThreeStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(model.x_sym[0], ub=1.0, name="state_limit")

    with pytest.raises(ValueError, match="finite and positive"):
        scalarize_sms_ia_path_constraints(
            ocp.path_constraints,
            default_smoothing_parameter=parameter,
        )

    with pytest.raises(ValueError, match="finite and positive"):
        scalarize_sms_ia_path_constraints(
            ocp.path_constraints,
            smoothing_overrides={"state_limit[0].upper": parameter},
        )


def test_scalarization_ignores_overrides_outside_the_supplied_constraints() -> None:
    model = ThreeStateModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(model.x_sym[0], ub=1.0, name="first_limit")
    ocp.add_path_constraint(model.x_sym[1], ub=2.0, name="second_limit")

    inequalities = scalarize_sms_ia_path_constraints(
        ocp.path_constraints[:1],
        smoothing_overrides={
            "first_limit[0].upper": 2e-3,
            "second_limit[0].upper": 3e-3,
        },
    )

    assert len(inequalities) == 1
    assert inequalities[0].name == "first_limit[0].upper"
    assert inequalities[0].smoothing_parameter == 2e-3


def test_builds_smoothed_sms_growth_rates_in_physical_time() -> None:
    model = GrowthModel()
    ocp = OptimalControlProblem(model)
    ocp.add_path_constraint(
        model.t_sym + model.x_sym**2,
        ub=0.0,
        name="time_state",
    )
    ocp.add_path_constraint(
        model.u_sym + model.p_sym,
        ub=0.0,
        name="control_parameter",
    )
    inequalities = scalarize_sms_ia_path_constraints(
        ocp.path_constraints,
        default_smoothing_parameter=0.2,
        smoothing_overrides={"control_parameter[0].upper": 0.4},
    )

    growth_expr = build_sms_ia_growth_expr(model, inequalities)
    growth_function = model.create_function("sms_growth", growth_expr)
    value = growth_function(2.0, [2.0], [3.0], [1.0])

    time_state_derivative = 1.0 + 2.0 * 2.0 * (3.0 + 1.0)
    expected = np.array(
        [
            0.5
            * (
                np.sqrt(time_state_derivative**2 + 0.2**2)
                + time_state_derivative
            ),
            0.5 * 0.4,
        ]
    )

    assert growth_expr.shape == (2, 1)
    np.testing.assert_allclose(
        value.full().reshape(-1),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_builds_empty_sms_growth_column() -> None:
    growth_expr = build_sms_ia_growth_expr(GrowthModel(), ())

    assert isinstance(growth_expr, ca.SX)
    assert growth_expr.shape == (0, 1)


def test_builds_constraints_and_reuses_matching_integrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls: list[tuple[float, ...]] = []

    def fake_integrator_builder(
        model: BaseDynamicModel,
        quadrature_expr,
        output_points,
        *,
        name,
        options,
    ) -> ca.Function:
        output_points = tuple(output_points)
        build_calls.append(output_points)
        x0 = ca.MX.sym("x0", model.nx)
        start_time = ca.MX.sym("start_time")
        duration = ca.MX.sym("duration")
        states = ca.repmat(x0, 1, len(output_points))
        integrals = ca.MX.zeros(
            int(quadrature_expr.numel()),
            len(output_points),
        )
        return ca.Function(
            name,
            [x0, start_time, duration],
            [states, integrals],
            ["x0", "start_time", "duration"],
            ["state", "integrals"],
        )

    monkeypatch.setattr(
        "sms_ocp.sms_ia_transcription.build_interval_integrator",
        fake_integrator_builder,
    )

    model = ConstantStateModel()
    ocp = OptimalControlProblem(model)
    ocp.set_time_horizon(tf=1.0)
    ocp.add_path_constraint(model.x_sym, ub=1.0, name="sms_limit")
    ocp.add_path_constraint(
        model.x_sym,
        lb=-2.0,
        ub=2.0,
        name="grid_limit",
        enforcement="grid_only",
    )
    ocp.add_initial_constraint(model.x_sym, name="initial_state")
    ocp.add_terminal_constraint(model.x_sym, name="terminal_state")

    transcription = SMSIATranscription(
        ocp,
        shooting_grid=[0.0, 0.5, 1.0],
    )
    transcription.initialize()
    build_calls.clear()
    output_points = (0.5, 1.0)
    interval_plans = (
        ShootingIntervalPlan(
            output_points=output_points,
            checking_intervals=(
                CheckingIntervalPlan(0, 0, 1, 2),
            ),
        ),
        ShootingIntervalPlan(
            output_points=output_points,
            checking_intervals=(
                CheckingIntervalPlan(0, 1, 1, 2),
            ),
        ),
    )
    shooting_interval_blocks = tuple(
        transcription._build_shooting_interval_transcription_block(
            shooting_interval_index,
            plan,
        )
        for shooting_interval_index, plan in enumerate(interval_plans)
    )
    transcription.shooting_interval_blocks = shooting_interval_blocks
    transcription._assemble_constraint_vectors()

    constraint_function = ca.Function(
        "constraints",
        [transcription.decision_vector],
        [transcription.constraint_vector],
    )
    values = constraint_function([0.0, 0.5, 1.0])

    assert build_calls == [output_points]
    assert len(transcription.shooting_interval_blocks) == 2
    assert tuple(
        block.plan
        for block in transcription.shooting_interval_blocks
    ) == interval_plans
    assert all(
        isinstance(block.lagrange_cost_contribution, ca.MX)
        and block.lagrange_cost_contribution.shape == (1, 1)
        for block in transcription.shooting_interval_blocks
    )
    assert len(transcription.defect_constraint_blocks) == 2
    assert len(transcription.node_constraint_blocks) == 5
    assert len(transcription.sms_upper_bound_constraint_blocks) == 2
    assert transcription.sms_upper_bound_constraint_slice == slice(7, 9)
    assert transcription.sms_ia_constraint_rows == (
        SMSIAConstraintRowInfo(7, 0, 0, 0),
        SMSIAConstraintRowInfo(8, 0, 1, 1),
    )
    assert transcription.constraint_vector.shape == (9, 1)
    np.testing.assert_allclose(
        values.full().reshape(-1),
        [-0.5, -0.5, 0.0, 0.5, 1.0, 0.0, 1.0, -1.0, -0.5],
    )
    np.testing.assert_array_equal(
        transcription.constraint_lower_bounds,
        [0.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0, -np.inf, -np.inf],
    )
    np.testing.assert_array_equal(
        transcription.constraint_upper_bounds,
        [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    )
