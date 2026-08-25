import casadi as ca
import numpy as np
import pytest

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.optimal_control_problem import OptimalControlProblem


class MinimalModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 2)
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = ca.vertcat(self.x_sym[1], self.u_sym[0])


class ParameterizedModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 2)
        self.u_sym = ca.SX.sym("u")
        self.p_sym = ca.SX.sym("p")
        self.ode_expr = ca.vertcat(self.x_sym[1], self.u_sym[0] + self.p_sym[0])


def test_builds_complete_ocp_contract() -> None:
    model = ParameterizedModel()
    ocp = OptimalControlProblem(model)

    ocp.set_time_horizon(t0=1.0, tf=(2.0, 4.0))
    ocp.set_objective(
        mayer=model.x_sym[0] ** 2,
        lagrange=model.u_sym[0] ** 2,
        min_time_weight=0.5,
    )
    ocp.set_variable_bounds("x", lb=[-5.0, -6.0], ub=[5.0, 6.0])
    ocp.set_variable_bounds("u", lb=-2.0, ub=2.0)
    ocp.set_variable_bounds("p", lb=0.0, ub=3.0)
    ocp.add_initial_constraint(model.x_sym, lb=[0.0, 1.0], name="initial_state")
    ocp.add_terminal_constraint(model.x_sym[0], lb=2.0, name="target_position")
    ocp.add_path_constraint(
        model.x_sym,
        lb=[-4.0, -3.0],
        ub=[4.0, 3.0],
        name="state_box",
        enforcement="grid_only",
    )

    assert ocp.model is model
    assert ocp.t0 == 1.0
    np.testing.assert_array_equal(ocp.tf_bounds, np.array([2.0, 4.0]))
    assert ocp.mayer_term_expr.shape == (1, 1)
    assert ocp.lagrange_term_expr.shape == (1, 1)
    assert ocp.min_time_weight == 0.5
    np.testing.assert_array_equal(ocp.x_lb, np.array([[-5.0], [-6.0]]))
    np.testing.assert_array_equal(ocp.x_ub, np.array([[5.0], [6.0]]))
    np.testing.assert_array_equal(ocp.u_lb, np.array([[-2.0]]))
    np.testing.assert_array_equal(ocp.u_ub, np.array([[2.0]]))
    np.testing.assert_array_equal(ocp.p_lb, np.array([[0.0]]))
    np.testing.assert_array_equal(ocp.p_ub, np.array([[3.0]]))

    initial = ocp.initial_constraints[0]
    assert (initial.name, initial.kind, initial.enforcement) == (
        "initial_state",
        "initial",
        None,
    )
    np.testing.assert_array_equal(initial.lb, np.array([[0.0], [1.0]]))
    np.testing.assert_array_equal(initial.ub, initial.lb)

    terminal = ocp.terminal_constraints[0]
    assert (terminal.name, terminal.kind, terminal.enforcement) == (
        "target_position",
        "terminal",
        None,
    )
    np.testing.assert_array_equal(terminal.lb, np.array([[2.0]]))
    np.testing.assert_array_equal(terminal.ub, terminal.lb)

    path = ocp.path_constraints[0]
    assert (path.name, path.kind, path.enforcement) == (
        "state_box",
        "path",
        "grid_only",
    )
    assert path.expr.shape == (2, 1)
    np.testing.assert_array_equal(path.lb, np.array([[-4.0], [-3.0]]))
    np.testing.assert_array_equal(path.ub, np.array([[4.0], [3.0]]))


@pytest.mark.parametrize("term", ["mayer", "lagrange"])
def test_objective_terms_must_be_scalar(term: str) -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)

    with pytest.raises(ValueError, match=rf"{term} must be scalar"):
        ocp.set_objective(**{term: model.x_sym})


def test_set_objective_only_updates_provided_terms() -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)
    ocp.set_objective(
        mayer=model.x_sym[0],
        lagrange=model.u_sym[0],
        min_time_weight=0.5,
    )
    original_lagrange = ocp.lagrange_term_expr

    ocp.set_objective(mayer=model.x_sym[1])

    assert str(ocp.mayer_term_expr) == "x_1"
    assert ocp.lagrange_term_expr is original_lagrange
    assert ocp.min_time_weight == 0.5


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_invalid_min_time_weight_does_not_partially_update_objective(
    weight: float,
) -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)
    ocp.set_objective(mayer=model.x_sym[0], min_time_weight=0.5)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        ocp.set_objective(lagrange=model.u_sym[0], min_time_weight=weight)

    assert ocp.lagrange_term_expr is None
    assert ocp.min_time_weight == 0.5


def test_invalid_time_horizon_does_not_partially_update_ocp() -> None:
    ocp = OptimalControlProblem(MinimalModel())
    ocp.set_time_horizon(t0=1.0, tf=3.0)

    with pytest.raises(ValueError, match="Fixed tf must be greater than t0"):
        ocp.set_time_horizon(t0=5.0, tf=4.0)

    assert ocp.t0 == 1.0
    np.testing.assert_array_equal(ocp.tf_bounds, np.array([3.0, 3.0]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"t0": np.nan}, "t0 must be finite"),
        ({"tf": np.nan}, "Fixed tf must be finite"),
        ({"tf": (1.0, np.nan)}, "tf_max must not be NaN"),
    ],
)
def test_time_horizon_rejects_nan(
    kwargs: dict[str, object],
    message: str,
) -> None:
    ocp = OptimalControlProblem(MinimalModel())

    with pytest.raises(ValueError, match=message):
        ocp.set_time_horizon(**kwargs)


def test_free_terminal_time_allows_infinite_upper_bound() -> None:
    ocp = OptimalControlProblem(MinimalModel())

    ocp.set_time_horizon(tf=(1.0, np.inf))

    np.testing.assert_array_equal(ocp.tf_bounds, np.array([1.0, np.inf]))


@pytest.mark.parametrize(
    ("method_name", "kind"),
    [
        ("add_path_constraint", "path"),
        ("add_initial_constraint", "initial"),
        ("add_terminal_constraint", "terminal"),
    ],
)
def test_constraint_names_must_be_unique_within_each_kind(
    method_name: str,
    kind: str,
) -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)
    add_constraint = getattr(ocp, method_name)

    add_constraint(model.x_sym[0], name="position")

    with pytest.raises(ValueError, match=rf"Duplicate {kind} constraint name"):
        add_constraint(model.x_sym[1], name="position")


def test_constraint_rejects_lower_bound_above_upper_bound() -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)

    with pytest.raises(ValueError, match="must satisfy lb <= ub"):
        ocp.add_path_constraint(model.x_sym, lb=[0.0, 2.0], ub=[1.0, 1.0])


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        (None, "must not be None"),
        (ca.SX.zeros(0, 1), "must not be empty"),
    ],
)
def test_constraint_rejects_missing_or_empty_expression(
    expr: object,
    message: str,
) -> None:
    ocp = OptimalControlProblem(MinimalModel())

    with pytest.raises(ValueError, match=message):
        ocp.add_path_constraint(expr)


def test_ocp_rejects_symbols_outside_model() -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)
    external = ca.SX.sym("external")

    with pytest.raises(ValueError, match="depend only on the model symbols"):
        ocp.set_objective(mayer=external)
    with pytest.raises(ValueError, match="depend only on the model symbols"):
        ocp.add_path_constraint(external)


def test_bounds_reject_nan() -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)

    with pytest.raises(ValueError, match="must not contain NaN"):
        ocp.set_variable_bounds("x", lb=[0.0, np.nan])
    with pytest.raises(ValueError, match="must not contain NaN"):
        ocp.add_path_constraint(model.x_sym, ub=[0.0, np.nan])


def test_bounds_allow_fully_unbounded_entries() -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)

    ocp.set_variable_bounds("x", lb=-np.inf, ub=np.inf)
    ocp.add_path_constraint(model.x_sym, lb=-np.inf, ub=np.inf)

    np.testing.assert_array_equal(ocp.x_lb, np.full((2, 1), -np.inf))
    np.testing.assert_array_equal(ocp.x_ub, np.full((2, 1), np.inf))
    np.testing.assert_array_equal(
        ocp.path_constraints[0].lb,
        np.full((2, 1), -np.inf),
    )
    np.testing.assert_array_equal(
        ocp.path_constraints[0].ub,
        np.full((2, 1), np.inf),
    )


def test_constraint_bounds_are_copied_and_read_only() -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)
    lower_bound = np.array([-1.0, -2.0])

    ocp.add_path_constraint(model.x_sym, lb=lower_bound, ub=2.0)
    constraint = ocp.path_constraints[0]
    lower_bound[0] = 99.0

    np.testing.assert_array_equal(
        constraint.lb,
        np.array([[-1.0], [-2.0]]),
    )
    assert not constraint.lb.flags.writeable
    assert not constraint.ub.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        constraint.lb[0, 0] = 99.0


def test_bounds_reject_infinities_in_the_wrong_direction() -> None:
    model = MinimalModel()
    ocp = OptimalControlProblem(model)

    with pytest.raises(ValueError, match=r"lower bound must not contain \+inf"):
        ocp.set_variable_bounds("x", lb=np.inf)
    with pytest.raises(ValueError, match="upper bound must not contain -inf"):
        ocp.add_path_constraint(model.x_sym, ub=-np.inf)


def test_variable_bounds_reject_conflict_without_partial_update() -> None:
    ocp = OptimalControlProblem(MinimalModel())
    ocp.set_variable_bounds("x", lb=0.0, ub=1.0)

    with pytest.raises(ValueError, match="must satisfy lb <= ub"):
        ocp.set_variable_bounds("x", lb=2.0, indices=0)

    np.testing.assert_array_equal(ocp.x_lb, np.array([[0.0], [0.0]]))
    np.testing.assert_array_equal(ocp.x_ub, np.array([[1.0], [1.0]]))


def test_variable_bounds_reject_invalid_kind() -> None:
    ocp = OptimalControlProblem(MinimalModel())

    with pytest.raises(ValueError, match="kind must be one of"):
        ocp.set_variable_bounds("state", lb=0.0)


def test_variable_bounds_reject_duplicate_indices_without_partial_update() -> None:
    ocp = OptimalControlProblem(MinimalModel())

    with pytest.raises(ValueError, match="indices must not contain duplicates"):
        ocp.set_variable_bounds("x", lb=[0.0, 1.0], indices=[0, 0])

    np.testing.assert_array_equal(ocp.x_lb, np.full((2, 1), -np.inf))
