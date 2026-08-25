import casadi as ca
import pytest

from sms_ocp.base_dynamic_model import BaseDynamicModel


class RowVectorDoubleIntegrator(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 1, 2)
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = ca.horzcat(self.x_sym[1], self.u_sym[0])


class BadOdeDimensionModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 2)
        self.ode_expr = ca.SX.sym("ode", 3)


class EmptyStateModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.ode_expr = ca.SX.zeros(0, 1)


class NumericStateModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.DM.zeros(1, 1)
        self.ode_expr = ca.DM.zeros(1, 1)


class FreeSymbolOdeModel(BaseDynamicModel):
    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x")
        self.ode_expr = ca.SX.sym("external")


def test_base_dynamic_model_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseDynamicModel()


def test_standardizes_and_evaluates_dynamics() -> None:
    model = RowVectorDoubleIntegrator()

    assert (model.nx, model.nu, model.np) == (2, 1, 0)
    assert model.x_sym.shape == (2, 1)
    assert model.u_sym.shape == (1, 1)
    assert model.p_sym.shape == (0, 1)
    assert model.ode_expr.shape == (2, 1)

    dynamics = model.create_function(
        "dynamics",
        model.ode_expr,
        output_name="dot_x",
    )
    value = dynamics(0.0, [2.0, 3.0], [4.0])

    assert value.shape == (2, 1)
    assert value.full().reshape(-1).tolist() == [3.0, 4.0]


def test_rejects_ode_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="ode_expr dimension 3 must match x_sym dimension 2"):
        BadOdeDimensionModel()


def test_requires_at_least_one_state() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        EmptyStateModel()


def test_state_must_be_an_sx_symbol() -> None:
    with pytest.raises(ValueError, match="x_sym must be a CasADi SX symbol"):
        NumericStateModel()


def test_rejects_free_symbols_in_ode() -> None:
    with pytest.raises(ValueError, match="depending only on"):
        FreeSymbolOdeModel()
