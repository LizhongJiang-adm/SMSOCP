"""Base class for continuous-time dynamic models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import casadi as ca

from sms_ocp.utils import CasadiExpr, as_casadi_column_vector


class BaseDynamicModel(ABC):
    """Represent a continuous-time system ``xdot = f(t, x, u, p)``."""

    def __init__(self) -> None:
        self.t_sym: ca.SX = ca.SX.sym("t")
        self.x_sym: CasadiExpr | None = None
        self.u_sym: CasadiExpr | None = None
        self.p_sym: CasadiExpr | None = None
        self.ode_expr: CasadiExpr | None = None

        self.nx = 0
        self.nu = 0
        self.np = 0

        self.setup_model()
        self._validate_and_standardize()

    @abstractmethod
    def setup_model(self) -> None:
        """Define ``x_sym``, ``u_sym``, ``p_sym``, and ``ode_expr``."""

    def create_function(
        self,
        name: str,
        expr: CasadiExpr,
        output_name: str = "out",
    ) -> ca.Function:
        """Create a CasADi function with model inputs ``(t, x, [u], [p])``."""
        inputs = self.function_inputs(
            t=self.t_sym,
            x=self.x_sym,
            u=self.u_sym,
            p=self.p_sym,
        )
        return ca.Function(
            name,
            list(inputs.values()),
            [expr],
            list(inputs),
            [output_name],
        )

    def function_inputs(
        self,
        *,
        t: CasadiExpr | float,
        x: CasadiExpr,
        u: CasadiExpr | None = None,
        p: CasadiExpr | None = None,
    ) -> dict[str, CasadiExpr | float]:
        """Collect values matching the model signature ``(t, x, [u], [p])``."""
        inputs: dict[str, CasadiExpr | float] = {
            "t": t,
            "x": x,
        }
        if self.nu > 0:
            if u is None:
                raise ValueError(
                    "u must be provided because the model has control inputs."
                )
            inputs["u"] = u
        if self.np > 0:
            if p is None:
                raise ValueError(
                    "p must be provided because the model has parameters."
                )
            inputs["p"] = p
        return inputs

    def _validate_and_standardize(self) -> None:
        self.x_sym, self.nx = as_casadi_column_vector(self.x_sym, "x_sym")
        self.u_sym, self.nu = as_casadi_column_vector(self.u_sym, "u_sym")
        self.p_sym, self.np = as_casadi_column_vector(self.p_sym, "p_sym")

        if self.nx == 0:
            raise ValueError("x_sym must contain at least one state.")

        self._validate_sx_symbol(self.t_sym, "t_sym")
        if self.t_sym.shape != (1, 1):
            raise ValueError("t_sym must be scalar.")

        self._validate_sx_symbol(self.x_sym, "x_sym")
        if self.nu > 0:
            self._validate_sx_symbol(self.u_sym, "u_sym")
        if self.np > 0:
            self._validate_sx_symbol(self.p_sym, "p_sym")

        if self.ode_expr is None:
            raise ValueError("ode_expr must be defined by setup_model().")

        self.ode_expr, ode_n = as_casadi_column_vector(self.ode_expr, "ode_expr")
        if ode_n != self.nx:
            raise ValueError(
                f"ode_expr dimension {ode_n} must match x_sym dimension {self.nx}."
            )

        try:
            self.create_function(
                "validate_dynamics",
                self.ode_expr,
                output_name="dot_x",
            )
        except (RuntimeError, NotImplementedError) as exc:
            raise ValueError(
                "ode_expr must be a valid CasADi SX expression depending only on "
                "t_sym, x_sym, u_sym, and p_sym."
            ) from exc

    @staticmethod
    def _validate_sx_symbol(expr: CasadiExpr, name: str) -> None:
        if not isinstance(expr, ca.SX) or not expr.is_symbolic():
            raise ValueError(f"{name} must be a CasADi SX symbol.")
