"""Container for continuous-time optimal-control problem contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.base_dynamic_model import BaseDynamicModel
from sms_ocp.utils import CasadiExpr, as_casadi_column_vector


VariableKind = Literal["x", "u", "p"]
ConstraintKind = Literal["path", "initial", "terminal"]
PathConstraintEnforcement = Literal["sms_ia", "grid_only"]

_PATH_ENFORCEMENTS = {"sms_ia", "grid_only"}


@dataclass(frozen=True)
class Constraint:
    """Store one vector constraint and its lower and upper bounds."""

    expr: CasadiExpr
    lb: np.ndarray
    ub: np.ndarray
    name: str
    kind: ConstraintKind
    enforcement: PathConstraintEnforcement | None = None


class OptimalControlProblem:
    """Collect the mathematical contract of an optimal-control problem."""

    def __init__(self, model: BaseDynamicModel) -> None:
        self.model = model

        self.t0 = 0.0
        self.tf_bounds = np.array([0.0, np.inf], dtype=float)

        self.mayer_term_expr: CasadiExpr | None = None
        self.lagrange_term_expr: CasadiExpr | None = None
        self.min_time_weight = 0.0

        self.path_constraints: list[Constraint] = []
        self.initial_constraints: list[Constraint] = []
        self.terminal_constraints: list[Constraint] = []

        self.x_lb = np.full((model.nx, 1), -np.inf)
        self.x_ub = np.full((model.nx, 1), np.inf)
        self.u_lb = np.full((model.nu, 1), -np.inf)
        self.u_ub = np.full((model.nu, 1), np.inf)
        self.p_lb = np.full((model.np, 1), -np.inf)
        self.p_ub = np.full((model.np, 1), np.inf)

    def set_time_horizon(
        self,
        *,
        t0: float = 0.0,
        tf: float | tuple[float, float] | list[float] | None = None,
    ) -> None:
        """Set the initial time and terminal-time bounds."""
        t0_value = float(t0)
        if not np.isfinite(t0_value):
            raise ValueError("t0 must be finite.")

        if tf is None:
            tf_bounds = np.array([t0_value, np.inf], dtype=float)
        elif isinstance(tf, (int, float)):
            tf_value = float(tf)
            if not np.isfinite(tf_value):
                raise ValueError("Fixed tf must be finite.")
            if tf_value <= t0_value:
                raise ValueError("Fixed tf must be greater than t0.")
            tf_bounds = np.array([tf_value, tf_value], dtype=float)
        else:
            if len(tf) != 2:
                raise ValueError(
                    "tf must be a scalar or a two-element bounds sequence."
                )

            tf_min, tf_max = float(tf[0]), float(tf[1])
            if not np.isfinite(tf_min) or np.isnan(tf_max):
                raise ValueError("tf_min must be finite and tf_max must not be NaN.")
            if tf_min < t0_value or tf_max < tf_min:
                raise ValueError("Time bounds must satisfy t0 <= tf_min <= tf_max.")
            tf_bounds = np.array([tf_min, tf_max], dtype=float)

        self.t0 = t0_value
        self.tf_bounds = tf_bounds

    def set_objective(
        self,
        *,
        mayer: CasadiExpr | None = None,
        lagrange: CasadiExpr | None = None,
        min_time_weight: float | None = None,
    ) -> None:
        """Update the provided Mayer, Lagrange, and minimum-time terms."""
        mayer_expr = None
        lagrange_expr = None
        weight = None

        if mayer is not None:
            mayer_expr = self._validate_scalar_expr(mayer, "mayer")
        if lagrange is not None:
            lagrange_expr = self._validate_scalar_expr(lagrange, "lagrange")
        if min_time_weight is not None:
            weight = float(min_time_weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(
                    "min_time_weight must be finite and nonnegative."
                )

        if mayer_expr is not None:
            self.mayer_term_expr = mayer_expr
        if lagrange_expr is not None:
            self.lagrange_term_expr = lagrange_expr
        if weight is not None:
            self.min_time_weight = weight

    def set_variable_bounds(
        self,
        kind: VariableKind,
        *,
        lb: ArrayLike | None = None,
        ub: ArrayLike | None = None,
        indices: int | Sequence[int] | None = None,
    ) -> None:
        """Set simple bounds for state, control, or parameter variables."""

        if kind not in ("x", "u", "p"):
            raise ValueError("kind must be one of: x, u, p.")

        lower = getattr(self, f"{kind}_lb")
        upper = getattr(self, f"{kind}_ub")
        dimension = lower.shape[0]

        if indices is None:
            idx = list(range(dimension))
        elif isinstance(indices, int):
            idx = [indices]
        else:
            idx = list(indices)

        if len(idx) != len(set(idx)):
            raise ValueError("indices must not contain duplicates.")
        if any(i < 0 or i >= dimension for i in idx):
            raise IndexError(f"indices must be within [0, {dimension}).")

        shape = (len(idx), 1)
        selected_lower = lower[idx, :].copy()
        selected_upper = upper[idx, :].copy()

        if lb is not None:
            selected_lower = self._normalize_bound(lb, shape, f"{kind} lower bound")
        if ub is not None:
            selected_upper = self._normalize_bound(ub, shape, f"{kind} upper bound")

        self._validate_bounds(
            selected_lower,
            selected_upper,
            f"{kind} variable bounds",
        )
        lower[idx, :] = selected_lower
        upper[idx, :] = selected_upper

    def add_path_constraint(
        self,
        expr: CasadiExpr,
        lb: ArrayLike = -np.inf,
        ub: ArrayLike = 0.0,
        *,
        name: str | None = None,
        enforcement: PathConstraintEnforcement = "sms_ia",
    ) -> None:
        """Add a path constraint and choose how it is enforced in SMS-OCP."""
        enforcement = self._validate_path_enforcement(enforcement)
        constraint_name = self._constraint_name(name, "path", len(self.path_constraints))
        self.path_constraints.append(
            self._make_constraint(
                expr,
                lb,
                ub,
                kind="path",
                name=constraint_name,
                enforcement=enforcement,
            )
        )

    def add_initial_constraint(
        self,
        expr: CasadiExpr,
        lb: ArrayLike = 0.0,
        ub: ArrayLike | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Add a constraint enforced at the initial time."""
        if ub is None:
            ub = lb
        constraint_name = self._constraint_name(
            name,
            "initial",
            len(self.initial_constraints),
        )
        self.initial_constraints.append(
            self._make_constraint(
                expr,
                lb,
                ub,
                kind="initial",
                name=constraint_name,
            )
        )

    def add_terminal_constraint(
        self,
        expr: CasadiExpr,
        lb: ArrayLike = 0.0,
        ub: ArrayLike | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Add a constraint enforced at the terminal time."""
        if ub is None:
            ub = lb
        constraint_name = self._constraint_name(
            name,
            "terminal",
            len(self.terminal_constraints),
        )
        self.terminal_constraints.append(
            self._make_constraint(
                expr,
                lb,
                ub,
                kind="terminal",
                name=constraint_name,
            )
        )

    def _validate_scalar_expr(
        self,
        expr: CasadiExpr,
        name: str,
    ) -> CasadiExpr:
        self._validate_model_expr(expr, name)
        if expr.shape != (1, 1):
            raise ValueError(f"{name} must be scalar, got shape {expr.shape}.")
        return expr

    def _make_constraint(
        self,
        expr: CasadiExpr,
        lb: ArrayLike,
        ub: ArrayLike,
        *,
        kind: ConstraintKind,
        name: str,
        enforcement: PathConstraintEnforcement | None = None,
    ) -> Constraint:
        constraints = getattr(self, f"{kind}_constraints")
        if any(constraint.name == name for constraint in constraints):
            raise ValueError(f"Duplicate {kind} constraint name: {name!r}.")

        if expr is None:
            raise ValueError(f"{kind} constraint expression must not be None.")
        expr_col, n_expr = as_casadi_column_vector(expr, f"{kind} constraint expression")
        if n_expr == 0:
            raise ValueError(f"{kind} constraint expression must not be empty.")
        self._validate_model_expr(expr_col, f"{kind} constraint {name!r}")

        shape = (n_expr, 1)
        normalized_lb = self._normalize_bound(lb, shape, f"{kind} lower bound")
        normalized_ub = self._normalize_bound(ub, shape, f"{kind} upper bound")
        self._validate_bounds(
            normalized_lb,
            normalized_ub,
            f"{kind} constraint {name!r}",
        )
        normalized_lb.setflags(write=False)
        normalized_ub.setflags(write=False)

        return Constraint(
            expr=expr_col,
            lb=normalized_lb,
            ub=normalized_ub,
            name=name,
            kind=kind,
            enforcement=enforcement,
        )

    @staticmethod
    def _normalize_bound(
        bound: ArrayLike,
        shape: tuple[int, int],
        name: str,
    ) -> np.ndarray:
        arr = np.asarray(bound, dtype=float)
        if arr.size == 1:
            return np.full(shape, arr.item())

        flat = arr.reshape(-1)
        expected = shape[0] * shape[1]
        if flat.size != expected:
            raise ValueError(f"{name} must be scalar or have {expected} entries.")
        return flat.reshape(shape).copy()

    def _validate_model_expr(
        self,
        expr: CasadiExpr,
        name: str,
    ) -> None:
        if not isinstance(expr, (ca.SX, ca.DM)):
            raise TypeError(f"{name} must be a CasADi SX or DM expression.")

        try:
            self.model.create_function("validate_ocp_expr", expr)
        except (RuntimeError, NotImplementedError) as exc:
            raise ValueError(
                f"{name} must depend only on the model symbols "
                "t_sym, x_sym, u_sym, and p_sym."
            ) from exc

    @staticmethod
    def _constraint_name(name: str | None, prefix: str, index: int) -> str:
        if name is None:
            return f"{prefix}_{index}"
        value = str(name).strip()
        if not value:
            raise ValueError("Constraint name must not be empty.")
        return value

    @staticmethod
    def _validate_path_enforcement(
        enforcement: str,
    ) -> PathConstraintEnforcement:
        if enforcement not in _PATH_ENFORCEMENTS:
            allowed = ", ".join(sorted(_PATH_ENFORCEMENTS))
            raise ValueError(f"path constraint enforcement must be one of: {allowed}.")
        return enforcement  # type: ignore[return-value]

    @staticmethod
    def _validate_bounds(
        lb: np.ndarray,
        ub: np.ndarray,
        name: str,
    ) -> None:
        if np.isnan(lb).any() or np.isnan(ub).any():
            raise ValueError(f"{name} must not contain NaN.")
        if np.isposinf(lb).any():
            raise ValueError(f"{name} lower bound must not contain +inf.")
        if np.isneginf(ub).any():
            raise ValueError(f"{name} upper bound must not contain -inf.")
        if np.any(lb > ub):
            raise ValueError(f"{name} must satisfy lb <= ub.")
