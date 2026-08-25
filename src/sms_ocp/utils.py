"""Small shared utilities for SMS-OCP."""

from __future__ import annotations

from typing import TypeAlias

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike


CasadiExpr: TypeAlias = ca.SX | ca.MX | ca.DM


def as_casadi_column_vector(expr: CasadiExpr | None, name: str) -> tuple[CasadiExpr, int]:
    """Return a CasADi expression as a column vector and its row count."""
    if expr is None:
        return ca.SX.zeros(0, 1), 0
    if not isinstance(expr, (ca.SX, ca.MX, ca.DM)):
        raise TypeError(f"{name} must be a CasADi SX, MX, or DM expression.")

    rows, cols = expr.shape
    if cols == 1:
        return expr, int(rows)
    if rows == 1:
        return expr.T, int(cols)
    raise ValueError(f"{name} must be scalar, a row vector, or a column vector.")


def normalize_grid_points(grid_points: ArrayLike) -> np.ndarray:
    """Return grid points as a strictly increasing one-dimensional array."""
    grid = np.asarray(grid_points, dtype=float).reshape(-1)
    if grid.size < 2:
        raise ValueError("grid_points must contain at least two points.")
    if not np.isfinite(grid).all():
        raise ValueError("grid_points must contain only finite values.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("grid_points must be strictly increasing.")
    return grid
