"""Represent and solve explicit nonlinear programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.utils import CasadiExpr


@dataclass(frozen=True)
class ExplicitNlp:
    """Store the expressions and bounds of an explicit NLP."""

    x: CasadiExpr
    f: CasadiExpr
    g: CasadiExpr
    lbx: np.ndarray
    ubx: np.ndarray
    lbg: np.ndarray
    ubg: np.ndarray


@dataclass(frozen=True, slots=True)
class NlpSolveResult:
    """Store one numerical solution of an explicit NLP."""

    nlp: ExplicitNlp
    decision_vector: np.ndarray
    objective: float
    constraint_values: np.ndarray
    solver_stats: Mapping[str, object]
    solver_success: bool
    is_accepted: bool
    max_violation: float
    solver_build_seconds: float
    solve_seconds: float


def solve_nlp(
    nlp: ExplicitNlp,
    initial_guess: ArrayLike,
    *,
    name: str,
    solver_options: Mapping[str, object] | None = None,
) -> NlpSolveResult:
    """Build and solve one explicit NLP with IPOPT."""
    point = np.asarray(initial_guess, dtype=float).reshape(-1)
    if point.size != nlp.lbx.size:
        raise ValueError(
            f"initial_guess must contain {nlp.lbx.size} entries."
        )
    if not np.isfinite(point).all():
        raise ValueError("initial_guess must contain only finite values.")

    options: dict[str, object] = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": False,
    }
    options.update(solver_options or {})

    start = perf_counter()
    solver = ca.nlpsol(
        name,
        "ipopt",
        {"x": nlp.x, "f": nlp.f, "g": nlp.g},
        options,
    )
    solver_build_seconds = perf_counter() - start

    start = perf_counter()
    solution = solver(
        x0=point,
        lbx=nlp.lbx,
        ubx=nlp.ubx,
        lbg=nlp.lbg,
        ubg=nlp.ubg,
    )
    solve_seconds = perf_counter() - start

    decision_vector = np.asarray(solution["x"], dtype=float).reshape(-1)
    constraint_values = np.asarray(solution["g"], dtype=float).reshape(-1)
    objective = float(solution["f"])
    solver_stats = dict(solver.stats())
    solver_success = bool(solver_stats.get("success", False))
    finite_result = (
        np.isfinite(decision_vector).all()
        and np.isfinite(constraint_values).all()
        and np.isfinite(objective)
    )

    return NlpSolveResult(
        nlp=nlp,
        decision_vector=decision_vector,
        objective=objective,
        constraint_values=constraint_values,
        solver_stats=solver_stats,
        solver_success=solver_success,
        is_accepted=solver_success and finite_result,
        max_violation=_maximum_nlp_violation(
            decision_vector,
            constraint_values,
            nlp,
        ),
        solver_build_seconds=solver_build_seconds,
        solve_seconds=solve_seconds,
    )


def _maximum_nlp_violation(
    decision_vector: np.ndarray,
    constraint_values: np.ndarray,
    nlp: ExplicitNlp,
) -> float:
    """Return the maximum variable- or constraint-bound violation."""
    return float(
        max(
            np.max(nlp.lbg - constraint_values, initial=0.0),
            np.max(constraint_values - nlp.ubg, initial=0.0),
            np.max(nlp.lbx - decision_vector, initial=0.0),
            np.max(decision_vector - nlp.ubx, initial=0.0),
        )
    )
