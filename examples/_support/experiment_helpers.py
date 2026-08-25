"""Solver and integrator settings shared by package examples."""

from __future__ import annotations

INTEGRATOR_OPTIONS = {
    "reltol": 1e-8,
    "abstol": 1e-8,
    "max_num_steps": 10000,
}
SOLVER_OPTIONS = {
    "ipopt.max_iter": 2000,
    "ipopt.tol": 1e-8,
    "ipopt.constr_viol_tol": 1e-8,
    "ipopt.acceptable_tol": 1e-6,
    "ipopt.acceptable_constr_viol_tol": 1e-6,
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.bound_relax_factor": 0.0,
    "ipopt.mu_strategy": "adaptive",
    "show_eval_warnings": False,
}
