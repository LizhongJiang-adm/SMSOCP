"""Solve the Jacobson–Lele State-Constraint Example.

Source:
    D. H. Jacobson and M. M. Lele, "A transformation technique for
    optimal control problems with a state variable inequality constraint,"
    IEEE Transactions on Automatic Control, 1969, Example 1.

This implementation adds finite control bounds for numerical robustness.
"""

from __future__ import annotations

from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sms_ocp import (
    BaseDynamicModel,
    OptimalControlProblem,
    SMSAlgorithmOptions,
    SMSAlgorithmResult,
    SMSIAOptions,
    SMSKKTOptions,
    build_interval_integrator,
    pack_initial_guess,
    solve_sms_ocp,
)


DISPLAY_NAME = "Jacobson–Lele State-Constraint Example"
NUM_INTERVALS = 20
INITIAL_STATE = np.array([0.0, -1.0])
INITIAL_CONTROL = 0.0
SAMPLES_PER_INTERVAL = 200
OUTPUT_DIRECTORY = Path(__file__).with_name("outputs")
INTEGRATOR_OPTIONS = {
    "reltol": 1e-8,
    "abstol": 1e-8,
    "max_num_steps": 10000,
}
POSTPROCESS_INTEGRATOR_OPTIONS = {
    **INTEGRATOR_OPTIONS,
    "reltol": 1e-10,
    "abstol": 1e-10,
}
SOLVER_OPTIONS = {
    "ipopt.max_iter": 1500,
    "ipopt.tol": 1e-8,
    "ipopt.constr_viol_tol": 1e-8,
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.limited_memory_max_history": 50,
    "ipopt.bound_relax_factor": 0.0,
}


class JacobsonLeleStateConstraintModel(BaseDynamicModel):
    """Two-state model from Jacobson and Lele's first example."""

    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 2)
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = ca.vertcat(
            self.x_sym[1],
            self.u_sym[0] - self.x_sym[1],
        )


def build_ocp() -> OptimalControlProblem:
    model = JacobsonLeleStateConstraintModel()
    ocp = OptimalControlProblem(model)
    x, u, t = model.x_sym, model.u_sym[0], model.t_sym
    ocp.set_time_horizon(tf=1.0)
    ocp.set_objective(
        lagrange=u**2 / 200.0 + x[0] ** 2 + x[1] ** 2
    )
    ocp.set_variable_bounds("u", lb=-15.0, ub=15.0)
    ocp.add_initial_constraint(x, INITIAL_STATE)
    ocp.add_path_constraint(
        x[1] - 8.0 * (t - 0.5) ** 2 + 0.5,
        name="jacobson_lele_time_varying_limit",
    )
    return ocp


def build_initial_guess(
    ocp: OptimalControlProblem,
    shooting_grid: np.ndarray,
) -> np.ndarray:
    controls = np.full(
        (ocp.model.nu, shooting_grid.size - 1),
        INITIAL_CONTROL,
    )
    states = np.zeros((ocp.model.nx, shooting_grid.size))
    states[:, 0] = INITIAL_STATE
    step = build_interval_integrator(
        ocp.model,
        ca.SX.zeros(0, 1),
        [1.0],
        name="jacobson_lele_initial_guess",
        options=INTEGRATOR_OPTIONS,
    )
    horizon = float(ocp.tf_bounds[0] - ocp.t0)
    for interval_index in range(shooting_grid.size - 1):
        normalized_start = float(shooting_grid[interval_index])
        normalized_duration = float(
            shooting_grid[interval_index + 1] - normalized_start
        )
        propagated = step(
            x0=states[:, interval_index],
            u=controls[:, interval_index],
            start_time=ocp.t0 + normalized_start * horizon,
            duration=normalized_duration * horizon,
        )
        states[:, interval_index + 1] = np.asarray(
            propagated["state"][:, -1],
            dtype=float,
        ).reshape(-1)
    return pack_initial_guess(
        ocp,
        shooting_grid,
        states=states,
        controls=controls,
    )


def postprocess_solution(result: SMSAlgorithmResult) -> None:
    if result.phase_two is None:
        raise RuntimeError(
            f"Phase II was not reached: {result.status}."
        )

    transcription = result.transcription
    solution = result.phase_two
    ocp = transcription.ocp
    layout = transcription.decision_layout
    states = layout.extract(solution.decision_vector, "x")
    controls = layout.extract(solution.decision_vector, "u")
    local_points = np.linspace(0.0, 1.0, SAMPLES_PER_INTERVAL + 1)
    propagator = build_interval_integrator(
        ocp.model,
        ca.SX.zeros(0, 1),
        local_points[1:],
        name="jacobson_lele_postprocess",
        options=POSTPROCESS_INTEGRATOR_OPTIONS,
    )
    horizon = float(ocp.tf_bounds[0] - ocp.t0)
    time_segments: list[np.ndarray] = []
    residual_segments: list[np.ndarray] = []

    for interval_index in range(transcription.num_intervals):
        normalized_start = float(
            transcription.shooting_grid[interval_index]
        )
        normalized_duration = float(
            transcription.shooting_grid[interval_index + 1]
            - normalized_start
        )
        start_time = ocp.t0 + normalized_start * horizon
        duration = normalized_duration * horizon
        propagated = propagator(
            x0=states[:, interval_index],
            u=controls[:, interval_index],
            start_time=start_time,
            duration=duration,
        )
        interval_states = np.column_stack(
            (
                states[:, interval_index],
                np.asarray(propagated["state"], dtype=float),
            )
        )
        point_count = (
            local_points.size
            if interval_index == transcription.num_intervals - 1
            else local_points.size - 1
        )
        interval_time = (
            start_time + duration * local_points[:point_count]
        )
        time_segments.append(interval_time)
        residual_segments.append(
            interval_states[1, :point_count]
            - 8.0 * (interval_time - 0.5) ** 2
            + 0.5
        )

    time = np.concatenate(time_segments)
    path_residual = np.concatenate(residual_segments)
    maximum_index = int(np.argmax(path_residual))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    solution_path = (
        OUTPUT_DIRECTORY / "jacobson_lele_state_constraint_solution.npz"
    )
    plot_path = (
        OUTPUT_DIRECTORY
        / "jacobson_lele_state_constraint_path_constraint.png"
    )
    np.savez(
        solution_path,
        decision_vector=solution.decision_vector,
        shooting_grid=transcription.shooting_grid,
        objective=solution.objective,
        dense_time=time,
        dense_path_residual=path_residual,
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(time, path_residual, label=r"$g(t)$")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        title="Jacobson–Lele state constraint",
        xlabel="Physical time",
        ylabel="Constraint residual",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    stats = solution.solver_stats
    print(
        f"{DISPLAY_NAME}: success={stats['success']}, "
        f"status={stats['return_status']}, "
        f"objective={solution.objective:.10g}, "
        f"dense_max_path={path_residual[maximum_index]:+.3e}, "
        f"maximum_time={time[maximum_index]:.9g}"
    )
    print(f"  solution={solution_path.resolve()}")
    print(f"  plot={plot_path.resolve()}")


def main() -> None:
    ocp = build_ocp()
    shooting_grid = np.linspace(0.0, 1.0, NUM_INTERVALS + 1)
    result = solve_sms_ocp(
        ocp,
        shooting_grid,
        build_initial_guess(ocp, shooting_grid),
        sms_ia_options=SMSIAOptions(
            default_smoothing_parameter=1e-3,
        ),
        algorithm_options=SMSAlgorithmOptions(),
        kkt_options=SMSKKTOptions(
            samples_per_atomic_interval=41,
            active_tolerance=1e-3,
            stationarity_tolerance=1e-3,
        ),
        integrator_options=INTEGRATOR_OPTIONS,
        solver_options=SOLVER_OPTIONS,
    )
    postprocess_solution(result)


if __name__ == "__main__":
    main()
