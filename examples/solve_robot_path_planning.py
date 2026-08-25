"""Solve the robot-path-planning problem with the complete SMS algorithm."""

from __future__ import annotations

from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _support.experiment_helpers import (
    INTEGRATOR_OPTIONS,
    SOLVER_OPTIONS,
)
from sms_ocp import (
    BaseDynamicModel,
    OptimalControlProblem,
    SMSAlgorithmOptions,
    SMSAlgorithmResult,
    SMSIAOptions,
    SMSKKTOptions,
    pack_initial_guess,
    solve_sms_ocp,
)


NUM_INTERVALS = 20
SAMPLES_PER_INTERVAL = 200
OUTPUT_DIRECTORY = Path(__file__).with_name("outputs")
FINAL_TIME = 10.0
INITIAL_POSITION = np.array([40.0, 5.0])
TERMINAL_POSITION = np.array([55.0, 70.0])
OBSTACLE_CENTERS = np.array(
    [
        [40.0, 20.0],
        [55.0, 40.0],
        [45.0, 65.0],
    ]
)
INNER_RADIUS = 10.0
OUTER_RADIUS = 80.0


class RobotPathPlanningModel(BaseDynamicModel):
    """Planar double-integrator robot model."""

    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 4)
        self.u_sym = ca.SX.sym("u", 2)
        self.ode_expr = ca.vertcat(
            self.x_sym[2],
            self.x_sym[3],
            self.u_sym[0],
            self.u_sym[1],
        )


def build_ocp() -> OptimalControlProblem:
    model = RobotPathPlanningModel()
    ocp = OptimalControlProblem(model)
    x, u = model.x_sym, model.u_sym
    ocp.set_time_horizon(tf=FINAL_TIME)
    ocp.set_objective(lagrange=ca.dot(u, u))
    ocp.set_variable_bounds(
        "x",
        lb=[0.0, 0.0, -10.0, -10.0],
        ub=[80.0, 80.0, 10.0, 10.0],
    )
    ocp.add_initial_constraint(x[:2], INITIAL_POSITION)
    ocp.add_terminal_constraint(x[:2], TERMINAL_POSITION)
    ocp.add_path_constraint(
        x,
        lb=[0.0, 0.0, -10.0, -10.0],
        ub=[80.0, 80.0, 10.0, 10.0],
        name="state_bounds",
        enforcement="sms_ia",
    )

    distance_squared = ca.vertcat(
        *(
            (x[0] - center[0]) ** 2
            + (x[1] - center[1]) ** 2
            for center in OBSTACLE_CENTERS
        )
    )
    ocp.add_path_constraint(
        distance_squared,
        lb=INNER_RADIUS**2,
        ub=OUTER_RADIUS**2,
        name="obstacle_distance_squared",
        enforcement="sms_ia",
    )
    return ocp


def build_initial_guess() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shooting_grid = np.linspace(0.0, 1.0, NUM_INTERVALS + 1)
    position = (
        INITIAL_POSITION[:, None]
        + (TERMINAL_POSITION - INITIAL_POSITION)[:, None]
        * shooting_grid[None, :]
    )
    velocity = (TERMINAL_POSITION - INITIAL_POSITION) / FINAL_TIME
    states = np.vstack(
        (
            position,
            np.repeat(
                velocity[:, None],
                NUM_INTERVALS + 1,
                axis=1,
            ),
        )
    )
    controls = np.zeros((2, NUM_INTERVALS))
    return shooting_grid, states, controls


def postprocess_solution(result: SMSAlgorithmResult) -> None:
    """Analytically reconstruct, check, save, and plot the robot path."""
    if result.phase_two is None:
        raise RuntimeError(
            f"Phase II was not reached: {result.status}."
        )

    transcription = result.transcription
    solution = result.phase_two
    layout = transcription.decision_layout
    states = layout.extract(solution.decision_vector, "x")
    controls = layout.extract(solution.decision_vector, "u")
    local_points = np.linspace(0.0, 1.0, SAMPLES_PER_INTERVAL + 1)
    time_segments: list[np.ndarray] = []
    state_segments: list[np.ndarray] = []

    for interval_index in range(transcription.num_intervals):
        normalized_start = float(
            transcription.shooting_grid[interval_index]
        )
        normalized_duration = float(
            transcription.shooting_grid[interval_index + 1]
            - normalized_start
        )
        start_time = normalized_start * FINAL_TIME
        interval_duration = normalized_duration * FINAL_TIME
        local_time = local_points * interval_duration
        position = (
            states[:2, interval_index, None]
            + states[2:, interval_index, None] * local_time
            + 0.5
            * controls[:, interval_index, None]
            * local_time**2
        )
        velocity = (
            states[2:, interval_index, None]
            + controls[:, interval_index, None] * local_time
        )
        point_count = (
            local_points.size
            if interval_index == transcription.num_intervals - 1
            else local_points.size - 1
        )
        time_segments.append(
            start_time + local_time[:point_count]
        )
        state_segments.append(
            np.vstack((position, velocity))[:, :point_count]
        )

    time = np.concatenate(time_segments)
    dense_states = np.hstack(state_segments)
    dense_position = dense_states[:2]
    dense_velocity = dense_states[2:]
    distance_squared = np.sum(
        (
            dense_position[None, :, :]
            - OBSTACLE_CENTERS[:, :, None]
        )
        ** 2,
        axis=1,
    )
    distances = np.sqrt(distance_squared)
    state_bound_residuals = np.vstack(
        (
            -dense_position,
            dense_position - 80.0,
            -10.0 - dense_velocity,
            dense_velocity - 10.0,
        )
    )
    obstacle_lower_residuals = INNER_RADIUS**2 - distance_squared
    obstacle_upper_residuals = distance_squared - OUTER_RADIUS**2
    path_residuals = np.vstack(
        (
            state_bound_residuals,
            obstacle_lower_residuals,
            obstacle_upper_residuals,
        )
    )
    residual_labels = (
        "x lower",
        "y lower",
        "x upper",
        "y upper",
        "vx lower",
        "vy lower",
        "vx upper",
        "vy upper",
        "obstacle 1 lower",
        "obstacle 2 lower",
        "obstacle 3 lower",
        "obstacle 1 upper",
        "obstacle 2 upper",
        "obstacle 3 upper",
    )
    maximum_flat_index = int(np.argmax(path_residuals))
    residual_index, time_index = np.unravel_index(
        maximum_flat_index,
        path_residuals.shape,
    )
    minimum_clearance = float(distances.min() - INNER_RADIUS)
    outer_margin = float(OUTER_RADIUS - distances.max())

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    solution_path = OUTPUT_DIRECTORY / "robot_path_planning_solution.npz"
    path_plot_path = (
        OUTPUT_DIRECTORY / "robot_path_planning_path_constraints.png"
    )
    trajectory_plot_path = (
        OUTPUT_DIRECTORY / "robot_path_planning_trajectory.png"
    )
    np.savez(
        solution_path,
        decision_vector=solution.decision_vector,
        shooting_grid=transcription.shooting_grid,
        objective=solution.objective,
        dense_time=time,
        dense_states=dense_states,
        dense_path_residuals=path_residuals,
        obstacle_centers=OBSTACLE_CENTERS,
        obstacle_radius=INNER_RADIUS,
        initial_position=INITIAL_POSITION,
        terminal_position=TERMINAL_POSITION,
    )

    path_figure, path_axis = plt.subplots(figsize=(8.5, 4.8))
    path_axis.plot(
        time,
        np.max(state_bound_residuals, axis=0),
        label="state bounds",
    )
    for obstacle_index, residual in enumerate(
        obstacle_lower_residuals,
        start=1,
    ):
        path_axis.plot(
            time,
            residual,
            label=f"obstacle {obstacle_index} lower",
        )
    path_axis.plot(
        time,
        np.max(obstacle_upper_residuals, axis=0),
        label="obstacle upper bounds",
    )
    path_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    path_axis.set(
        title="Robot path-planning constraints: analytic trajectory",
        xlabel="Physical time",
        ylabel="Constraint residual",
    )
    path_axis.grid(alpha=0.25)
    path_axis.legend(fontsize=7)
    path_figure.tight_layout()
    path_figure.savefig(path_plot_path, dpi=160)
    plt.close(path_figure)

    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    for obstacle_index, center in enumerate(OBSTACLE_CENTERS):
        axis.add_patch(
            plt.Circle(
                center,
                INNER_RADIUS,
                color="tab:red",
                alpha=0.22,
                label=(
                    "Forbidden disk"
                    if obstacle_index == 0
                    else None
                ),
            )
        )
    axis.plot(
        dense_position[0],
        dense_position[1],
        color="tab:blue",
        linewidth=2.0,
        label="Dense trajectory",
    )
    axis.plot(
        states[0],
        states[1],
        "o",
        color="tab:blue",
        markersize=3,
        label="Shooting nodes",
    )
    axis.plot(*INITIAL_POSITION, "s", color="tab:green", label="Start")
    axis.plot(*TERMINAL_POSITION, "*", color="black", markersize=10, label="Goal")
    axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        xlim=(0.0, 80.0),
        ylim=(0.0, 80.0),
        aspect="equal",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(trajectory_plot_path, dpi=160)
    plt.close(figure)

    stats = solution.solver_stats
    print(
        "Robot path planning: "
        f"success={stats['success']}, "
        f"status={stats['return_status']}, "
        f"objective={solution.objective:.10g}, "
        "analytic_max_path="
        f"{path_residuals[residual_index, time_index]:+.3e}, "
        f"maximum_time={time[time_index]:.9g}, "
        f"constraint={residual_labels[residual_index]}, "
        f"minimum_clearance={minimum_clearance:+.6e}, "
        f"outer_radius_margin={outer_margin:+.6e}"
    )
    print(f"  solution={solution_path.resolve()}")
    print(f"  path_plot={path_plot_path.resolve()}")
    print(f"  trajectory_plot={trajectory_plot_path.resolve()}")


def main() -> None:
    ocp = build_ocp()
    shooting_grid, states, controls = build_initial_guess()
    result = solve_sms_ocp(
        ocp,
        shooting_grid,
        pack_initial_guess(
            ocp,
            shooting_grid,
            states=states,
            controls=controls,
        ),
        sms_ia_options=SMSIAOptions(
            default_smoothing_parameter=1e-3,
        ),
        algorithm_options=SMSAlgorithmOptions(),
        kkt_options=SMSKKTOptions(
            samples_per_atomic_interval=41,
            active_tolerance=1e-3,
            stationarity_tolerance=1e-3,
            active_point_sample_stride=1,
        ),
        integrator_options=INTEGRATOR_OPTIONS,
        solver_options=SOLVER_OPTIONS,
    )
    postprocess_solution(result)


if __name__ == "__main__":
    main()
