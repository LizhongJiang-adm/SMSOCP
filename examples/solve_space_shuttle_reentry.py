"""Solve the Space Shuttle Reentry Trajectory problem.

Source:
    J. T. Betts, Practical Methods for Optimal Control and Estimation
    Using Nonlinear Programming, 2nd ed., SIAM, 2010, Section 6.1.

The state variables are scaled internally to improve numerical conditioning.
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


NUM_INTERVALS = 30
POSTPROCESS_SAMPLES_PER_INTERVAL = 8
INITIAL_TERMINAL_TIME = 2.0
HEATING_LIMIT = 70.0
STATE_SCALES = np.array([1e5, 1.0, 1.0, 1e4, 1.0, 1.0])
TIME_SCALE = 1e3
INITIAL_PHYSICAL_STATE = np.array(
    [
        260000.0,
        0.0,
        0.0,
        25600.0,
        np.deg2rad(-1.0),
        np.deg2rad(90.0),
    ]
)
OUTPUT_DIRECTORY = Path(__file__).with_name("outputs")
INTEGRATOR_OPTIONS = {
    "reltol": 1e-8,
    "abstol": 1e-8,
    "max_num_steps": 10000,
    "quad_err_con": True,
}
SOLVER_OPTIONS = {
    "ipopt.print_level": 5,
    "ipopt.max_iter": 2000,
    "ipopt.tol": 1e-7,
    "ipopt.constr_viol_tol": 1e-6,
    "ipopt.acceptable_tol": 1e-5,
    "ipopt.acceptable_constr_viol_tol": 1e-6,
    "ipopt.acceptable_iter": 10,
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.nlp_scaling_method": "gradient-based",
    "ipopt.mu_strategy": "adaptive",
    "ipopt.bound_relax_factor": 0.0,
    "show_eval_warnings": False,
}


def to_scaled_state(physical_state: np.ndarray) -> np.ndarray:
    """Convert ``(h, phi, theta, v, gamma, psi)`` to scaled coordinates."""
    return np.asarray(physical_state, dtype=float) / STATE_SCALES


def to_physical_state(scaled_state: np.ndarray) -> np.ndarray:
    """Convert one scaled state vector or state matrix to physical units."""
    values = np.asarray(scaled_state, dtype=float)
    scales = STATE_SCALES if values.ndim == 1 else STATE_SCALES[:, None]
    return values * scales


class SpaceShuttleReentryModel(BaseDynamicModel):
    """Six-state shuttle model with time measured in thousands of seconds."""

    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 6)
        self.u_sym = ca.SX.sym("u", 2)

        physical_state = ca.diag(ca.DM(STATE_SCALES)) @ self.x_sym
        altitude = physical_state[0]
        latitude = physical_state[2]
        velocity = physical_state[3]
        flight_path_angle = physical_state[4]
        heading_angle = physical_state[5]
        angle_of_attack, bank_angle = self.u_sym[0], self.u_sym[1]
        angle_of_attack_degrees = angle_of_attack * 180.0 / np.pi

        density = 0.002378 * ca.exp(-altitude / 23800.0)
        lift_coefficient = (
            -0.20704 + 0.029244 * angle_of_attack_degrees
        )
        drag_coefficient = (
            0.07854
            - 0.0061592 * angle_of_attack_degrees
            + 0.000621408 * angle_of_attack_degrees**2
        )
        dynamic_pressure_factor = (
            0.5 * 2690.0 * density * velocity**2
        )
        lift = lift_coefficient * dynamic_pressure_factor
        drag = drag_coefficient * dynamic_pressure_factor

        radius = 20902900.0 + altitude
        gravity = 0.14076539e17 / radius**2
        mass = 203000.0 / 32.174

        physical_derivative = ca.vertcat(
            velocity * ca.sin(flight_path_angle),
            velocity
            / radius
            * ca.cos(flight_path_angle)
            * ca.sin(heading_angle)
            / ca.cos(latitude),
            velocity
            / radius
            * ca.cos(flight_path_angle)
            * ca.cos(heading_angle),
            -drag / mass - gravity * ca.sin(flight_path_angle),
            lift
            / (mass * velocity)
            * ca.cos(bank_angle)
            + ca.cos(flight_path_angle)
            * (velocity / radius - gravity / velocity),
            lift
            * ca.sin(bank_angle)
            / (
                mass
                * velocity
                * ca.cos(flight_path_angle)
            )
            + velocity
            / radius
            * ca.cos(flight_path_angle)
            * ca.sin(heading_angle)
            * ca.tan(latitude),
        )
        self.ode_expr = (
            TIME_SCALE
            * physical_derivative
            / ca.DM(STATE_SCALES)
        )

        heat_factor = (
            1.0672181
            - 0.019213774 * angle_of_attack_degrees
            + 0.00021286289 * angle_of_attack_degrees**2
            - 0.0000010117249 * angle_of_attack_degrees**3
        )
        self.heating_rate_expr = (
            heat_factor
            * 17700.0
            * ca.sqrt(density)
            * (0.0001 * velocity) ** 3.07
        )


def build_ocp() -> OptimalControlProblem:
    """Build the scaled space-shuttle reentry optimal-control problem."""
    model = SpaceShuttleReentryModel()
    ocp = OptimalControlProblem(model)
    state = model.x_sym

    ocp.set_time_horizon(t0=0.0, tf=(1.0, 3.0))
    ocp.set_objective(mayer=-state[2])
    ocp.set_variable_bounds(
        "x",
        lb=[
            0.0,
            0.0,
            np.deg2rad(-89.0),
            1000.0 / STATE_SCALES[3],
            np.deg2rad(-89.0),
            0.0,
        ],
        ub=[
            300000.0 / STATE_SCALES[0],
            np.deg2rad(89.0),
            np.deg2rad(89.0),
            30000.0 / STATE_SCALES[3],
            np.deg2rad(89.0),
            np.deg2rad(90.0),
        ],
    )
    ocp.set_variable_bounds(
        "u",
        lb=np.deg2rad([-90.0, -89.0]),
        ub=np.deg2rad([90.0, 1.0]),
    )
    ocp.add_initial_constraint(
        state,
        to_scaled_state(INITIAL_PHYSICAL_STATE),
        name="initial_state",
    )
    ocp.add_terminal_constraint(
        ca.vertcat(state[0], state[3], state[4]),
        [
            80000.0 / STATE_SCALES[0],
            2500.0 / STATE_SCALES[3],
            np.deg2rad(-5.0),
        ],
        name="terminal_state",
    )
    ocp.add_path_constraint(
        model.heating_rate_expr / HEATING_LIMIT - 1.0,
        ub=0.0,
        name="normalized_heating",
        enforcement="sms_ia",
    )
    return ocp


def build_initial_guess(
    ocp: OptimalControlProblem,
    shooting_grid: np.ndarray,
) -> np.ndarray:
    """Build a dynamically consistent guess from a simple control seed."""
    model = ocp.model
    num_intervals = shooting_grid.size - 1
    states = np.empty((model.nx, num_intervals + 1))
    controls = np.empty((model.nu, num_intervals))
    states[:, 0] = to_scaled_state(INITIAL_PHYSICAL_STATE)

    propagator = build_interval_integrator(
        model,
        ca.SX.zeros(0, 1),
        [1.0],
        name="space_shuttle_reentry_initial_guess",
        options=INTEGRATOR_OPTIONS,
    )
    for interval_index in range(num_intervals):
        interval_start = shooting_grid[interval_index]
        interval_end = shooting_grid[interval_index + 1]
        midpoint = 0.5 * (interval_start + interval_end)
        controls[:, interval_index] = np.deg2rad(
            [17.4, -75.0 * (1.0 - midpoint)]
        )
        propagated = propagator(
            x0=states[:, interval_index],
            u=controls[:, interval_index],
            start_time=interval_start * INITIAL_TERMINAL_TIME,
            duration=(
                (interval_end - interval_start)
                * INITIAL_TERMINAL_TIME
            ),
        )
        states[:, interval_index + 1] = np.asarray(
            propagated["state"],
            dtype=float,
        ).reshape(-1)

    states = np.minimum(np.maximum(states, ocp.x_lb), ocp.x_ub)
    return pack_initial_guess(
        ocp,
        shooting_grid,
        states=states,
        controls=controls,
        terminal_time=INITIAL_TERMINAL_TIME,
    )


def postprocess_solution(result: SMSAlgorithmResult) -> None:
    """Save and plot the multiple-shooting state and control values."""
    if result.phase_two is None:
        raise RuntimeError(
            f"Phase II was not reached: {result.status}."
        )

    transcription = result.transcription
    solution = result.phase_two
    layout = transcription.decision_layout
    states = layout.extract(solution.decision_vector, "x")
    controls = layout.extract(solution.decision_vector, "u")
    terminal_time = float(
        layout.extract(solution.decision_vector, "T").item()
    )
    model = transcription.ocp.model
    assert isinstance(model, SpaceShuttleReentryModel)
    physical_states = to_physical_state(states)
    physical_time = (
        transcription.shooting_grid
        * terminal_time
        * TIME_SCALE
    )
    control_degrees = np.rad2deg(controls)
    local_points = np.linspace(
        0.0,
        1.0,
        POSTPROCESS_SAMPLES_PER_INTERVAL + 1,
    )
    propagator = build_interval_integrator(
        model,
        ca.SX.zeros(0, 1),
        local_points[1:],
        name="space_shuttle_reentry_postprocess",
        options=INTEGRATOR_OPTIONS,
    )
    dense_time_segments: list[np.ndarray] = []
    dense_state_segments: list[np.ndarray] = []
    dense_control_segments: list[np.ndarray] = []
    for interval_index in range(transcription.num_intervals):
        normalized_start = float(
            transcription.shooting_grid[interval_index]
        )
        normalized_duration = float(
            transcription.shooting_grid[interval_index + 1]
            - normalized_start
        )
        start_time = normalized_start * terminal_time
        duration = normalized_duration * terminal_time
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
        dense_time_segments.append(
            (start_time + duration * local_points[:point_count])
            * TIME_SCALE
        )
        dense_state_segments.append(
            interval_states[:, :point_count]
        )
        dense_control_segments.append(
            np.repeat(
                controls[:, interval_index, None],
                point_count,
                axis=1,
            )
        )

    dense_time = np.concatenate(dense_time_segments)
    dense_scaled_states = np.hstack(dense_state_segments)
    dense_physical_states = to_physical_state(dense_scaled_states)
    dense_controls = np.hstack(dense_control_segments)
    heating_function = ca.Function(
        "space_shuttle_reentry_heating_postprocess",
        [model.x_sym, model.u_sym],
        [model.heating_rate_expr],
    )
    dense_heating_rate = np.array(
        [
            float(
                heating_function(
                    dense_scaled_states[:, index],
                    dense_controls[:, index],
                )
            )
            for index in range(dense_time.size)
        ]
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    solution_path = OUTPUT_DIRECTORY / "space_shuttle_reentry_solution.npz"
    plot_path = OUTPUT_DIRECTORY / "space_shuttle_reentry_trajectory.png"
    np.savez(
        solution_path,
        decision_vector=solution.decision_vector,
        shooting_grid=transcription.shooting_grid,
        objective=solution.objective,
        terminal_time_scaled=terminal_time,
        terminal_time_seconds=terminal_time * TIME_SCALE,
        physical_state_nodes=physical_states,
        control_nodes=controls,
        control_nodes_degrees=control_degrees,
        dense_time_seconds=dense_time,
        dense_physical_states=dense_physical_states,
        dense_controls=dense_controls,
        dense_heating_rate=dense_heating_rate,
        heating_limit=HEATING_LIMIT,
    )

    state_series = (
        (physical_states[0], "Altitude [ft]"),
        (physical_states[3], "Velocity [ft/s]"),
        (np.rad2deg(physical_states[1]), "Longitude [deg]"),
        (np.rad2deg(physical_states[2]), "Latitude [deg]"),
        (
            np.rad2deg(physical_states[4]),
            "Flight-path angle [deg]",
        ),
        (np.rad2deg(physical_states[5]), "Heading angle [deg]"),
    )
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(11.0, 12.0),
        sharex=True,
    )
    for axis, (values, label) in zip(
        axes.flat[:6],
        state_series,
        strict=True,
    ):
        axis.plot(physical_time, values, marker="o", markersize=2.5)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)

    for axis, values, label in (
        (axes[3, 0], control_degrees[0], "Angle of attack [deg]"),
        (axes[3, 1], control_degrees[1], "Bank angle [deg]"),
    ):
        axis.step(
            physical_time,
            np.append(values, values[-1]),
            where="post",
        )
        axis.set(xlabel="Time [s]", ylabel=label)
        axis.grid(alpha=0.25)

    figure.suptitle("Space Shuttle Reentry Trajectory")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    stats = solution.solver_stats
    sampled_max_path_value = (
        None
        if result.kkt_check is None
        else result.kkt_check.max_path_value
    )
    print(
        "Space Shuttle Reentry Trajectory: "
        f"algorithm_success={result.is_success}, "
        f"algorithm_status={result.status}, "
        f"nlp_success={stats['success']}, "
        f"nlp_status={stats['return_status']}, "
        f"objective={solution.objective:.10g}, "
        "terminal_latitude="
        f"{np.rad2deg(physical_states[2, -1]):.6f} deg, "
        f"terminal_time={terminal_time * TIME_SCALE:.6f} s"
    )
    if sampled_max_path_value is not None:
        print(
            "  sampled_max_path_value="
            f"{sampled_max_path_value:+.6e}, "
            "sampled_max_heating_rate="
            f"{HEATING_LIMIT * (1.0 + sampled_max_path_value):.9f}"
        )
    print(f"  solution={solution_path.resolve()}")
    print(f"  trajectory_plot={plot_path.resolve()}")


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
        algorithm_options=SMSAlgorithmOptions(
            print_ipopt_output=False,
        ),
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
