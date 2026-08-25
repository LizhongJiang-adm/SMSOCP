"""Swing a nonlinear cart-pole from the downward to upright position."""

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
    build_interval_integrator,
    pack_initial_guess,
    solve_sms_ocp,
)


NUM_INTERVALS = 50
SAMPLES_PER_INTERVAL = 160
FINAL_TIME = 4.0
POSITION_LOWER_BOUND = -0.65
POSITION_UPPER_BOUND = 0.65
OUTPUT_DIRECTORY = Path(__file__).with_name("outputs")
POSTPROCESS_INTEGRATOR_OPTIONS = {
    **INTEGRATOR_OPTIONS,
    "reltol": 1e-10,
    "abstol": 1e-10,
}
INITIAL_STATE = np.array([0.0, 0.0, np.pi, 0.0])
TERMINAL_STATE = np.zeros(4)


class CartPoleSwingUpModel(BaseDynamicModel):
    """Nonlinear cart-pole with angle zero at the upright equilibrium."""

    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 4)
        self.u_sym = ca.SX.sym("u", 1)

        position = self.x_sym[0]
        velocity = self.x_sym[1]
        angle = self.x_sym[2]
        angular_velocity = self.x_sym[3]
        force = self.u_sym[0]
        cart_mass = 1.0
        pole_mass = 0.3
        pole_length = 0.5
        gravity = 9.81

        sine = ca.sin(angle)
        cosine = ca.cos(angle)
        denominator = cart_mass + pole_mass * sine**2
        cart_acceleration = (
            force
            - pole_mass * gravity * sine * cosine
            + pole_mass * pole_length * angular_velocity**2 * sine
        ) / denominator
        angular_acceleration = (
            gravity * sine - cart_acceleration * cosine
        ) / pole_length

        self.ode_expr = ca.vertcat(
            velocity,
            cart_acceleration,
            angular_velocity,
            angular_acceleration,
        )


def build_ocp() -> OptimalControlProblem:
    """Create the constrained nonlinear swing-up problem."""
    model = CartPoleSwingUpModel()
    ocp = OptimalControlProblem(model)
    state, control = model.x_sym, model.u_sym

    ocp.set_time_horizon(tf=FINAL_TIME)
    ocp.set_objective(lagrange=control[0] ** 2)
    ocp.set_variable_bounds(
        "x",
        lb=[-1.5, -10.0, -2.0 * np.pi, -20.0],
        ub=[1.5, 10.0, 2.0 * np.pi, 20.0],
    )
    ocp.set_variable_bounds("u", lb=-20.0, ub=20.0)
    ocp.add_initial_constraint(
        state,
        INITIAL_STATE,
        name="initial_state",
    )
    ocp.add_terminal_constraint(
        state,
        TERMINAL_STATE,
        name="terminal_state",
    )
    ocp.add_path_constraint(
        POSITION_LOWER_BOUND - state[0],
        name="cart_left_limit",
        enforcement="sms_ia",
    )
    ocp.add_path_constraint(
        state[0] - POSITION_UPPER_BOUND,
        name="cart_right_limit",
        enforcement="sms_ia",
    )
    return ocp


def build_initial_guess(
    ocp: OptimalControlProblem,
    shooting_grid: np.ndarray,
) -> np.ndarray:
    """Build a smooth directional guess for the nonconvex swing-up."""
    normalized_time = shooting_grid
    smooth_progress = (
        3.0 * normalized_time**2 - 2.0 * normalized_time**3
    )
    angle = np.pi * (1.0 - smooth_progress)
    angular_velocity = (
        -6.0
        * np.pi
        * normalized_time
        * (1.0 - normalized_time)
        / FINAL_TIME
    )

    phase = np.pi * normalized_time
    position = (
        0.42
        * np.sin(phase) ** 2
        * np.sin(2.0 * phase)
    )
    velocity = (
        0.42
        * np.pi
        * (
            np.sin(2.0 * phase) ** 2
            + 2.0 * np.sin(phase) ** 2 * np.cos(2.0 * phase)
        )
        / FINAL_TIME
    )
    states = np.vstack(
        (position, velocity, angle, angular_velocity)
    )
    controls = np.zeros((ocp.model.nu, NUM_INTERVALS))
    return pack_initial_guess(
        ocp,
        shooting_grid,
        states=states,
        controls=controls,
    )


def postprocess_solution(result: SMSAlgorithmResult) -> None:
    """Reconstruct, check, save, plot, and animate the swing-up."""
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
    local_points = np.linspace(
        0.0,
        1.0,
        SAMPLES_PER_INTERVAL + 1,
    )
    propagator = build_interval_integrator(
        ocp.model,
        ca.SX.zeros(0, 1),
        local_points[1:],
        name="cart_pole_swing_up_postprocess",
        options=POSTPROCESS_INTEGRATOR_OPTIONS,
    )
    horizon = float(ocp.tf_bounds[0] - ocp.t0)
    time_segments: list[np.ndarray] = []
    state_segments: list[np.ndarray] = []
    control_segments: list[np.ndarray] = []

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
        time_segments.append(
            start_time + duration * local_points[:point_count]
        )
        state_segments.append(interval_states[:, :point_count])
        control_segments.append(
            np.repeat(
                controls[:, interval_index, None],
                point_count,
                axis=1,
            )
        )

    time = np.concatenate(time_segments)
    dense_states = np.hstack(state_segments)
    dense_controls = np.hstack(control_segments)
    position = dense_states[0]
    path_residuals = np.vstack(
        (
            POSITION_LOWER_BOUND - position,
            position - POSITION_UPPER_BOUND,
        )
    )
    maximum_flat_index = int(np.argmax(path_residuals))
    constraint_index, time_index = np.unravel_index(
        maximum_flat_index,
        path_residuals.shape,
    )
    constraint_names = ("cart_left_limit", "cart_right_limit")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    solution_path = OUTPUT_DIRECTORY / "cart_pole_swing_up_solution.npz"
    plot_path = (
        OUTPUT_DIRECTORY
        / "cart_pole_swing_up_path_constraints.png"
    )
    np.savez(
        solution_path,
        decision_vector=solution.decision_vector,
        shooting_grid=transcription.shooting_grid,
        objective=solution.objective,
        dense_time=time,
        dense_states=dense_states,
        dense_controls=dense_controls,
        dense_path_residuals=path_residuals,
        position_bounds=np.array(
            [POSITION_LOWER_BOUND, POSITION_UPPER_BOUND]
        ),
    )

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(
        time,
        path_residuals[0],
        linewidth=1.8,
        label="Left limit residual",
    )
    axis.plot(
        time,
        path_residuals[1],
        linewidth=1.8,
        label="Right limit residual",
    )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        title="Cart-pole swing-up position constraints",
        xlabel="Physical time [s]",
        ylabel="Path-constraint residual",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    stats = solution.solver_stats
    print(
        "Cart-pole swing-up: "
        f"success={stats['success']}, "
        f"status={stats['return_status']}, "
        f"objective={solution.objective:.10g}, "
        "dense_max_path="
        f"{path_residuals[constraint_index, time_index]:+.3e}, "
        f"maximum_time={time[time_index]:.9g}, "
        f"constraint={constraint_names[constraint_index]}"
    )
    print(
        "  position_range="
        f"[{np.min(position):+.6f}, {np.max(position):+.6f}]"
    )
    print(f"  solution={solution_path.resolve()}")
    print(f"  plot={plot_path.resolve()}")


def main() -> None:
    ocp = build_ocp()
    shooting_grid = np.linspace(
        0.0,
        1.0,
        NUM_INTERVALS + 1,
    )
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
