"""Solve the archived high-frequency narrow-corridor OCP with SMS-OCP."""

from __future__ import annotations

import json
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
    pack_initial_guess,
    solve_sms_ocp,
)
from sms_ocp.sms_ia_transcription import SMSIATranscription


NUM_INTERVALS = 10
SAMPLES_PER_INTERVAL = 200
BASE_WIDTH = 0.08
CORRIDOR_CENTERS = np.array([0.12, 0.50, 0.84])
CORRIDOR_SIGMAS = np.array([0.08, 0.06, 0.09])
CORRIDOR_MIN_WIDTHS = np.array([0.030, 0.035, 0.010])
CENTERLINE_CENTERS = np.array([0.10, 0.32, 0.53, 0.73, 0.88])
CENTERLINE_AMPLITUDES = np.array([0.045, 0.026, 0.044, 0.030, 0.022])
CENTERLINE_CYCLES = np.array([7.0, 10.5, 13.0, 8.0, 11.5])
CENTERLINE_SIGMAS = np.array([0.16, 0.13, 0.18, 0.11, 0.10])
CENTERLINE_PHASES = np.array([0.4, 1.7, 2.9, 5.1, 3.6])
INITIAL_STATE = np.array([0.0, 0.0, 1.0, 0.0])
TERMINAL_STATE = np.array([1.0, 0.0, 1.0, 0.0])
OUTPUT_DIRECTORY = Path(__file__).with_name("outputs")
INTEGRATOR_OPTIONS = {
    "reltol": 1e-9,
    "abstol": 1e-9,
    "max_num_steps": 10000,
}
SOLVER_OPTIONS = {
    "ipopt.max_iter": 500,
    "ipopt.tol": 1e-8,
    "ipopt.constr_viol_tol": 1e-8,
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.limited_memory_max_history": 50,
    "ipopt.bound_relax_factor": 0.0,
}


class NarrowCorridorModel(BaseDynamicModel):
    """Planar double integrator with acceleration controls."""

    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 4)
        self.u_sym = ca.SX.sym("u", 2)
        self.ode_expr = ca.vertcat(
            self.x_sym[2],
            self.x_sym[3],
            self.u_sym[0],
            self.u_sym[1],
        )


def corridor_width_expr(time: ca.SX) -> ca.SX:
    """Return the archived irregular corridor half-width."""
    width = ca.SX(BASE_WIDTH)
    for center, sigma, minimum_width in zip(
        CORRIDOR_CENTERS,
        CORRIDOR_SIGMAS,
        CORRIDOR_MIN_WIDTHS,
        strict=True,
    ):
        scaled_time = (time - center) / sigma
        width -= (
            (BASE_WIDTH - minimum_width)
            * ca.exp(-(scaled_time**2))
        )
    return width


def corridor_centerline_expr(time: ca.SX) -> ca.SX:
    """Return the archived high-frequency corridor centerline."""
    centerline = ca.SX(0.0)
    for center, amplitude, cycles, sigma, phase in zip(
        CENTERLINE_CENTERS,
        CENTERLINE_AMPLITUDES,
        CENTERLINE_CYCLES,
        CENTERLINE_SIGMAS,
        CENTERLINE_PHASES,
        strict=True,
    ):
        scaled_time = (time - center) / sigma
        angle = 2.0 * np.pi * cycles * (time - center) + phase
        centerline += (
            amplitude
            * ca.exp(-(scaled_time**2))
            * ca.sin(angle)
        )
    return centerline


def corridor_width_values(time: np.ndarray) -> np.ndarray:
    """Evaluate the corridor half-width numerically."""
    width = np.full_like(time, BASE_WIDTH, dtype=float)
    for center, sigma, minimum_width in zip(
        CORRIDOR_CENTERS,
        CORRIDOR_SIGMAS,
        CORRIDOR_MIN_WIDTHS,
        strict=True,
    ):
        width -= (
            (BASE_WIDTH - minimum_width)
            * np.exp(-((time - center) / sigma) ** 2)
        )
    return width


def corridor_centerline_values(time: np.ndarray) -> np.ndarray:
    """Evaluate the corridor centerline numerically."""
    centerline = np.zeros_like(time, dtype=float)
    for center, amplitude, cycles, sigma, phase in zip(
        CENTERLINE_CENTERS,
        CENTERLINE_AMPLITUDES,
        CENTERLINE_CYCLES,
        CENTERLINE_SIGMAS,
        CENTERLINE_PHASES,
        strict=True,
    ):
        centerline += (
            amplitude
            * np.exp(-((time - center) / sigma) ** 2)
            * np.sin(2.0 * np.pi * cycles * (time - center) + phase)
        )
    return centerline


def build_ocp(enforcement: str = "sms_ia") -> OptimalControlProblem:
    """Build the frozen narrow-corridor optimal-control problem."""
    model = NarrowCorridorModel()
    ocp = OptimalControlProblem(model)
    state, control, time = model.x_sym, model.u_sym, model.t_sym
    width = corridor_width_expr(time)
    centerline = corridor_centerline_expr(time)

    ocp.set_time_horizon(t0=0.0, tf=1.0)
    ocp.set_objective(
        lagrange=(
            0.05 * ca.dot(control, control)
            + 0.01 * state[1] ** 2
            + 0.005 * state[3] ** 2
        )
    )
    ocp.set_variable_bounds(
        "u",
        lb=[-8.0, -8.0],
        ub=[8.0, 8.0],
    )
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
        state[1] - centerline - width,
        name="corridor_upper",
        enforcement=enforcement,
    )
    ocp.add_path_constraint(
        centerline - state[1] - width,
        name="corridor_lower",
        enforcement=enforcement,
    )
    return ocp


def build_initial_guess(
    ocp: OptimalControlProblem,
    shooting_grid: np.ndarray,
) -> np.ndarray:
    """Return the dynamically consistent straight-line initial guess."""
    states = np.zeros((4, shooting_grid.size))
    states[0] = shooting_grid
    states[2] = 1.0
    controls = np.zeros((2, shooting_grid.size - 1))
    return pack_initial_guess(
        ocp,
        shooting_grid,
        states=states,
        controls=controls,
    )


def reconstruct_trajectory(
    transcription: SMSIATranscription,
    decision_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly reconstruct each constant-control double-integrator segment."""
    layout = transcription.decision_layout
    states = layout.extract(decision_vector, "x")
    controls = layout.extract(decision_vector, "u")
    time_segments: list[np.ndarray] = []
    state_segments: list[np.ndarray] = []

    for interval_index in range(transcription.num_intervals):
        start = float(transcription.shooting_grid[interval_index])
        end = float(transcription.shooting_grid[interval_index + 1])
        local_time = np.linspace(
            0.0,
            end - start,
            SAMPLES_PER_INTERVAL + 1,
        )
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
            local_time.size
            if interval_index == transcription.num_intervals - 1
            else local_time.size - 1
        )
        time_segments.append(start + local_time[:point_count])
        state_segments.append(
            np.vstack((position, velocity))[:, :point_count]
        )

    return np.concatenate(time_segments), np.hstack(state_segments)


def save_smsia_results(result: SMSAlgorithmResult) -> None:
    """Save a compact numerical record and the corridor diagnostic plot."""
    if result.phase_two is None or result.kkt_check is None:
        raise RuntimeError(
            f"Complete Phase-II/KKT result was not obtained: {result.status}."
        )

    time, states = reconstruct_trajectory(
        result.transcription,
        result.phase_two.decision_vector,
    )
    centerline = corridor_centerline_values(time)
    width = corridor_width_values(time)
    upper_residual = states[1] - centerline - width
    lower_residual = centerline - states[1] - width
    dense_max_path_value = float(
        np.max(np.maximum(upper_residual, lower_residual))
    )
    phase_one_solves = result.phase_one_refinement_count + 1
    phase_two_solves = result.phase_two_refinement_count + 1

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    solution_path = OUTPUT_DIRECTORY / "narrow_corridor_solution.npz"
    summary_path = OUTPUT_DIRECTORY / "narrow_corridor_summary.json"
    plot_path = OUTPUT_DIRECTORY / "narrow_corridor_trajectory.png"
    np.savez(
        solution_path,
        decision_vector=result.phase_two.decision_vector,
        shooting_grid=result.transcription.shooting_grid,
        dense_time=time,
        dense_states=states,
        dense_upper_residual=upper_residual,
        dense_lower_residual=lower_residual,
    )
    summary = {
        "algorithm_status": result.status,
        "algorithm_success": result.is_success,
        "phase_one_nlp_solves": phase_one_solves,
        "phase_one_refinements": result.phase_one_refinement_count,
        "phase_one_rho": result.phase_one_rho,
        "phase_two_nlp_solves": phase_two_solves,
        "phase_two_refinements": result.phase_two_refinement_count,
        "final_checking_intervals": len(
            result.transcription.sms_ia_constraint_rows
        ),
        "objective": result.phase_two.objective,
        "kkt_stationarity_residual": (
            result.kkt_check.stationarity_residual
        ),
        "kkt_sampled_max_path_value": result.kkt_check.max_path_value,
        "dense_max_path_value": dense_max_path_value,
        "total_seconds": result.timing["total_seconds"],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].fill_between(
        time,
        centerline - width,
        centerline + width,
        color="tab:blue",
        alpha=0.18,
        label="Admissible corridor",
    )
    axes[0].plot(time, centerline, "--", color="tab:blue", label="Centerline")
    axes[0].plot(time, states[1], color="tab:orange", label="Trajectory")
    axes[0].set_ylabel("Vertical position")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(time, upper_residual, label="Upper residual")
    axes[1].plot(time, lower_residual, label="Lower residual")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set(
        xlabel="Time",
        ylabel="Path-constraint residual",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("High-frequency narrow-corridor SMS solution")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    print(
        "Narrow corridor: "
        f"status={result.status}, "
        f"phase_one_solves={phase_one_solves}, "
        f"phase_one_refinements={result.phase_one_refinement_count}, "
        f"phase_one_rho={result.phase_one_rho:+.6e}, "
        f"phase_two_solves={phase_two_solves}, "
        "checking_intervals="
        f"{len(result.transcription.sms_ia_constraint_rows)}, "
        f"objective={result.phase_two.objective:.12g}, "
        "kkt_stationarity="
        f"{result.kkt_check.stationarity_residual:.6e}, "
        f"dense_max_path={dense_max_path_value:+.6e}"
    )
    print(f"  solution={solution_path.resolve()}")
    print(f"  summary={summary_path.resolve()}")
    print(f"  plot={plot_path.resolve()}")


def main() -> None:
    shooting_grid = np.linspace(0.0, 1.0, NUM_INTERVALS + 1)
    ocp = build_ocp()
    initial_guess = build_initial_guess(ocp, shooting_grid)

    smsia_result = solve_sms_ocp(
        ocp,
        shooting_grid,
        initial_guess,
        sms_ia_options=SMSIAOptions(
            default_smoothing_parameter=0.01,
        ),
        algorithm_options=SMSAlgorithmOptions(
            max_phase_one_refinements=12,
            max_phase_two_refinements=10,
            sms_upper_bound_active_tolerance=1e-6,
        ),
        kkt_options=SMSKKTOptions(
            samples_per_atomic_interval=21,
            active_tolerance=1e-3,
            stationarity_tolerance=1e-3,
        ),
        integrator_options=INTEGRATOR_OPTIONS,
        solver_options=SOLVER_OPTIONS,
    )
    save_smsia_results(smsia_result)

if __name__ == "__main__":
    main()
