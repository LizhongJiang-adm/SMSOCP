"""Solve the Rayleigh mixed-constraint problem with the complete SMS algorithm."""

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


NUM_INTERVALS = 20
SAMPLES_PER_INTERVAL = 200
OUTPUT_DIRECTORY = Path(__file__).with_name("outputs")
POSTPROCESS_INTEGRATOR_OPTIONS = {
    **INTEGRATOR_OPTIONS,
    "reltol": 1e-10,
    "abstol": 1e-10,
}


class RayleighMixedModel(BaseDynamicModel):
    """Nonlinear Rayleigh oscillator with one control input."""

    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 2)
        self.u_sym = ca.SX.sym("u")
        x_1, x_2 = self.x_sym[0], self.x_sym[1]
        u = self.u_sym[0]
        self.ode_expr = ca.vertcat(
            x_2,
            -x_1 + x_2 * (1.4 - 0.14 * x_2**2) + 4.0 * u,
        )


def build_ocp() -> OptimalControlProblem:
    model = RayleighMixedModel()
    ocp = OptimalControlProblem(model)
    x, u = model.x_sym, model.u_sym[0]
    ocp.set_time_horizon(tf=4.5)
    ocp.set_objective(lagrange=x[0] ** 2 + u**2)
    ocp.add_initial_constraint(x, [-5.0, -5.0])
    ocp.add_path_constraint(
        u + x[0] / 6.0,
        name="mixed_limit",
        enforcement="sms_ia",
    )
    return ocp


def build_initial_guess(
    ocp: OptimalControlProblem,
    shooting_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    controls = np.zeros((1, NUM_INTERVALS))
    states = np.zeros((2, NUM_INTERVALS + 1))
    states[:, 0] = [-5.0, -5.0]
    step = build_interval_integrator(
        ocp.model,
        ca.SX.zeros(0, 1),
        [1.0],
        name="rayleigh_initial_guess",
        options=INTEGRATOR_OPTIONS,
    )
    horizon = float(ocp.tf_bounds[0] - ocp.t0)

    for interval_index in range(NUM_INTERVALS):
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
    return states, controls


def postprocess_solution(result: SMSAlgorithmResult) -> None:
    """Densely reconstruct, check, save, and plot the solved trajectory."""
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
        name="rayleigh_mixed_postprocess",
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
        time_segments.append(
            start_time + duration * local_points[:point_count]
        )
        residual_segments.append(
            controls[0, interval_index]
            + interval_states[0, :point_count] / 6.0
        )

    time = np.concatenate(time_segments)
    path_residual = np.concatenate(residual_segments)
    maximum_index = int(np.argmax(path_residual))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    solution_path = OUTPUT_DIRECTORY / "rayleigh_mixed_solution.npz"
    plot_path = OUTPUT_DIRECTORY / "rayleigh_mixed_path_constraints.png"
    np.savez(
        solution_path,
        decision_vector=solution.decision_vector,
        shooting_grid=transcription.shooting_grid,
        objective=solution.objective,
        dense_time=time,
        dense_path_residual=path_residual,
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(time, path_residual, label=r"$u(t)+x_1(t)/6$")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        title="Rayleigh mixed path constraint",
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
        "Rayleigh mixed: "
        f"success={stats['success']}, "
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
    states, controls = build_initial_guess(ocp, shooting_grid)
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
        ),
        integrator_options=INTEGRATOR_OPTIONS,
        solver_options=SOLVER_OPTIONS,
    )
    postprocess_solution(result)


if __name__ == "__main__":
    main()
