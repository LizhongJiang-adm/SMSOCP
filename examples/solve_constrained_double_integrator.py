"""Solve a constrained double-integrator problem with SMS-OCP."""

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


class ConstrainedDoubleIntegrator(BaseDynamicModel):
    """Double integrator with a continuous-time position constraint."""

    def setup_model(self) -> None:
        self.x_sym = ca.SX.sym("x", 2)
        self.u_sym = ca.SX.sym("u")
        self.ode_expr = ca.vertcat(self.x_sym[1], self.u_sym[0])


def build_ocp() -> OptimalControlProblem:
    model = ConstrainedDoubleIntegrator()
    ocp = OptimalControlProblem(model)
    x = model.x_sym
    u = model.u_sym[0]
    ocp.set_time_horizon(tf=1.0)
    ocp.set_objective(lagrange=0.5 * u**2)
    ocp.add_initial_constraint(x, [0.0, 1.0])
    ocp.add_terminal_constraint(x, [0.0, -1.0])
    ocp.add_path_constraint(
        x[0],
        ub=1.0 / 9.0,
        name="position_limit",
        enforcement="sms_ia",
    )
    return ocp


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
        name="constrained_double_integrator_postprocess",
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
        residual_segments.append(interval_states[0, :point_count] - 1.0 / 9.0)

    time = np.concatenate(time_segments)
    path_residual = np.concatenate(residual_segments)
    maximum_index = int(np.argmax(path_residual))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    solution_path = (
        OUTPUT_DIRECTORY / "constrained_double_integrator_solution.npz"
    )
    plot_path = (
        OUTPUT_DIRECTORY
        / "constrained_double_integrator_path_constraints.png"
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
    axis.plot(time, path_residual, label=r"$x_1(t)-1/9$")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        title="Constrained double-integrator path constraint",
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
        "Constrained Double Integrator: "
        f"success={stats['success']}, "
        f"status={stats['return_status']}, "
        f"objective={solution.objective:.10g}, "
        f"dense_max_path={path_residual[maximum_index]:+.3e}, "
        f"maximum_time={time[maximum_index]:.6g}"
    )
    print(f"  solution={solution_path.resolve()}")
    print(f"  plot={plot_path.resolve()}")


def main() -> None:
    ocp = build_ocp()
    shooting_grid = np.linspace(0.0, 1.0, NUM_INTERVALS + 1)
    states = np.vstack(
        (
            shooting_grid * (1.0 - shooting_grid),
            1.0 - 2.0 * shooting_grid,
        )
    )
    controls = np.full((1, NUM_INTERVALS), -2.0)
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
