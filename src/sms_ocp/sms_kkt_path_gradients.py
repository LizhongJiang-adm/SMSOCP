"""Evaluate path-constraint gradients at selected KKT points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.integrator import build_interval_integrator
from sms_ocp.sms_ia_transcription import SMSIATranscription
from sms_ocp.sms_kkt_path_sampling import (
    SampledPathConstraintPoint,
)
from sms_ocp.utils import CasadiExpr


_POINT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class SMSKKTPathGradientEvaluation:
    """Selected path points and matching full-decision Jacobian rows."""

    points: tuple[SampledPathConstraintPoint, ...]
    jacobian: np.ndarray


@dataclass(frozen=True, slots=True)
class _PathGradientRequest:
    """Identify one scalar path inequality and trajectory location."""

    inequality_index: int
    shooting_interval_index: int
    normalized_time: float


class SMSKKTPathGradientEvaluator:
    """Differentiate selected path points through shooting trajectories."""

    def __init__(
        self,
        transcription: SMSIATranscription,
    ) -> None:
        if not transcription.is_initialized:
            raise RuntimeError(
                "Initialize the SMS-IA transcription before "
                "building its KKT path-gradient evaluator."
            )

        self.transcription = transcription
        model = transcription.ocp.model
        if transcription.scalar_inequalities:
            path_expressions = ca.vertcat(
                *(
                    inequality.expr
                    for inequality in transcription.scalar_inequalities
                )
            )
            self._path_value_function: ca.Function | None = (
                model.create_function(
                    "sms_kkt_path_gradient_values",
                    path_expressions,
                    output_name="values",
                )
            )
        else:
            self._path_value_function = None

        self._state_integrator_cache: dict[
            tuple[float, ...],
            ca.Function,
        ] = {}
        self._gradient_function_cache: dict[
            tuple[_PathGradientRequest, ...],
            ca.Function,
        ] = {}

    def evaluate(
        self,
        decision_point: ArrayLike,
        points: Sequence[SampledPathConstraintPoint],
    ) -> SMSKKTPathGradientEvaluation:
        """Evaluate one full decision-gradient row per selected point."""
        point = np.asarray(decision_point, dtype=float).reshape(-1)
        decision_size = self.transcription.decision_layout.size
        if point.size != decision_size:
            raise ValueError(
                f"decision_point must contain {decision_size} entries."
            )
        if not np.isfinite(point).all():
            raise ValueError(
                "decision_point must contain only finite values."
            )

        point_records = tuple(points)
        if not point_records:
            return SMSKKTPathGradientEvaluation(
                points=(),
                jacobian=np.empty((0, decision_size)),
            )
        if self._path_value_function is None:
            raise ValueError(
                "points must be empty when the transcription has no "
                "SMS-IA path inequalities."
            )

        requests = self._normalize_requests(
            point,
            point_records,
        )
        jacobian = np.asarray(
            self._get_gradient_function(requests)(point),
            dtype=float,
        ).reshape(len(requests), decision_size)
        if not np.isfinite(jacobian).all():
            raise RuntimeError(
                "Path-gradient evaluation returned non-finite values."
            )

        return SMSKKTPathGradientEvaluation(
            points=point_records,
            jacobian=jacobian,
        )

    def _normalize_requests(
        self,
        decision_point: np.ndarray,
        points: tuple[SampledPathConstraintPoint, ...],
    ) -> tuple[_PathGradientRequest, ...]:
        """Validate points and merge nearby times within each interval."""
        transcription = self.transcription
        grid = transcription.shooting_grid
        number_of_inequalities = len(
            transcription.scalar_inequalities
        )
        preliminary_requests: list[_PathGradientRequest] = []

        terminal_time = (
            float(
                transcription.decision_layout.extract(
                    decision_point,
                    "T",
                ).item()
            )
            if transcription.decision_layout.has_block("T")
            else float(transcription.terminal_time)
        )
        initial_time = transcription.ocp.t0
        horizon_duration = terminal_time - initial_time
        if horizon_duration <= 0.0:
            raise ValueError(
                "The terminal time at decision_point must be "
                "greater than the initial time."
            )

        for path_point in points:
            if not isinstance(
                path_point,
                SampledPathConstraintPoint,
            ):
                raise TypeError(
                    "points must contain only "
                    "SampledPathConstraintPoint records."
                )

            inequality_index = path_point.inequality_index
            if (
                isinstance(inequality_index, (bool, np.bool_))
                or not isinstance(
                    inequality_index,
                    (int, np.integer),
                )
                or not 0 <= inequality_index < number_of_inequalities
            ):
                raise IndexError(
                    "Each path point must contain a valid "
                    "inequality_index."
                )

            shooting_interval_index = (
                path_point.shooting_interval_index
            )
            if (
                isinstance(
                    shooting_interval_index,
                    (bool, np.bool_),
                )
                or not isinstance(
                    shooting_interval_index,
                    (int, np.integer),
                )
                or not 0
                <= shooting_interval_index
                < transcription.num_intervals
            ):
                raise IndexError(
                    "Each path point must contain a valid "
                    "shooting_interval_index."
                )

            normalized_time = float(path_point.normalized_time)
            if not np.isfinite(normalized_time):
                raise ValueError(
                    "Path-point normalized times must be finite."
                )
            shooting_left = float(
                grid[shooting_interval_index]
            )
            shooting_right = float(
                grid[shooting_interval_index + 1]
            )
            if np.isclose(
                normalized_time,
                shooting_left,
                rtol=0.0,
                atol=_POINT_TOLERANCE,
            ):
                normalized_time = shooting_left
            elif np.isclose(
                normalized_time,
                shooting_right,
                rtol=0.0,
                atol=_POINT_TOLERANCE,
            ):
                normalized_time = shooting_right
            elif not shooting_left < normalized_time < shooting_right:
                raise ValueError(
                    "Each path point must lie in its recorded "
                    "shooting interval."
                )

            expected_time = (
                initial_time
                + normalized_time * horizon_duration
            )
            if not np.isclose(
                path_point.time,
                expected_time,
                rtol=1e-10,
                atol=1e-10,
            ):
                raise ValueError(
                    "Each path-point time must match decision_point "
                    "and normalized_time."
                )
            preliminary_requests.append(
                _PathGradientRequest(
                    inequality_index=int(inequality_index),
                    shooting_interval_index=int(
                        shooting_interval_index
                    ),
                    normalized_time=normalized_time,
                )
            )

        canonical_times_by_interval: dict[
            int,
            tuple[float, ...],
        ] = {}
        for shooting_interval_index in {
            request.shooting_interval_index
            for request in preliminary_requests
        }:
            canonical_times: list[float] = []
            for normalized_time in sorted(
                request.normalized_time
                for request in preliminary_requests
                if request.shooting_interval_index
                == shooting_interval_index
            ):
                if (
                    not canonical_times
                    or not np.isclose(
                        normalized_time,
                        canonical_times[-1],
                        rtol=0.0,
                        atol=_POINT_TOLERANCE,
                    )
                ):
                    canonical_times.append(normalized_time)
            canonical_times_by_interval[
                shooting_interval_index
            ] = tuple(canonical_times)

        return tuple(
            _PathGradientRequest(
                inequality_index=request.inequality_index,
                shooting_interval_index=(
                    request.shooting_interval_index
                ),
                normalized_time=min(
                    canonical_times_by_interval[
                        request.shooting_interval_index
                    ],
                    key=lambda point: abs(
                        point - request.normalized_time
                    ),
                ),
            )
            for request in preliminary_requests
        )

    def _get_gradient_function(
        self,
        requests: tuple[_PathGradientRequest, ...],
    ) -> ca.Function:
        """Return a cached full-decision path-gradient function."""
        if requests not in self._gradient_function_cache:
            function_index = len(self._gradient_function_cache)
            self._gradient_function_cache[requests] = (
                self._build_gradient_function(
                    requests,
                    name=f"sms_kkt_path_gradients_{function_index}",
                )
            )
        return self._gradient_function_cache[requests]

    def _build_gradient_function(
        self,
        requests: tuple[_PathGradientRequest, ...],
        *,
        name: str,
    ) -> ca.Function:
        """Build one graph with one state propagation per used interval."""
        transcription = self.transcription
        model = transcription.ocp.model
        horizon_duration = (
            transcription.terminal_time - transcription.ocp.t0
        )
        requests_by_interval: dict[
            int,
            list[_PathGradientRequest],
        ] = {}
        for request in requests:
            requests_by_interval.setdefault(
                request.shooting_interval_index,
                [],
            ).append(request)

        state_expressions: dict[tuple[int, float], ca.MX] = {}
        for shooting_interval_index, interval_requests in (
            requests_by_interval.items()
        ):
            shooting_left = float(
                transcription.shooting_grid[
                    shooting_interval_index
                ]
            )
            unique_times = sorted(
                {
                    request.normalized_time
                    for request in interval_requests
                }
            )
            times_after_left = [
                normalized_time
                for normalized_time in unique_times
                if normalized_time > shooting_left
            ]
            if shooting_left in unique_times:
                state_expressions[
                    shooting_interval_index,
                    shooting_left,
                ] = transcription.state_nodes[
                    shooting_interval_index
                ]
            if not times_after_left:
                continue

            farthest_time = times_after_left[-1]
            output_points = tuple(
                (normalized_time - shooting_left)
                / (farthest_time - shooting_left)
                for normalized_time in times_after_left
            )
            integrator_inputs: dict[
                str,
                CasadiExpr | float,
            ] = {
                "x0": transcription.state_nodes[
                    shooting_interval_index
                ],
                "start_time": (
                    transcription.ocp.t0
                    + shooting_left * horizon_duration
                ),
                "duration": (
                    (farthest_time - shooting_left)
                    * horizon_duration
                ),
            }
            if model.nu > 0:
                integrator_inputs["u"] = (
                    transcription.control_intervals[
                        shooting_interval_index
                    ]
                )
            if model.np > 0:
                assert transcription.parameter_vector is not None
                integrator_inputs["p"] = (
                    transcription.parameter_vector
                )

            state_outputs = self._get_state_integrator(
                output_points
            )(**integrator_inputs)["state"]
            for output_index, normalized_time in enumerate(
                times_after_left
            ):
                state_expressions[
                    shooting_interval_index,
                    normalized_time,
                ] = state_outputs[:, output_index]

        assert self._path_value_function is not None
        path_expressions: list[ca.MX] = []
        for request in requests:
            shooting_interval_index = (
                request.shooting_interval_index
            )
            model_inputs = model.function_inputs(
                t=(
                    transcription.ocp.t0
                    + request.normalized_time * horizon_duration
                ),
                x=state_expressions[
                    shooting_interval_index,
                    request.normalized_time,
                ],
                u=(
                    transcription.control_intervals[
                        shooting_interval_index
                    ]
                    if model.nu > 0
                    else None
                ),
                p=transcription.parameter_vector,
            )
            path_values = self._path_value_function(
                *model_inputs.values()
            )
            path_expressions.append(
                path_values[request.inequality_index]
            )

        path_vector = ca.vertcat(*path_expressions)
        jacobian = ca.jacobian(
            path_vector,
            transcription.decision_vector,
        )
        return ca.Function(
            name,
            [transcription.decision_vector],
            [jacobian],
            ["decision_variables"],
            ["jacobian"],
        )

    def _get_state_integrator(
        self,
        output_points: tuple[float, ...],
    ) -> ca.Function:
        """Return a cached state-only multi-output integrator."""
        if output_points not in self._state_integrator_cache:
            integrator_index = len(
                self._state_integrator_cache
            )
            self._state_integrator_cache[output_points] = (
                build_interval_integrator(
                    self.transcription.ocp.model,
                    ca.SX.zeros(0, 1),
                    output_points,
                    name=(
                        "sms_kkt_path_gradient_state_"
                        f"{integrator_index}"
                    ),
                    options=(
                        self.transcription.integrator_options
                    ),
                )
            )
        return self._state_integrator_cache[output_points]
