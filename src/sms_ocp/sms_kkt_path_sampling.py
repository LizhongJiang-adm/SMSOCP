"""Sample path-constraint values for SMS KKT checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.integrator import build_interval_integrator
from sms_ocp.sms_ia_checking_intervals import ShootingIntervalPlan
from sms_ocp.sms_ia_transcription import SMSIATranscription
from sms_ocp.sms_kkt_path_active_point_selection import (
    PathGradientPointStrategy,
    select_active_path_point_indices,
)


_POINT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class SampledPathConstraintPoint:
    """Identify one sampled scalar path-constraint value."""

    inequality_index: int
    shooting_interval_index: int
    inequality_checking_interval_index: int
    shooting_interval_sample_index: int
    normalized_time: float
    time: float
    value: float


@dataclass(frozen=True, slots=True)
class SMSKKTPathSamplingResult:
    """Store path maxima and points selected for KKT gradients."""

    path_maxima: tuple[SampledPathConstraintPoint, ...]
    active_path_points: tuple[SampledPathConstraintPoint, ...]


@dataclass(frozen=True, slots=True)
class _CheckingIntervalSampleRange:
    """Locate one checking interval in a shooting sample grid."""

    inequality_index: int
    inequality_checking_interval_index: int
    sample_start_index: int
    sample_stop_index: int


@dataclass(frozen=True, slots=True)
class _ShootingIntervalPathSamplingPlan:
    """Store one shared path-constraint sample grid."""

    local_sample_points: tuple[float, ...]
    integrator_output_points: tuple[float, ...]
    checking_intervals: tuple[_CheckingIntervalSampleRange, ...]


class SMSKKTPathSampler:
    """Sample path maxima and active points on shared trajectories."""

    def __init__(
        self,
        transcription: SMSIATranscription,
        *,
        samples_per_atomic_interval: int = 41,
        active_tolerance: float = 1e-6,
        active_point_sample_stride: int = 3,
        path_gradient_point_strategy: (
            PathGradientPointStrategy
        ) = "sparse_active",
        path_constraint_scales: ArrayLike | None = None,
        integrator_options: Mapping[str, object] | None = None,
    ) -> None:
        if not transcription.is_initialized:
            raise RuntimeError(
                "Initialize the SMS-IA transcription before "
                "building its KKT path sampler."
            )

        self.transcription = transcription
        if (
            isinstance(
                samples_per_atomic_interval,
                (bool, np.bool_),
            )
            or not isinstance(
                samples_per_atomic_interval,
                (int, np.integer),
            )
            or samples_per_atomic_interval < 2
        ):
            raise ValueError(
                "samples_per_atomic_interval must be "
                "an integer greater than or equal to 2."
            )
        self.samples_per_atomic_interval = int(
            samples_per_atomic_interval
        )
        self.active_tolerance = float(active_tolerance)
        self.active_point_sample_stride = int(
            active_point_sample_stride
        )
        self.path_gradient_point_strategy = (
            path_gradient_point_strategy
        )
        self.path_constraint_scales = (
            np.ones(len(transcription.scalar_inequalities))
            if path_constraint_scales is None
            else np.asarray(
                path_constraint_scales,
                dtype=float,
            ).reshape(-1)
        )
        assert self.active_tolerance >= 0.0
        assert self.active_point_sample_stride >= 1
        assert self.path_gradient_point_strategy in (
            "maximum",
            "sparse_active",
        )
        assert self.path_constraint_scales.shape == (
            len(transcription.scalar_inequalities),
        )
        assert np.isfinite(self.path_constraint_scales).all()
        assert np.all(self.path_constraint_scales > 0.0)
        self.integrator_options = {
            **transcription.integrator_options,
            **dict(integrator_options or {}),
        }
        model = transcription.ocp.model
        if transcription.scalar_inequalities:
            path_expression = ca.vertcat(
                *(
                    inequality.expr
                    for inequality in transcription.scalar_inequalities
                )
            )
            self._path_value_function: ca.Function | None = (
                model.create_function(
                    "sms_kkt_path_values",
                    path_expression,
                    output_name="values",
                )
            )
        else:
            self._path_value_function = None

        self._state_integrator_cache: dict[
            tuple[float, ...],
            ca.Function,
        ] = {}
        self._mapped_path_function_cache: dict[int, ca.Function] = {}

    def compute(
        self,
        decision_point: ArrayLike,
    ) -> SMSKKTPathSamplingResult:
        """Sample path maxima and select active KKT gradient points."""
        if self._path_value_function is None:
            return SMSKKTPathSamplingResult(
                path_maxima=(),
                active_path_points=(),
            )

        (
            state_values,
            control_values,
            parameter_values,
            terminal_time,
        ) = self._extract_decision_values(decision_point)
        initial_time = self.transcription.ocp.t0
        horizon_duration = terminal_time - initial_time
        if horizon_duration <= 0.0:
            raise ValueError(
                "The terminal time at decision_point must be "
                "greater than the initial time."
            )

        path_maxima: list[SampledPathConstraintPoint] = []
        active_path_points: list[SampledPathConstraintPoint] = []
        seen_active_locations: set[tuple[int, int, int]] = set()
        for shooting_interval_index, interval_plan in enumerate(
            self.transcription.interval_plans
        ):
            if not interval_plan.checking_intervals:
                continue

            (
                sampling_plan,
                normalized_times,
                physical_times,
                path_values,
            ) = self._sample_shooting_interval(
                shooting_interval_index,
                interval_plan,
                state_values,
                control_values,
                parameter_values,
                initial_time,
                horizon_duration,
            )
            for check in sampling_plan.checking_intervals:
                interval_values = path_values[
                    check.inequality_index,
                    check.sample_start_index:
                    check.sample_stop_index,
                ]
                maximum_sample_index = (
                    check.sample_start_index
                    + int(np.argmax(interval_values))
                )
                maximum = _build_sampled_path_constraint_point(
                    check,
                    shooting_interval_index=shooting_interval_index,
                    sample_index=maximum_sample_index,
                    normalized_times=normalized_times,
                    physical_times=physical_times,
                    path_values=path_values,
                )
                path_maxima.append(maximum)

                scale = self.path_constraint_scales[
                    check.inequality_index
                ]
                local_active_indices = (
                    select_active_path_point_indices(
                        interval_values / scale,
                        strategy=(
                            self.path_gradient_point_strategy
                        ),
                        active_tolerance=self.active_tolerance,
                        active_point_sample_stride=(
                            self.active_point_sample_stride
                        ),
                    )
                )
                for local_sample_index in local_active_indices:
                    sample_index = (
                        check.sample_start_index
                        + local_sample_index
                    )
                    location = (
                        check.inequality_index,
                        shooting_interval_index,
                        sample_index,
                    )
                    if location in seen_active_locations:
                        continue
                    seen_active_locations.add(location)
                    active_path_points.append(
                        maximum
                        if sample_index == maximum_sample_index
                        else _build_sampled_path_constraint_point(
                            check,
                            shooting_interval_index=(
                                shooting_interval_index
                            ),
                            sample_index=sample_index,
                            normalized_times=normalized_times,
                            physical_times=physical_times,
                            path_values=path_values,
                        )
                    )

        return SMSKKTPathSamplingResult(
            path_maxima=tuple(path_maxima),
            active_path_points=tuple(active_path_points),
        )

    def _extract_decision_values(
        self,
        decision_point: ArrayLike,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ]:
        """Validate and unpack one decision point."""
        transcription = self.transcription
        point = np.asarray(decision_point, dtype=float).reshape(-1)
        if point.size != transcription.decision_layout.size:
            raise ValueError(
                "decision_point must contain "
                f"{transcription.decision_layout.size} entries."
            )
        if not np.isfinite(point).all():
            raise ValueError(
                "decision_point must contain only finite values."
            )

        layout = transcription.decision_layout
        state_values = layout.extract(point, "x")
        control_values = (
            layout.extract(point, "u")
            if layout.has_block("u")
            else None
        )
        parameter_values = (
            layout.extract(point, "p")
            if layout.has_block("p")
            else None
        )
        terminal_time = (
            float(layout.extract(point, "T").item())
            if layout.has_block("T")
            else float(transcription.terminal_time)
        )
        return (
            state_values,
            control_values,
            parameter_values,
            terminal_time,
        )

    def _sample_shooting_interval(
        self,
        shooting_interval_index: int,
        interval_plan: ShootingIntervalPlan,
        state_values: np.ndarray,
        control_values: np.ndarray | None,
        parameter_values: np.ndarray | None,
        initial_time: float,
        horizon_duration: float,
    ) -> tuple[
        _ShootingIntervalPathSamplingPlan,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate all scalar path inequalities on one shared grid."""
        transcription = self.transcription
        model = transcription.ocp.model
        sampling_plan = _build_shooting_interval_path_sampling_plan(
            interval_plan,
            samples_per_atomic_interval=(
                self.samples_per_atomic_interval
            ),
        )
        shooting_left = float(
            transcription.shooting_grid[shooting_interval_index]
        )
        shooting_right = float(
            transcription.shooting_grid[shooting_interval_index + 1]
        )
        normalized_shooting_duration = shooting_right - shooting_left
        shooting_start_time = (
            initial_time + shooting_left * horizon_duration
        )
        shooting_duration = (
            normalized_shooting_duration * horizon_duration
        )

        integrator_inputs: dict[str, ArrayLike | float] = {
            "x0": state_values[:, shooting_interval_index],
            "start_time": shooting_start_time,
            "duration": shooting_duration,
        }
        if control_values is not None:
            integrator_inputs["u"] = control_values[
                :,
                shooting_interval_index,
            ]
        if parameter_values is not None:
            integrator_inputs["p"] = parameter_values

        state_outputs = np.asarray(
            self._get_state_integrator(
                sampling_plan.integrator_output_points
            )(**integrator_inputs)["state"],
            dtype=float,
        )
        local_sample_points = np.asarray(
            sampling_plan.local_sample_points,
            dtype=float,
        )
        if np.isclose(
            local_sample_points[0],
            0.0,
            rtol=0.0,
            atol=_POINT_TOLERANCE,
        ):
            state_samples = np.column_stack(
                (
                    state_values[:, shooting_interval_index],
                    state_outputs,
                )
            )
        else:
            state_samples = state_outputs

        normalized_times = (
            shooting_left
            + normalized_shooting_duration * local_sample_points
        )
        physical_times = initial_time + horizon_duration * normalized_times
        number_of_samples = local_sample_points.size
        control_samples = (
            None
            if control_values is None
            else np.repeat(
                control_values[:, [shooting_interval_index]],
                number_of_samples,
                axis=1,
            )
        )
        parameter_samples = (
            None
            if parameter_values is None
            else np.repeat(
                parameter_values,
                number_of_samples,
                axis=1,
            )
        )
        path_inputs = model.function_inputs(
            t=physical_times.reshape(1, -1),
            x=state_samples,
            u=control_samples,
            p=parameter_samples,
        )
        path_values = np.asarray(
            self._get_mapped_path_function(number_of_samples)(
                *path_inputs.values()
            ),
            dtype=float,
        ).reshape(
            len(transcription.scalar_inequalities),
            number_of_samples,
        )
        if not np.isfinite(path_values).all():
            raise RuntimeError(
                "Path-constraint evaluation returned non-finite values "
                f"in shooting interval {shooting_interval_index}."
            )

        return (
            sampling_plan,
            normalized_times,
            physical_times,
            path_values,
        )

    def _get_state_integrator(
        self,
        output_points: tuple[float, ...],
    ) -> ca.Function:
        """Return a cached state-only sampling integrator."""
        if output_points not in self._state_integrator_cache:
            integrator_index = len(self._state_integrator_cache)
            self._state_integrator_cache[output_points] = (
                build_interval_integrator(
                    self.transcription.ocp.model,
                    ca.SX.zeros(0, 1),
                    output_points,
                    name=f"sms_kkt_path_sampler_{integrator_index}",
                    options=self.integrator_options,
                )
            )
        return self._state_integrator_cache[output_points]

    def _get_mapped_path_function(
        self,
        number_of_samples: int,
    ) -> ca.Function:
        """Return a path-value function mapped over sample columns."""
        if number_of_samples not in self._mapped_path_function_cache:
            assert self._path_value_function is not None
            self._mapped_path_function_cache[
                number_of_samples
            ] = self._path_value_function.map(
                number_of_samples,
                "serial",
            )
        return self._mapped_path_function_cache[number_of_samples]


def _build_sampled_path_constraint_point(
    check: _CheckingIntervalSampleRange,
    *,
    shooting_interval_index: int,
    sample_index: int,
    normalized_times: np.ndarray,
    physical_times: np.ndarray,
    path_values: np.ndarray,
) -> SampledPathConstraintPoint:
    """Build one path-point record from shared sampled arrays."""
    return SampledPathConstraintPoint(
        inequality_index=check.inequality_index,
        shooting_interval_index=shooting_interval_index,
        inequality_checking_interval_index=(
            check.inequality_checking_interval_index
        ),
        shooting_interval_sample_index=sample_index,
        normalized_time=float(normalized_times[sample_index]),
        time=float(physical_times[sample_index]),
        value=float(
            path_values[
                check.inequality_index,
                sample_index,
            ]
        ),
    )


def _build_shooting_interval_path_sampling_plan(
    interval_plan: ShootingIntervalPlan,
    *,
    samples_per_atomic_interval: int,
) -> _ShootingIntervalPathSamplingPlan:
    """Expand checking boundaries into a shared atomic sample grid."""
    integration_points = (0.0, *interval_plan.output_points)
    raw_checking_intervals = tuple(
        (
            check,
            float(integration_points[check.left_point_index]),
            float(integration_points[check.right_point_index]),
        )
        for check in interval_plan.checking_intervals
    )

    atomic_boundaries: list[float] = []
    for point in sorted(
        point
        for _, left, right in raw_checking_intervals
        for point in (left, right)
    ):
        if (
            not atomic_boundaries
            or not np.isclose(
                point,
                atomic_boundaries[-1],
                rtol=0.0,
                atol=_POINT_TOLERANCE,
            )
        ):
            atomic_boundaries.append(point)

    checking_intervals = tuple(
        (
            check,
            min(
                atomic_boundaries,
                key=lambda point: abs(point - left),
            ),
            min(
                atomic_boundaries,
                key=lambda point: abs(point - right),
            ),
        )
        for check, left, right in raw_checking_intervals
    )
    sample_points: list[float] = []
    for atomic_left, atomic_right in zip(
        atomic_boundaries[:-1],
        atomic_boundaries[1:],
        strict=True,
    ):
        interval_is_used = any(
            checking_left <= atomic_left
            and atomic_right <= checking_right
            for _, checking_left, checking_right in checking_intervals
        )
        if not interval_is_used:
            continue

        for sample in np.linspace(
            atomic_left,
            atomic_right,
            samples_per_atomic_interval,
        ):
            sample = float(sample)
            if (
                sample_points
                and np.isclose(
                    sample,
                    sample_points[-1],
                    rtol=0.0,
                    atol=_POINT_TOLERANCE,
                )
            ):
                continue
            sample_points.append(sample)

    local_sample_points = np.asarray(sample_points, dtype=float)
    sample_ranges: list[_CheckingIntervalSampleRange] = []
    for check, checking_left, checking_right in checking_intervals:
        sample_start_index = int(
            np.searchsorted(
                local_sample_points,
                checking_left,
                side="left",
            )
        )
        sample_stop_index = int(
            np.searchsorted(
                local_sample_points,
                checking_right,
                side="right",
            )
        )
        assert np.isclose(
            local_sample_points[sample_start_index],
            checking_left,
            rtol=0.0,
            atol=_POINT_TOLERANCE,
        ), (
            "Every checking-interval left endpoint must "
            "appear in the shared path sample grid."
        )
        assert np.isclose(
            local_sample_points[sample_stop_index - 1],
            checking_right,
            rtol=0.0,
            atol=_POINT_TOLERANCE,
        ), (
            "Every checking-interval right endpoint must "
            "appear in the shared path sample grid."
        )
        sample_ranges.append(
            _CheckingIntervalSampleRange(
                inequality_index=check.inequality_index,
                inequality_checking_interval_index=(
                    check.inequality_checking_interval_index
                ),
                sample_start_index=sample_start_index,
                sample_stop_index=sample_stop_index,
            )
        )

    return _ShootingIntervalPathSamplingPlan(
        local_sample_points=tuple(
            float(point)
            for point in local_sample_points
        ),
        integrator_output_points=tuple(
            float(point)
            for point in local_sample_points
            if point > 0.0
        ),
        checking_intervals=tuple(sample_ranges),
    )
