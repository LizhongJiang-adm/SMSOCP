"""Checking-interval validation and integration-grid planning for SMS-IA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.sms_ia_inequality_path_constraints import (
    ScalarPathInequality,
)
from sms_ocp.utils import normalize_grid_points


@dataclass(frozen=True)
class InequalityCheckingIntervals:
    """Store global checking intervals for one scalar inequality."""

    inequality_index: int
    intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class CheckingIntervalUpdate:
    """Replace one inequality's intervals within one shooting interval."""

    inequality_name: str
    shooting_interval_index: int
    intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CheckingIntervalPlan:
    """Locate one checking interval in a shooting integrator's outputs.

    Point indices address ``(0.0, *output_points)``. Column zero therefore
    means the shooting-node state and zero cumulative quadrature.
    ``inequality_checking_interval_index`` addresses the interval in the
    corresponding inequality's complete checking-interval sequence.
    """

    inequality_index: int
    inequality_checking_interval_index: int
    left_point_index: int
    right_point_index: int


@dataclass(frozen=True)
class ShootingIntervalPlan:
    """Store the shared integration grid and checks for one shooting interval.

    ``output_points`` contains local normalized times in ``(0, 1]``. All
    checking intervals reuse these outputs through their stored point indices.
    """

    output_points: tuple[float, ...]
    checking_intervals: tuple[CheckingIntervalPlan, ...]


def resolve_initial_checking_intervals(
    shooting_grid: ArrayLike,
    scalar_inequalities: Sequence[ScalarPathInequality],
    checking_interval_overrides: Mapping[
        str,
        Sequence[tuple[float, float]],
    ] | None = None,
) -> tuple[InequalityCheckingIntervals, ...]:
    """Resolve default intervals and name-based user overrides."""
    grid = normalize_grid_points(shooting_grid)
    default_intervals = tuple(
        (float(left), float(right))
        for left, right in zip(grid[:-1], grid[1:])
    )
    overrides = dict(checking_interval_overrides or {})

    valid_names = {
        inequality.name
        for inequality in scalar_inequalities
    }
    unknown_names = set(overrides).difference(valid_names)
    if unknown_names:
        names = ", ".join(
            repr(name)
            for name in sorted(unknown_names)
        )
        raise ValueError(
            f"Unknown checking-interval override names: {names}."
        )

    return tuple(
        InequalityCheckingIntervals(
            inequality_index=inequality_index,
            intervals=_normalize_checking_intervals(
                overrides.get(inequality.name, default_intervals),
                shooting_grid=grid,
                inequality_name=inequality.name,
            ),
        )
        for inequality_index, inequality in enumerate(scalar_inequalities)
    )


def _normalize_checking_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    shooting_grid: np.ndarray,
    inequality_name: str,
) -> tuple[tuple[float, float], ...]:
    """Normalize and validate one scalar inequality's intervals."""
    if len(intervals) == 0:
        raise ValueError(
            f"Checking intervals for {inequality_name!r} "
            "must not be empty."
        )

    def snap_to_shooting_node(value: float) -> float:
        distances = np.abs(shooting_grid - value)
        closest_index = int(np.argmin(distances))
        if distances[closest_index] <= 1e-12:
            return float(shooting_grid[closest_index])
        return value

    normalized: list[tuple[float, float]] = []
    previous_right: float | None = None

    for interval_index, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError(
                f"Checking interval {interval_index} for "
                f"{inequality_name!r} must contain two endpoints."
            )

        left, right = float(interval[0]), float(interval[1])
        if not np.isfinite(left) or not np.isfinite(right):
            raise ValueError(
                f"Checking intervals for {inequality_name!r} "
                "must contain only finite endpoints."
            )

        left = snap_to_shooting_node(left)
        right = snap_to_shooting_node(right)
        if (
            previous_right is not None
            and abs(left - previous_right) <= 1e-12
        ):
            left = previous_right

        if not shooting_grid[0] <= left < right <= shooting_grid[-1]:
            raise ValueError(
                f"Checking intervals for {inequality_name!r} "
                "must lie within the shooting grid and satisfy left < right."
            )

        shooting_interval_index = int(
            np.searchsorted(
                shooting_grid,
                left,
                side="right",
            )
            - 1
        )
        if (
            shooting_interval_index < 0
            or shooting_interval_index >= shooting_grid.size - 1
            or right > shooting_grid[shooting_interval_index + 1]
        ):
            raise ValueError(
                f"Checking interval {(left, right)} for "
                f"{inequality_name!r} must not cross a shooting node."
            )

        if previous_right is not None and left < previous_right:
            raise ValueError(
                f"Checking intervals for {inequality_name!r} "
                "must be ordered and non-overlapping."
            )

        normalized.append((left, right))
        previous_right = right

    return tuple(normalized)


def resolve_updated_checking_intervals(
    shooting_grid: ArrayLike,
    current_checking_intervals: Sequence[
        InequalityCheckingIntervals
    ],
    updates: Sequence[CheckingIntervalUpdate],
    *,
    inequality_index_by_name: Mapping[str, int],
) -> tuple[
    tuple[InequalityCheckingIntervals, ...],
    frozenset[int],
]:
    """Apply complete local replacements to a copy of the current intervals."""
    grid = normalize_grid_points(shooting_grid)
    num_shooting_intervals = grid.size - 1
    updated_checking_intervals = list(current_checking_intervals)
    seen_update_keys: set[tuple[str, int]] = set()
    changed_shooting_interval_indices: set[int] = set()

    for update in updates:
        if not isinstance(update.inequality_name, str):
            raise TypeError("inequality_name must be a string.")
        try:
            inequality_index = inequality_index_by_name[
                update.inequality_name
            ]
        except KeyError:
            raise ValueError(
                "Unknown scalar inequality name: "
                f"{update.inequality_name!r}."
            ) from None

        shooting_interval_index = update.shooting_interval_index
        if (
            isinstance(shooting_interval_index, (bool, np.bool_))
            or not isinstance(
                shooting_interval_index,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "shooting_interval_index must be an integer."
            )
        shooting_interval_index = int(shooting_interval_index)
        if not 0 <= shooting_interval_index < num_shooting_intervals:
            raise IndexError(
                "shooting_interval_index must be within "
                f"[0, {num_shooting_intervals})."
            )

        update_key = (
            update.inequality_name,
            shooting_interval_index,
        )
        if update_key in seen_update_keys:
            raise ValueError(
                "Each (inequality_name, shooting_interval_index) "
                "pair may be updated at most once per call."
            )
        seen_update_keys.add(update_key)

        shooting_left = float(grid[shooting_interval_index])
        shooting_right = float(grid[shooting_interval_index + 1])
        new_local_intervals = _normalize_checking_intervals(
            update.intervals,
            shooting_grid=grid[
                shooting_interval_index:
                shooting_interval_index + 2
            ],
            inequality_name=update.inequality_name,
        )

        assert (
            updated_checking_intervals[
                inequality_index
            ].inequality_index
            == inequality_index
        )
        existing_intervals = updated_checking_intervals[
            inequality_index
        ].intervals
        old_local_intervals = tuple(
            interval
            for interval in existing_intervals
            if shooting_left <= interval[0] < shooting_right
        )
        if new_local_intervals == old_local_intervals:
            continue

        preserved_intervals = tuple(
            interval
            for interval in existing_intervals
            if not shooting_left <= interval[0] < shooting_right
        )
        updated_checking_intervals[inequality_index] = (
            InequalityCheckingIntervals(
                inequality_index=inequality_index,
                intervals=tuple(
                    sorted(
                        (
                            *preserved_intervals,
                            *new_local_intervals,
                        )
                    )
                ),
            )
        )
        changed_shooting_interval_indices.add(
            shooting_interval_index
        )

    return (
        tuple(updated_checking_intervals),
        frozenset(changed_shooting_interval_indices),
    )


def build_shooting_interval_plans(
    shooting_grid: ArrayLike,
    checking_intervals: Sequence[InequalityCheckingIntervals],
) -> tuple[ShootingIntervalPlan, ...]:
    """Map global checking intervals to local integrator output points."""
    grid = normalize_grid_points(shooting_grid)
    num_shooting_intervals = grid.size - 1

    output_point_sets: list[set[float]] = [
        {1.0}
        for _ in range(num_shooting_intervals)
    ]
    checking_records: list[
        list[tuple[int, int, float, float]]
    ] = [
        []
        for _ in range(num_shooting_intervals)
    ]

    for inequality_checking in checking_intervals:
        inequality_index = inequality_checking.inequality_index

        for inequality_checking_interval_index, (left, right) in enumerate(
            inequality_checking.intervals
        ):
            left = float(left)
            right = float(right)
            shooting_interval_index = int(
                np.searchsorted(
                    grid,
                    left,
                    side="right",
                )
                - 1
            )
            assert 0 <= shooting_interval_index < num_shooting_intervals

            shooting_left = float(grid[shooting_interval_index])
            shooting_right = float(grid[shooting_interval_index + 1])
            assert shooting_left <= left < right <= shooting_right

            shooting_duration = shooting_right - shooting_left
            local_left = (
                0.0
                if left == shooting_left
                else (left - shooting_left) / shooting_duration
            )
            local_right = (
                1.0
                if right == shooting_right
                else (right - shooting_left) / shooting_duration
            )

            if local_left > 0.0:
                output_point_sets[shooting_interval_index].add(
                    local_left
                )
            output_point_sets[shooting_interval_index].add(local_right)
            checking_records[shooting_interval_index].append(
                (
                    inequality_index,
                    inequality_checking_interval_index,
                    local_left,
                    local_right,
                )
            )

    shooting_interval_plans: list[ShootingIntervalPlan] = []

    for shooting_interval_index in range(num_shooting_intervals):
        output_points = tuple(
            sorted(output_point_sets[shooting_interval_index])
        )
        point_indices = {
            0.0: 0,
            **{
                point: point_index + 1
                for point_index, point in enumerate(output_points)
            },
        }
        interval_plans = tuple(
            CheckingIntervalPlan(
                inequality_index=inequality_index,
                inequality_checking_interval_index=(
                    inequality_checking_interval_index
                ),
                left_point_index=point_indices[local_left],
                right_point_index=point_indices[local_right],
            )
            for (
                inequality_index,
                inequality_checking_interval_index,
                local_left,
                local_right,
            ) in checking_records[shooting_interval_index]
        )
        shooting_interval_plans.append(
            ShootingIntervalPlan(
                output_points=output_points,
                checking_intervals=interval_plans,
            )
        )

    return tuple(shooting_interval_plans)
