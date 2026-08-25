"""Select active path-constraint sample points for SMS KKT gradients."""

from __future__ import annotations

from typing import Literal

import numpy as np


PathGradientPointStrategy = Literal[
    "maximum",
    "sparse_active",
]


def select_active_path_point_indices(
    scaled_values: np.ndarray,
    *,
    strategy: PathGradientPointStrategy,
    active_tolerance: float,
    active_point_sample_stride: int = 3,
) -> tuple[int, ...]:
    """Select local sample indices used for KKT path gradients."""
    assert strategy in ("maximum", "sparse_active")
    assert scaled_values.ndim == 1
    assert np.isfinite(scaled_values).all()
    assert active_tolerance >= 0.0
    assert active_point_sample_stride >= 1

    if scaled_values.size == 0:
        return ()

    if strategy == "maximum":
        maximum_index = int(np.argmax(scaled_values))
        if scaled_values[maximum_index] < -active_tolerance:
            return ()
        return (maximum_index,)

    if strategy == "sparse_active":
        active_indices = np.flatnonzero(
            scaled_values >= -active_tolerance
        )
        if active_indices.size == 0:
            return ()

        split_locations = (
            np.flatnonzero(np.diff(active_indices) > 1) + 1
        )
        active_regions = np.split(active_indices, split_locations)
        selected_indices: list[int] = []

        for active_region in active_regions:
            maximum_index = int(
                active_region[
                    np.argmax(scaled_values[active_region])
                ]
            )
            selected_indices.append(maximum_index)

            last_selected_index = maximum_index
            for sample_index in reversed(
                active_region[active_region < maximum_index]
            ):
                sample_index = int(sample_index)
                if (
                    last_selected_index - sample_index
                    >= active_point_sample_stride
                ):
                    selected_indices.append(sample_index)
                    last_selected_index = sample_index

            last_selected_index = maximum_index
            for sample_index in active_region[
                active_region > maximum_index
            ]:
                sample_index = int(sample_index)
                if (
                    sample_index - last_selected_index
                    >= active_point_sample_stride
                ):
                    selected_indices.append(sample_index)
                    last_selected_index = sample_index

        return tuple(sorted(selected_indices))

    raise AssertionError("Unreachable path-point strategy.")
