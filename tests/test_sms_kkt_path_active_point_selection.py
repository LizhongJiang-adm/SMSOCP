import numpy as np

from sms_ocp.sms_kkt_path_active_point_selection import (
    select_active_path_point_indices,
)


def test_selects_maximum_or_sparse_active_points() -> None:
    scaled_values = np.array(
        [
            -2.0,
            -5e-4,
            -4e-4,
            -3e-4,
            -2e-4,
            -1e-4,
            -2e-4,
            -3e-4,
            -4e-4,
            -5e-4,
            -2.0,
            -5e-4,
            -1e-4,
            -5e-4,
        ]
    )

    maximum_indices = select_active_path_point_indices(
        scaled_values,
        strategy="maximum",
        active_tolerance=1e-3,
    )
    sparse_indices = select_active_path_point_indices(
        scaled_values,
        strategy="sparse_active",
        active_tolerance=1e-3,
        active_point_sample_stride=3,
    )

    assert maximum_indices == (5,)
    assert sparse_indices == (2, 5, 8, 12)


def test_returns_no_points_when_constraint_is_inactive() -> None:
    scaled_values = np.array([-2e-3, -3e-3])

    assert (
        select_active_path_point_indices(
            scaled_values,
            strategy="maximum",
            active_tolerance=1e-3,
        )
        == ()
    )
    assert (
        select_active_path_point_indices(
            scaled_values,
            strategy="sparse_active",
            active_tolerance=1e-3,
        )
        == ()
    )
