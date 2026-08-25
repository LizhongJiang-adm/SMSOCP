"""Decision-vector layout for multiple-shooting transcriptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from sms_ocp.optimal_control_problem import OptimalControlProblem
from sms_ocp.utils import normalize_grid_points


@dataclass(frozen=True)
class DecisionBlock:
    """Describe one contiguous matrix block in the decision vector."""

    name: str
    shape: tuple[int, int]
    start: int
    stop: int


@dataclass(frozen=True)
class DecisionLayout:
    """Describe the column-major flattening of all decision-variable blocks."""

    blocks: tuple[DecisionBlock, ...]

    @property
    def size(self) -> int:
        """Return the total number of scalar decision variables."""
        return self.blocks[-1].stop if self.blocks else 0

    def block(self, name: str) -> DecisionBlock:
        """Return a decision block by name."""
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f"Unknown decision block: {name!r}.")

    def has_block(self, name: str) -> bool:
        """Return whether the layout contains a named decision block."""
        return any(block.name == name for block in self.blocks)

    def extract(
        self,
        decision_vector: ArrayLike,
        name: str,
    ) -> np.ndarray:
        """Recover one matrix block from a numerical decision vector."""
        vector = np.asarray(decision_vector, dtype=float).reshape(-1)
        if vector.size != self.size:
            raise ValueError(
                f"decision_vector must contain {self.size} entries."
            )

        block = self.block(name)
        return vector[block.start : block.stop].reshape(
            block.shape,
            order="F",
        )


def build_multiple_shooting_decision_layout(
    ocp: OptimalControlProblem,
    num_intervals: int,
) -> DecisionLayout:
    """Build the type-major layout ``vec(X), vec(U), p, [T]``."""
    if (
        not isinstance(num_intervals, int)
        or isinstance(num_intervals, bool)
        or num_intervals <= 0
    ):
        raise ValueError("num_intervals must be a positive integer.")

    model = ocp.model
    specifications = [
        ("x", (model.nx, num_intervals + 1)),
    ]
    if model.nu > 0:
        specifications.append(
            ("u", (model.nu, num_intervals))
        )
    if model.np > 0:
        specifications.append(
            ("p", (model.np, 1))
        )
    if ocp.tf_bounds[0] != ocp.tf_bounds[1]:
        specifications.append(
            ("T", (1, 1))
        )

    blocks: list[DecisionBlock] = []
    offset = 0
    for name, shape in specifications:
        size = shape[0] * shape[1]
        blocks.append(
            DecisionBlock(
                name=name,
                shape=shape,
                start=offset,
                stop=offset + size,
            )
        )
        offset += size

    return DecisionLayout(blocks=tuple(blocks))


def pack_initial_guess(
    ocp: OptimalControlProblem,
    shooting_grid: ArrayLike,
    *,
    states: ArrayLike,
    controls: ArrayLike | None = None,
    parameters: ArrayLike | None = None,
    terminal_time: float | ArrayLike | None = None,
) -> np.ndarray:
    """Pack multiple-shooting initial values into one decision vector.

    Two-dimensional state and control inputs must match their corresponding
    decision-block shapes. One-dimensional inputs are interpreted in
    column-major order when their number of entries is correct.
    """
    grid = normalize_grid_points(shooting_grid)
    layout = build_multiple_shooting_decision_layout(
        ocp,
        grid.size - 1,
    )
    supplied_values = {
        "x": states,
        "u": controls,
        "p": parameters,
        "T": terminal_time,
    }
    guess = np.empty(layout.size)

    for block in layout.blocks:
        values = supplied_values[block.name]
        if values is None:
            raise ValueError(
                f"{block.name!r} initial values are required for this OCP."
            )

        array = np.asarray(values, dtype=float)
        expected_size = block.shape[0] * block.shape[1]
        if array.size != expected_size:
            raise ValueError(
                f"{block.name!r} initial values must contain "
                f"{expected_size} entries, got {array.size}."
            )
        if array.ndim > 1 and array.shape != block.shape:
            raise ValueError(
                f"{block.name!r} initial values must have shape "
                f"{block.shape}, got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"{block.name!r} initial values must be finite."
            )

        guess[block.start : block.stop] = array.reshape(-1, order="F")

    for block_name, values in supplied_values.items():
        if values is not None and not layout.has_block(block_name):
            raise ValueError(
                f"{block_name!r} initial values were supplied, "
                "but this OCP has no corresponding decision block."
            )

    return guess
