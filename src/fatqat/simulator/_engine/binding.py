"""Bind sweep parameter values into executable matrix plans.

A sweep lowers once. Parameter-holding gates stay `ApplyParametricMatrixStep`
values carrying one value column per `Parameter` field, and `_bind_rows`
turns the plan into one engine-ready step tuple per row. Every other step is
shared by reference across rows.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from math import prod
from typing import Any

import numpy as np

from ..._backends.steps import (
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    ResolvedStep,
)
from ...errors import MatrixImplementationError
from ...implementation.matrix_map import (
    _BatchedBuilder,
    _build_matrix,
    _validate_matrix_shape,
)

# Bytes of batched gate matrices held at once during binding. A chunk takes as
# many rows as fit, and at least one, so a high-dimensional gate never holds
# more matrices at once than per-row binding would.
_REPLAY_CHUNK_BYTES = 64 * 2**20


def _bind_step(
    step: ApplyParametricMatrixStep,
    row: int,
    target_dims: tuple[int, ...],
) -> ApplyMatrixStep:
    """Bind one row into one step through the step's own matrix rule.

    `Program.assign_parameters` re-runs ``validate_targets`` when it rebuilds
    an applied operation, so a bound row gives a parameter-dependent target
    check the same chance to reject the substituted value.
    """
    operation = replace(
        step.operation, **{name: column[row] for name, column in step.values}
    )
    operation.validate_targets(step.targets)
    matrix = _build_matrix(
        step.implementation,
        operation,
        targets=step.targets,
        target_dims=target_dims,
    )
    return ApplyMatrixStep(
        matrix=matrix,
        target_indices=step.target_indices,
        condition=step.condition,
        kernel_key=step.implementation._kernel_key(
            step.operation, targets=step.targets
        ),
    )


def _build_matrix_batch(
    step: ApplyParametricMatrixStep,
    builder: _BatchedBuilder,
    target_dims: tuple[int, ...],
    rows: slice,
) -> np.ndarray:
    """Build one step's read-only ``(rows, d, d)`` matrix stack.

    Value columns are evaluated in float64, the package's working precision;
    a float64 column is sliced without copying. Builder failures raise
    `MatrixImplementationError` and a wrong-shaped stack raises
    `BackendValidationError`, as per-row binding would.
    """
    template = step.operation
    try:
        columns = {
            name: np.asarray(column[rows], dtype=np.float64)
            for name, column in step.values
        }

        def field(name: str) -> Any:
            return columns[name] if name in columns else getattr(template, name)

        stack = builder(field, step.targets)
    except Exception as exc:
        raise MatrixImplementationError(
            f"implementation for {type(template).__name__} raised: {exc}"
        ) from exc
    _validate_matrix_shape(stack.shape[1:], template, target_dims)
    stack.flags.writeable = False
    return stack


def _bind_rows(
    plan: Sequence[ResolvedStep],
    n_rows: int,
    system_dims: Sequence[int],
) -> Iterator[tuple[ResolvedStep, ...]]:
    """Yield the bound, engine-ready plan for each sweep row in order.

    Built-in parametric gates get their matrices for a chunk of rows at once
    from their batched rules, so a failure there raises when its chunk is
    reached. Every other parametric step calls its rule per row and raises
    when its row is reached, after the earlier rows were yielded.
    """
    if any(
        isinstance(step, ApplyParametricMatrixStep) and not step.values for step in plan
    ):
        raise RuntimeError("parameter values must be injected before binding")
    builders = [
        (
            step.implementation._batch_builder(step.operation)
            if isinstance(step, ApplyParametricMatrixStep)
            else None
        )
        for step in plan
    ]
    dimensions = [
        (
            tuple(system_dims[i] for i in step.target_indices)
            if isinstance(step, ApplyParametricMatrixStep)
            else ()
        )
        for step in plan
    ]
    row_bytes = sum(
        prod(target_dims) ** 2 * np.dtype(complex).itemsize
        for builder, target_dims in zip(builders, dimensions)
        if builder is not None
    )
    chunk_rows = max(1, _REPLAY_CHUNK_BYTES // row_bytes) if row_bytes else 1
    for start in range(0, n_rows, chunk_rows):
        rows = slice(start, min(start + chunk_rows, n_rows))
        stacks = [
            (
                None
                if builder is None
                else _build_matrix_batch(step, builder, target_dims, rows)
            )
            for step, builder, target_dims in zip(plan, builders, dimensions)
        ]
        for index, row in enumerate(range(rows.start, rows.stop)):
            bound: list[ResolvedStep] = []
            for step, stack, target_dims in zip(plan, stacks, dimensions):
                if not isinstance(step, ApplyParametricMatrixStep):
                    bound.append(step)
                elif stack is None:
                    bound.append(_bind_step(step, row, target_dims))
                else:
                    bound.append(
                        ApplyMatrixStep(
                            matrix=stack[index],
                            target_indices=step.target_indices,
                            condition=step.condition,
                            kernel_key=step.implementation._kernel_key(
                                step.operation, targets=step.targets
                            ),
                        )
                    )
            yield tuple(bound)
