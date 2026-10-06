"""Private semantic and execution records for matrix simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._backends.engine_contract import _ResultRequest

ExecutionShape = Literal["operator", "single_pass", "per_shot"]


@dataclass(frozen=True, slots=True)
class _PlanFacts:
    """Runtime-independent semantic facts derived from one lowered plan."""

    execution_shape: ExecutionShape
    deferred_measurements: tuple[tuple[int, int], ...]
    written_clbits: frozenset[int]
    stochastic_final_state: bool
    has_measurement: bool
    has_reset: bool
    has_channel: bool
    has_condition: bool


@dataclass(frozen=True, slots=True)
class _QuantumCapabilities:
    """Representation semantics, independent of numerical runtime."""

    representation: Literal["statevector", "density_matrix", "unitary", "superop"]
    supports_nonunitary: bool
    nonunitary_is_stochastic: bool

    @property
    def is_operator(self) -> bool:
        """Whether evolution computes a map rather than a state under it."""
        return self.representation in {"unitary", "superop"}


@dataclass(frozen=True, slots=True)
class _TrajectoryCapabilities:
    """Classical components supported by the complete trajectory executor."""

    classical_register: bool
    occupancy: bool


@dataclass(frozen=True, slots=True)
class _KernelCapabilities:
    """Numerical controls, separate from supported trajectory state."""

    supports_kernel_threads: bool
    thread_capacity: int
    supports_fusion: bool


@dataclass(frozen=True, slots=True)
class _EngineCapabilities:
    """Engine-owned static support; no evolving storage or plan allocation."""

    quantum: _QuantumCapabilities
    trajectory: _TrajectoryCapabilities | None
    kernels: _KernelCapabilities

    @property
    def supports_classical_register(self) -> bool:
        """Whether measurement reports and conditions have a register."""
        return self.trajectory is not None and self.trajectory.classical_register

    @property
    def supports_occupancy(self) -> bool:
        """Whether an execution may own explicit carrier occupancy."""
        return self.trajectory is not None and self.trajectory.occupancy


@dataclass(frozen=True, slots=True)
class _ExecutionPolicy:
    """Final implementation and routing decisions for one execution."""

    shot_strategy: Literal["none", "serial", "threads", "processes"]
    kernel_strategy: Literal["serial", "adaptive", "threads"]
    worker_limit: int | None
    fusion: bool
    use_compiled_multi_shot_kernel: bool = False


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    """Semantic and numerical values executed under a resolved policy."""

    execution_shape: ExecutionShape
    request: _ResultRequest
    system_dims: tuple[int, ...]
    n_clbits: int
    shots: int
    seed: int | None
    initial_state: np.ndarray | None
    initial_occupied: frozenset[int] | None
