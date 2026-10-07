"""State owned by one active engine evolution or compiled shot.

System configuration, execution policy, RNGs, and numerical caches live outside
these records. Compiled multi-shot kernels own their state locally per shot.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


@dataclass(slots=True)
class QuantumState:
    """The authoritative numerical buffer and its concrete representation."""

    buffer: np.ndarray
    representation: str


@dataclass(slots=True)
class ClassicalState:
    """Classical container whose components are allocated only when needed."""

    # Unallocated register storage leaves unwritten report digits at zero.
    clbits: list[int] | None = None
    # None means implicitly full occupancy; an empty set means no carriers.
    occupied: set[int] | None = None


@dataclass(slots=True)
class EvolutionState:
    """Quantum and classical containers owned by one active evolution.

    An empty classical container allocates no component buffers and does not
    indicate capability support. Every evolution receives its own container.
    """

    quantum: QuantumState
    classical: ClassicalState = field(default_factory=ClassicalState)


class NumbaEvolutionState(NamedTuple):
    """Quantum and classical arrays owned by one compiled statevector shot.

    This path assumes full occupancy. Replacing the record swaps array
    references without copying data or discarding classical digits.
    """

    quantum: np.ndarray
    classical: np.ndarray
