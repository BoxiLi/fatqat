"""State owned by one active engine evolution or compiled shot.

System configuration, execution policy, RNGs, and numerical caches live outside
these records. Compiled multi-shot kernels own their state locally per shot.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(slots=True)
class QuantumState:
    """The authoritative numerical buffer and its concrete representation."""

    buffer: np.ndarray
    representation: str


@dataclass(slots=True)
class ClassicalState:
    """Reported classical digits and occupied subsystem indices for one shot."""

    clbits: list[int]
    occupied: set[int]


@dataclass(slots=True)
class EvolutionState:
    """Quantum state with optional classical state for dynamic execution."""

    quantum: QuantumState
    classical: ClassicalState | None = None


class CompiledEvolutionState(NamedTuple):
    """Quantum and classical arrays owned by one compiled statevector shot.

    This path assumes full occupancy. Replacing the record swaps array
    references without copying data or discarding classical digits.
    """

    quantum: np.ndarray
    classical: np.ndarray
