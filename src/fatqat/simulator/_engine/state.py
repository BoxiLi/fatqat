"""Mutable state owned by one active reference-path engine evolution.

System configuration, execution policy, RNGs, and numerical caches live outside
these records. Compiled multi-shot kernels retain their own local arrays.
"""

from dataclasses import dataclass

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
