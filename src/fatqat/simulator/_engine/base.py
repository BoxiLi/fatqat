from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..._backends.engine_contract import RawResult
from ..._backends.steps import ApplyMatrixStep, ResolvedStep
from .._execution_contract import (
    _EngineCapabilities,
    _KernelCapabilities,
    _QuantumCapabilities,
    _TrajectoryCapabilities,
    _ExecutionContext as ExecutionContext,
    _ExecutionPolicy as ExecutionPolicy,
)
from .state import EvolutionState, QuantumState


def _shot_seed_sequences(
    seed: int | None, n_iters: int
) -> list[np.random.SeedSequence]:
    """Spawn stable, ordered child streams for sampled shots."""
    return np.random.SeedSequence(seed).spawn(n_iters)


class MatrixEngine(ABC):
    """
    Abstract base class and interface contract for all engines.
    """

    # Concrete engines declare all support; numeric/execution helpers may be shared.
    _quantum_capabilities: _QuantumCapabilities
    _trajectory_capabilities: _TrajectoryCapabilities
    _kernel_capabilities: _KernelCapabilities

    def __init__(
        self,
        name: str,
        *,
        state_semantics: Literal["sv", "dm"],
    ):
        self.name = name
        self.state_semantics = state_semantics

        self._evolution_state: EvolutionState | None = None
        self._dims: tuple[int, ...] = ()
        self._reversed_dims: tuple[int, ...] = ()
        self._n_clbits = 0

    @property
    def _state(self) -> np.ndarray | None:
        """Forward existing kernel access to the single owned quantum buffer."""
        if self._evolution_state is None:
            return None
        return self._evolution_state.quantum.buffer

    @_state.setter
    def _state(self, value: np.ndarray | None) -> None:
        """Replace the quantum buffer without losing current classical state."""
        if value is None:
            self._evolution_state = None
        elif self._evolution_state is None:
            self._evolution_state = EvolutionState(
                QuantumState(value, self._state_field)
            )
        else:
            self._evolution_state.quantum.buffer = value

    @property
    def state(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("MatrixEngine state has not been initialized.")
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        self._state = value

    @property
    def n_subsystems(self) -> int:
        return len(self._dims)

    @property
    def _state_field(self) -> str:
        """Use the declared representation for evolving state and results."""
        return self._quantum_capabilities.representation

    @property
    def capabilities(self) -> _EngineCapabilities:
        """Return supported quantum, classical, and numerical execution."""
        return _EngineCapabilities(
            quantum=self._quantum_capabilities,
            trajectory=self._trajectory_capabilities,
            kernels=self._kernel_capabilities,
        )

    def compiled_multi_shot_compatible(self, plan: Sequence[ResolvedStep]) -> bool:
        """Whether this engine can own the complete per-shot outer loop."""
        return False

    def configure_system(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        """Configure dimensions without allocating an evolving state."""
        self._set_dims(system_dims)
        self._n_clbits = int(n_clbits)
        self._state = None

    @abstractmethod
    def initialize(
        self,
        system_dims: Sequence[int],
        n_clbits: int = 0,
        *,
        initial_state: np.ndarray | None = None,
    ) -> None:
        """Configure the system and allocate a fresh owned evolving state."""

    def _set_dims(self, system_dims: Sequence[int]) -> None:
        """Set ``_dims`` and its cached reverse together, so they never drift apart."""
        self._dims = tuple(int(d) for d in system_dims)
        self._reversed_dims = tuple(reversed(self._dims))

    @abstractmethod
    def materialize_execution(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        deferred_measurements: tuple[tuple[int, int], ...],
        policy: ExecutionPolicy,
    ) -> Any:
        """Build the engine-owned immutable payload for this run."""

    @abstractmethod
    def execute_local(
        self,
        context: ExecutionContext,
        payload: Any,
        policy: ExecutionPolicy,
    ) -> RawResult:
        """Execute a materialized payload locally without dispatching."""

    def execute_shot_batch(
        self,
        context: ExecutionContext,
        payload: Any,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
    ) -> list[tuple[int, ...]]:
        """Execute one ordered shot batch on engines that support it."""
        raise NotImplementedError

    @abstractmethod
    def measure_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> tuple[int, ...]: ...

    def measure_subsystem(self, index: int, rng: np.random.Generator) -> int:
        """
        Measure a single subsystem and return the result.
        """
        return self.measure_subsystems([index], rng)[0]

    @abstractmethod
    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None: ...

    def reset_subsystem(self, index: int, rng: np.random.Generator) -> None:
        """
        Reset a single subsystem to the |0> state.
        """
        self.reset_subsystems([index], rng)

    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """Return the computational-basis probability distribution of the state."""

    @abstractmethod
    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Sample one outcome, project the internal state, return the flat index."""

    @abstractmethod
    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply a single matrix step to the internal state in place."""

    def export_state(self) -> np.ndarray:
        """
        Export the current state of the engine as a numpy array.
        """
        return self.state.copy()

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample flat basis-state indices from the current state.
        """
        return rng.choice(self.state.shape[0], size=shots, p=self.probabilities())
