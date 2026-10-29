"""Matrix simulation of linear-optical programs on truncated modes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Mapping

import numpy as np

from .._backends.backend_utils import _normalize_config
from ..errors import BackendValidationError
from ..operations import Measurement
from ..parameters import Parameter
from ..program import Program
from ..registers import (
    ClassicalRegister,
    QuantumRegister,
    Register,
    RegisterRef,
    RegisterView,
)
from ..resource_layout import ResourceLayout
from ._engine.state import BasisState, EvolutionState
from ._execution_contract import _SimulationConfig
from .simulator import Simulator, _PreparedExecution

__all__ = ["LOSimulator"]

_CUTOFF_ERROR = "cutoff must be None or a positive int"


@dataclass(frozen=True)
class _LOSimulationConfig(_SimulationConfig):
    """Matrix-simulator controls plus the per-run photon cutoff."""

    cutoff: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cutoff is not None and (
            type(self.cutoff) is not int or self.cutoff < 1
        ):
            raise BackendValidationError(f"{_CUTOFF_ERROR}, got {self.cutoff!r}")


class LOSimulator(Simulator):
    """Run linear-optical programs on modes truncated to a per-run cutoff.

    A `fatqat.ModeRegister` has no fixed dimension. For each run, this
    backend gives every mode ``cutoff + 1`` levels (``0`` to ``cutoff``
    photons) and every `fatqat.CountRegister` slot the same range, then runs
    the program exactly as `Simulator` would. Methods, runtimes, noise,
    results, and ``run_sweep`` follow `Simulator`. Programs without a
    `fatqat.ModeRegister` or `fatqat.CountRegister` run unchanged.

    ``simulation_config`` accepts every `Simulator` key plus:

    - ``"cutoff"`` (``int | None``, default ``None``): Maximum photon number
      per mode, a positive integer. ``None`` infers it from ``initial_state``.
      For a `BasisState` it is the total photon number in the modes (at least
      1), so passive gates are exact. For a dense array its length must equal
      ``(cutoff + 1)`` to the power of the number of modes, times the
      dimensions of any other quantum registers. Without an ``initial_state``
      the cutoff must be given.

    Truncation is an approximation: `fatqat.operations.BeamSplitter` is exact
    only while the two modes it mixes hold at most ``cutoff`` photons in
    total. Choose a cutoff at least the largest photon number the program can
    bring into one pair of modes. The backend warns when an explicit cutoff is
    below the photon total of a `BasisState`; it does not inspect dense
    arrays. A mode level above the cutoff does not fit and raises.

    Noise selectors that name a `fatqat.ModeRegister` reference are not
    supported; select those modes by device label instead.

    Examples:
        Two photons meeting on a balanced beam splitter leave together
        (Hong-Ou-Mandel interference):

        >>> import math
        >>> import numpy as np
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> modes = fq.ModeRegister(2)
        >>> program = fq.Program([modes])
        >>> program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
        >>> one_one = fq.simulator.BasisState({modes[0]: 1, modes[1]: 1})
        >>> state = (  # the inferred cutoff is 2, one level per photon
        ...     fq.simulator.LOSimulator("SV")
        ...     .run(program, initial_state=one_one)
        ...     .result()
        ...     .get_statevector()
        ... )
        >>> np.round(np.abs(state) ** 2, 3).reshape(3, 3)
        array([[0. , 0. , 0.5],
               [0. , 0. , 0. ],
               [0.5, 0. , 0. ]])
    """

    _simulation_config_cls = _LOSimulationConfig

    def _prepare_execution(
        self,
        program: Program,
        *,
        resource_layout: ResourceLayout | None,
        initial_state: Any,
        simulation_config: dict[str, Any] | None,
        parameter_values: Mapping[Parameter, np.ndarray] | None = None,
    ) -> _PreparedExecution:
        """Resolve mode dimensions, then prepare the run like `Simulator`."""
        registers = program.quantum_registers + program.classical_registers
        if any(register.dim is None for register in registers):
            simulation = _normalize_config(
                simulation_config,
                self._simulation_config_cls,
                "simulation_config",
                backend_name=type(self).__name__,
            )
            quantum = (
                initial_state.quantum
                if isinstance(initial_state, EvolutionState)
                else initial_state
            )
            cutoff = simulation.cutoff
            if cutoff is None:
                cutoff = _infer_cutoff(program, quantum)
            elif isinstance(quantum, BasisState):
                # Only a basis state gives the photon number directly; dense
                # arrays are not inspected (they may not be NumPy arrays).
                photons = _mode_photons(quantum)
                if photons > cutoff:
                    warnings.warn(
                        f"cutoff {cutoff} is below the {photons} photons in "
                        "initial_state; BeamSplitter results may be truncated",
                        stacklevel=3,
                    )
            self._reject_mode_noise_selectors(program)
            # Matrix rules, classical allocation and count formatting all read
            # `register.dim`, so the run uses a copy with concrete registers.
            # This keeps per-run sizing here instead of adding dimension hooks
            # to the generic Simulator, and keeps the cutoff out of the program.
            program, refs = _resolve_mode_dimensions(program, cutoff + 1)
            # The copy has new refs; translate a caller's layout onto them.
            if resource_layout is not None:
                resource_layout = ResourceLayout(
                    {
                        refs.get(ref, ref): resource_layout.device_label(ref)
                        for ref in resource_layout.refs
                    }
                )
            # A basis state names refs too; translate it the same way.
            if isinstance(quantum, BasisState):
                quantum = BasisState(
                    {refs.get(ref, ref): level for ref, level in quantum.levels.items()}
                )
                initial_state = (
                    replace(initial_state, quantum=quantum)
                    if isinstance(initial_state, EvolutionState)
                    else quantum
                )
        return super()._prepare_execution(
            program,
            resource_layout=resource_layout,
            initial_state=initial_state,
            simulation_config=simulation_config,
            parameter_values=parameter_values,
        )

    def _reject_mode_noise_selectors(self, program: Program) -> None:
        """Reject noise that names a mode ref, which per-run sizing replaces.

        The rebuilt program has new refs, so a ref selector would silently
        stop matching. Device-label selectors survive the rebuild because the
        default layout keeps declaration-order labels. Translating ref
        selectors too is possible but not needed yet.
        """
        modes = {
            id(register)
            for register in program.quantum_registers
            if register.dim is None
        }
        selectors = [
            registration.selector
            for registration in self._noise_model._noise_registrations
        ]
        selectors.extend(
            (selector,)
            for selector, _declaration in self._noise_model._readout_registrations
        )
        for selector in selectors:
            if selector is not None and any(
                isinstance(ref, RegisterRef) and id(ref.register) in modes
                for ref in selector
            ):
                raise BackendValidationError(
                    "LOSimulator does not support noise selectors that name a "
                    "ModeRegister reference; select modes by device label"
                )


def _mode_photons(basis: BasisState) -> int:
    """Return the total photon number a basis state puts in the modes."""
    return sum(level for ref, level in basis.levels.items() if ref.register.dim is None)


def _infer_cutoff(program: Program, quantum: Any) -> int:
    """Infer the cutoff from a basis state or a dense initial state's length."""
    if isinstance(quantum, BasisState):
        # Passive gates never add photons, so no mode can exceed the total.
        return max(1, _mode_photons(quantum))
    num_modes = sum(
        register.size for register in program.quantum_registers if register.dim is None
    )
    other = prod(
        register.dim**register.size
        for register in program.quantum_registers
        if register.dim is not None
    )
    shape = np.shape(quantum) if quantum is not None else ()
    if shape:
        per_modes, remainder = divmod(shape[0], other)
        dim = round(per_modes ** (1 / num_modes)) if num_modes else 0
        if remainder == 0 and dim >= 2 and dim**num_modes == per_modes:
            return dim - 1
        raise BackendValidationError(
            f"cannot infer cutoff from an initial_state of length {shape[0]} "
            f"for {num_modes} mode(s); pass simulation_config['cutoff']"
        )
    raise BackendValidationError(
        "a ModeRegister program needs simulation_config['cutoff'] or an "
        "initial_state to infer it from"
    )


def _resolve_mode_dimensions(
    program: Program, dim: int
) -> tuple[Program, dict[RegisterRef, RegisterRef]]:
    """Rebuild a program with every dimensionless register sized to ``dim``.

    Returns the rebuilt program and the map from each replaced ref to its
    replacement. Rebuilding re-runs the target checks that were deferred while
    the dimension was unknown.
    """
    replaced: dict[int, Register] = {}

    def resolve(register: Register, kind: type[Register]) -> Register:
        if register.dim is not None:
            return register
        replaced[id(register)] = kind(
            register.size, name=register.name, metadata=register.metadata, dim=dim
        )
        return replaced[id(register)]

    resolved = Program(
        [resolve(register, QuantumRegister) for register in program.quantum_registers],
        [
            resolve(register, ClassicalRegister)
            for register in program.classical_registers
        ],
        metadata=program.metadata,
    )

    def ref(target: RegisterRef) -> RegisterRef:
        register = replaced.get(id(target.register))
        return target if register is None else register[target.index]

    def target(item: RegisterRef | RegisterView) -> RegisterRef | RegisterView:
        if isinstance(item, RegisterView):
            register = replaced.get(id(item.register), item.register)
            return replace(item, register=register)
        return ref(item)

    for instruction in program._instructions:
        if isinstance(instruction, Measurement):
            resolved.measure(
                tuple(ref(t) for t in instruction.targets),
                tuple(ref(o) for o in instruction.outputs),
            )
            continue
        condition = instruction.condition
        resolved.add(
            instruction.operation,
            tuple(target(t) for t in instruction.targets),
            condition=(
                None
                if condition is None
                else tuple((ref(slot), value) for slot, value in condition)
            ),
        )
    refs = {
        register[index]: replaced[id(register)][index]
        for register in program.quantum_registers
        if id(register) in replaced
        for index in range(register.size)
    }
    return resolved, refs
