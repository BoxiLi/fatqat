"""Editable, hardware-independent input for the FatQat compiler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

from . import operations as ops
from .operations import Operation
from .parameters import Parameter, ParameterVector
from .program import Program
from .registers import ClassicalRegister, QuantumRegister, RegisterRef

__all__ = ["LogicalProgram"]

QuantumRegisters = int | list[QuantumRegister] | tuple[QuantumRegister, ...]
ClassicalRegisters = int | list[ClassicalRegister] | tuple[ClassicalRegister, ...]
Qubit = int | RegisterRef
Clbit = int | RegisterRef


class LogicalProgram:  # pylint: disable=too-many-public-methods
    """Build a gate-level program that can be compiled to a hardware family.

    The object is mutable for convenient Python construction. Compilation
    freezes it into the compiler's immutable logical IR before any lowering
    runs.
    """

    IR_ID: ClassVar[str] = "gate.logical.source.v1"

    def __init__(
        self,
        quantum_registers: QuantumRegisters,
        classical_registers: ClassicalRegisters = 0,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._program = Program(
            quantum_registers,
            classical_registers,
            metadata=metadata,
        )

    @property
    def quantum_registers(self) -> tuple[QuantumRegister, ...]:
        """Return quantum registers in declaration order."""
        return self._program.quantum_registers

    @property
    def classical_registers(self) -> tuple[ClassicalRegister, ...]:
        """Return classical registers in declaration order."""
        return self._program.classical_registers

    @property
    def metadata(self) -> dict[str, Any]:
        """Return mutable application metadata."""
        return self._program.metadata

    def add(self, operation: Operation, *qubits: Qubit | tuple[Qubit, ...]) -> Self:
        """Append an operation using separate or grouped qubit operands."""
        targets = qubits[0] if len(qubits) == 1 else qubits
        self._program.add(operation, targets)
        return self

    def i(self, qubit: Qubit) -> Self:
        return self.add(ops.I, qubit)

    def h(self, qubit: Qubit) -> Self:
        return self.add(ops.H, qubit)

    def x(self, qubit: Qubit) -> Self:
        return self.add(ops.X, qubit)

    def y(self, qubit: Qubit) -> Self:
        return self.add(ops.Y, qubit)

    def z(self, qubit: Qubit) -> Self:
        return self.add(ops.Z, qubit)

    def s(self, qubit: Qubit) -> Self:
        return self.add(ops.S, qubit)

    def sdg(self, qubit: Qubit) -> Self:
        return self.add(ops.Sdg, qubit)

    def t(self, qubit: Qubit) -> Self:
        return self.add(ops.T, qubit)

    def tdg(self, qubit: Qubit) -> Self:
        return self.add(ops.Tdg, qubit)

    def sx(self, qubit: Qubit) -> Self:
        return self.add(ops.SX, qubit)

    def rx(self, theta: float | Parameter, qubit: Qubit) -> Self:
        return self.add(ops.RX(theta), qubit)

    def ry(self, theta: float | Parameter, qubit: Qubit) -> Self:
        return self.add(ops.RY(theta), qubit)

    def rz(self, theta: float | Parameter, qubit: Qubit) -> Self:
        return self.add(ops.RZ(theta), qubit)

    def phase(self, theta: float | Parameter, qubit: Qubit) -> Self:
        return self.add(ops.Phase(theta), qubit)

    def cx(self, control: Qubit, target: Qubit) -> Self:
        return self.add(ops.CX, control, target)

    def cz(self, first: Qubit, second: Qubit) -> Self:
        return self.add(ops.CZ, first, second)

    def swap(self, first: Qubit, second: Qubit) -> Self:
        return self.add(ops.Swap, first, second)

    def reset(self, qubit: Qubit) -> Self:
        return self.add(ops.Reset, qubit)

    def measure(
        self,
        qubits: Qubit | tuple[Qubit, ...],
        clbits: Clbit | tuple[Clbit, ...],
    ) -> Self:
        """Measure qubits into classical slots."""
        self._program.measure(qubits, clbits)
        return self

    def measure_all(self) -> Self:
        """Measure every declared qubit into the corresponding classical slot."""
        self._program.measure_all()
        return self

    def copy(self) -> LogicalProgram:
        """Return an independently editable copy."""
        return type(self)._from_program(self._program.copy())

    def assign_parameters(
        self,
        values: Mapping[Parameter | ParameterVector, object],
    ) -> LogicalProgram:
        """Return a copy with selected symbolic parameters bound."""
        return type(self)._from_program(self._program.assign_parameters(values))

    @classmethod
    def _from_program(cls, program: Program) -> LogicalProgram:
        copied = cls.__new__(cls)
        copied._program = program
        return copied

    def draw(self, renderer: str = "matplotlib", **kwargs: Any) -> Any:
        """Draw the logical circuit using the existing Program renderer."""
        return self._program.draw(renderer, **kwargs)

    def dag(self):
        """Return the hardware-independent instruction DAG."""
        return self._program.dag()
