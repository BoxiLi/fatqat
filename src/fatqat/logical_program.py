"""Device-independent program authoring for compilation and simulation."""

from __future__ import annotations

from typing import ClassVar

from . import operations as ops
from .operations import Operation
from .program import ConditionInput, Program
from .registers import RegisterRef, RegisterView

__all__ = ["LogicalProgram"]


_LOGICAL_OPERATION_TYPES = frozenset(
    {
        type(ops.I),
        type(ops.H),
        type(ops.S),
        type(ops.Sdg),
        type(ops.SX),
        type(ops.T),
        type(ops.Tdg),
        type(ops.X),
        type(ops.Y),
        type(ops.Z),
        type(ops.CX),
        type(ops.CZ),
        type(ops.Swap),
        type(ops.CY),
        type(ops.CS),
        type(ops.iSwap),
        type(ops.CCX),
        type(ops.CSwap),
        ops.RX,
        ops.RY,
        ops.RZ,
        ops.Phase,
        ops.U,
        ops.U1,
        ops.U2,
        ops.U3,
        ops.CPhase,
        type(ops.Reset),
        type(ops.Barrier),
        ops.Shift,
        ops.Clock,
        type(ops.Sum),
        ops.SwapLevels,
        type(ops.Fourier),
        type(ops.InverseFourier),
        ops.SubspaceRX,
        ops.SubspaceRY,
        ops.SubspaceRZ,
        ops.CClock,
    }
)


class LogicalProgram(Program):
    """Build a device-independent circuit for compilation or simulation.

    Use Program's constructor, add(), and measurement interface. Authoring
    methods mutate in place and return None. Built-in circuit gates, including
    qudit gates, reset, and barriers are accepted; device operations and custom
    Operation subclasses are rejected. Use measure() for measurements.

    Conditions remain valid for direct simulation. Static compiler and target
    restrictions apply when compiling. The inherited copy() and
    assign_parameters() methods preserve LogicalProgram.

    Attributes:
        IR_ID: Compiler identity for this editable source representation.
    """

    IR_ID: ClassVar[str] = "gate.logical.source.v1"

    def add(
        self,
        op: Operation,
        targets: (
            int
            | RegisterRef
            | RegisterView
            | tuple[int | RegisterRef | RegisterView, ...]
        ) = (),
        *,
        condition: ConditionInput = None,
    ) -> None:
        """Append a built-in device-independent operation in place.

        Arguments and target/condition validation follow Program.add.
        Device operations and custom Operation subclasses raise ValueError
        before the program changes. Measurements use measure() instead.

        Args:
            op: A built-in device-independent Operation instance.
            targets: Scalar register operands or compatible register views,
                using the same forms as Program.add.
            condition: Optional classical condition, default None; uses the
                same register/literal pairs and conjunction rules as Program.add.

        Returns:
            None.

        Raises:
            ValueError: If op is not a built-in logical operation, or inherited
                target or condition validation fails.
            TypeError: If an argument has an unsupported type.
            IndexError: If an integer operand is outside its register.
        """
        if isinstance(op, Operation) and type(op) not in _LOGICAL_OPERATION_TYPES:
            raise ValueError(f"{op.name} is not a logical operation")
        super().add(op, targets, condition=condition)

    def _new_copy(self) -> LogicalProgram:
        return LogicalProgram.__new__(LogicalProgram)
