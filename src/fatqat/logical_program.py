"""The under-development logical program frontend."""

from __future__ import annotations

from typing import ClassVar

from .program import Program

__all__ = ["LogicalProgram"]


class LogicalProgram(Program):
    """Build a circuit using the same constructor and operations as Program.

    Under development: this class does not yet enforce a formal logical-program
    contract. It currently accepts the same operations as Program, including
    resource operations. Restricting it to logical operations requires further
    work. Static compiler and target restrictions still apply when compiling.

    Register counts, register collections, metadata, and authoring methods are
    inherited from Program. The inherited copy() and assign_parameters()
    methods return Program instances, which the compiler also accepts.

    Attributes:
        IR_ID: Compiler identity for this editable source representation.
    """

    IR_ID: ClassVar[str] = "gate.logical.source.v1"
