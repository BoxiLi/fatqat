"""Shared execution contracts for objects accepted by matrix simulators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .program import Program
from .resource_layout import ResourceLayout


@runtime_checkable
class ExecutableProgram(Protocol):
    """A simulator program bundled with its required device layout."""

    @property
    def program(self) -> Program:
        """Return the gate-level program to execute."""
        raise NotImplementedError

    @property
    def resource_layout(self) -> ResourceLayout:
        """Return the device mapping required by ``program``."""
        raise NotImplementedError
