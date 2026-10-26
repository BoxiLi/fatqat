"""Independent custom loss with per-shot tracking through public circuit APIs.

Run with this checkout installed: python examples/custom_loss.py
"""

from dataclasses import dataclass
from math import prod

import numpy as np

import fatqat as fq
import fatqat.operations as ops
from fatqat.implementation import (
    CustomOperationImplementation,
)
from fatqat.simulator import (
    AtomArraySimulator,
    ClassicalState,
    EvolutionState,
    ExecutionContext,
)


@dataclass(frozen=True)
class SetLossProbability(ops.CustomOperation):
    """Set the probability used by later loss operations in this shot."""

    name = "SetLossProbability"
    num_subsystems = 0
    probability: float


class MyLoss(ops.CustomOperation):
    """An independent operation description, unrelated to built-in Loss."""

    name = "MyLoss"
    num_subsystems = None


@dataclass(frozen=True)
class ReportLossCount(ops.CustomOperation):
    """Write the extra-state counter to a public classical slot."""

    name = "ReportLossCount"
    num_subsystems = 0
    destination: int


class SetLossProbabilityImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(operation, state, context):
        probability = operation.probability
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be finite and in [0, 1]")
        state.classical.extra["loss_probability"] = probability
        return state


def discard_subsystem(quantum, target, system_dims, rng):
    """Discard one subsystem and put it in ground using only NumPy.

    Statevectors sample a conditional pure state. Density matrices sum over
    the discarded subsystem. Indices use the simulator's least-significant-first
    subsystem order.
    """
    stride = prod(system_dims[:target])
    dimension = system_dims[target]
    basis = np.arange(quantum.shape[0])
    digits = (basis // stride) % dimension
    ground = basis[digits == 0]
    result = np.zeros_like(quantum)
    if quantum.ndim == 1:
        probabilities = np.abs(quantum) ** 2
        sampled = rng.choice(len(basis), p=probabilities / probabilities.sum())
        selected = ground + digits[sampled] * stride
        branch = quantum[selected]
        result[ground] = branch / np.linalg.norm(branch)
    else:
        for digit in range(dimension):
            selected = ground + digit * stride
            result[np.ix_(ground, ground)] += quantum[np.ix_(selected, selected)]
    return result


class MyLossImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(
        operation: MyLoss, state: EvolutionState, context: ExecutionContext
    ) -> EvolutionState:
        occupied = state.classical.occupied
        if occupied is None:
            raise ValueError("MyLoss requires explicit atom occupancy")
        extra = state.classical.extra
        for target in context.targets:
            if target in occupied and context.rng.random() < extra["loss_probability"]:
                # Discard quantum correlations and record the same loss event.
                state.quantum = discard_subsystem(
                    state.quantum, target, context.system_dims, context.rng
                )
                occupied.remove(target)
                extra["loss_count"] += 1
        return state


class ReportLossCountImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(operation, state, context):
        if not 0 <= operation.destination < len(state.classical.clbits):
            raise ValueError("destination must be a valid classical slot")
        state.classical.clbits[operation.destination] = state.classical.extra[
            "loss_count"
        ]
        return state


def make_simulator(runtime="numba", method="statevector"):
    implementations = AtomArraySimulator(
        runtime=runtime, method=method
    ).implementation_map
    implementations.add(SetLossProbability, SetLossProbabilityImplementation())
    implementations.add(MyLoss, MyLossImplementation())
    implementations.add(ReportLossCount, ReportLossCountImplementation())
    return AtomArraySimulator(
        runtime=runtime,
        method=method,
        implementation_map=implementations,
    )


def initial_state():
    """Per-run initialization; quantum=None requests the usual ground state."""
    return EvolutionState(
        classical=ClassicalState(extra={"loss_probability": 0.0, "loss_count": 0})
    )


def make_program(probability=0.25):
    program = fq.Program(1, 3)
    program.add(ops.Put, 0)
    program.add(ops.RX(np.pi), 0)
    program.add(SetLossProbability(probability), ())
    program.add(MyLoss(), 0)
    program.add(ops.RX(np.pi), 0)  # Surviving atoms return to ground.
    program.measure(0, 0)
    program.add(ops.Put, 0)  # Refill a missing atom in ground.
    program.measure(0, 1)
    program.add(ReportLossCount(2), ())
    return program


if __name__ == "__main__":
    for selected_runtime in ("numpy", "numba"):
        for selected_method in ("statevector", "density_matrix"):
            run_result = (
                make_simulator(selected_runtime, selected_method)
                .run(
                    make_program(),
                    shots=256,
                    initial_state=initial_state(),
                    simulation_config={"seed": 36},
                )
                .result()
            )
            print(selected_runtime, selected_method, run_result.get_counts())
