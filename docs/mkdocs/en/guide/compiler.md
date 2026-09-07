# Compile a logical circuit

Use [`LogicalProgram`][fatqat.LogicalProgram] when you want FatQat to choose a
hardware-family instruction set, map logical qubits, and route two-qubit gates.
It is a small, editable circuit frontend with familiar gate methods:

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
```

`LogicalProgram` is distinct from [`Program`][fatqat.Program]. A
`LogicalProgram` is input to the compiler. A `Program` is the general container
accepted by simulators and emulators, including direct controls and dynamic
behavior that the current compiler does not lower.

## Compile and run on an SC profile

The public superconducting route targets
[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator]. Its coupling graph is
the compiler's routing target; it may be any valid graph and does not need to
be a grid.

```python
backend = fq.simulator.SCQubitSimulator(
    num_qubits=3,
    couplings=((0, 1), (1, 2)),
    runtime="numpy",
)

compiled = fq.compiler.compile_to_sc(circuit, backend, seed=7)
native_program, layout = fq.compiler.to_sc_simulator_program(compiled.output)

counts = (
    backend.run(
        native_program,
        shots=100,
        resource_layout=layout,
        simulation_config={"seed": 7},
    )
    .result()
    .get_counts()
)
```

The route freezes the editable circuit, normalizes it to the common SC gate
IR, performs SABRE mapping and routing, and lowers it to the public
X/SX/RZ/CZ native profile. The rotation/iSWAP profile is currently private and
is not part of the public compiler contract.

## Compile for a neutral-atom architecture

The neutral-atom route uses the bundled ZAP scheduler:

```python
from fatqat.compiler.algorithms.zap import load_architecture

architecture = load_architecture("default")
compiled = fq.compiler.compile_to_na(circuit, architecture)
plan = compiled.output
```

The result is a zoned physical plan. Use
[`to_na_simulator_program`][fatqat.compiler.to_na_simulator_program] when you
want to execute that plan with the atom-array simulator.

## Supported source behavior

The v0.3 compiler accepts static, numeric gate circuits. Measurements must be
terminal. Bind symbolic parameters with `assign_parameters()` before
compilation; mid-circuit
measurement, conditions, feed-forward, and general control flow are not yet
compiler inputs.

```python
theta = fq.Parameter("theta")
template = fq.LogicalProgram(1).rx(theta, 0)
bound = template.assign_parameters({theta: 0.25})
```

The frontend offers the union of the current logical gate methods. Target
normalization reports unsupported combinations: in particular, the current NA
route rejects `sx()` and `reset()`. Such a failure is reported as a
[`PassError`][fatqat.compiler.PassError] naming the pass and the underlying
unsupported operation.

OpenQASM remains an equal frontend through
[`compile_qasm_to_sc`][fatqat.compiler.compile_qasm_to_sc] and
[`compile_qasm_to_na`][fatqat.compiler.compile_qasm_to_na]. Python and QASM
inputs converge at the same immutable logical IR, so all later lowering is
shared.
