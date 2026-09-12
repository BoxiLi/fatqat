---
title: "Compiler"
---

# Compiler

!!! warning "Compiler under development"

    The compiler module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

Pass a [`Program`][fatqat.Program] to the static gate compiler to map and
lower it for a superconducting or neutral-atom target. Compilation snapshots
the source without editing it. See the [compiler guide](../guide/compiler.md)
for supported source behavior and complete execution examples.

```python
import fatqat as fq

circuit = fq.Program(2, 2)
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()
```

## Compile a Python circuit

::: fatqat.compiler.compile_to_sc

::: fatqat.compiler.compile_to_na

## Compile OpenQASM

::: fatqat.compiler.compile_qasm_to_sc

::: fatqat.compiler.compile_qasm_to_na

## Results

Final target helpers return an executable compilation result. It retains the
compiler IR and pass route while carrying the simulator program and device
layout required by `Simulator.run()`.

::: fatqat.compiler.ExecutableCompilationResult

::: fatqat.compiler.CompilationResult

::: fatqat.ExecutableProgram

## Low-level simulator translation

These functions project manually created or modified final IR. Normal callers
can pass the result of a final-target compile helper directly to the matching
simulator.

SC native programs can declare `classical_registers` as a keyword-only tuple
of the original register objects in output order. The bridge retains all slots,
including unwritten slots and entirely unused registers. Distinct registers may
share a name; declaring the same object twice is invalid. Measurement outputs
must belong to declared registers. The bridge and native verifier raise
`ValidationError` for invalid explicit declarations.

The default `None` preserves compatibility with native programs constructed
without declarations: the bridge discovers registers in first measurement
occurrence order. This fallback cannot recover original declaration order or
registers with no measurements. Use `()` to declare no classical registers;
it does not request inference. Compiler-generated native programs always carry
explicit declarations. Adding this field preserves the existing three positional
constructor arguments; it also participates in dataclass equality and reflection.

::: fatqat.compiler.dialects.SCNativeProgram

::: fatqat.compiler.to_sc_simulator_program

::: fatqat.compiler.to_na_simulator_program

## Errors

::: fatqat.compiler.PassError

## LogicalProgram

[`LogicalProgram`][fatqat.LogicalProgram] is a restricted
[`Program`](program.md) for device-independent circuits. It uses the same
constructor and authoring interface; `add()`, `measure()`, and `measure_all()`
mutate in place and return `None`. Both types can be passed to
`compile_to_sc()` and `compile_to_na()`.

`add()` accepts the built-in qubit and qudit circuit gates, reset, and
barriers. It rejects device operations (including atom placement, pairing,
and pulse controls) and custom operation classes, including subclasses of
built-in gates, with `ValueError`. Measurements use `measure()`.

Classical conditions remain available for direct simulation. Register views
follow `Program`'s authoring rules and expand into scalar gates when compiled.
Compilation applies the
[static gate and target restrictions](../guide/compiler.md#supported-source-behavior).

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2, metadata={"name": "bell"})
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()
compiled = fq.compiler.compile_to_sc(circuit, fq.simulator.SCQubitSimulator())
```

The inherited `copy()` and `assign_parameters()` methods return independently
editable `LogicalProgram` instances with the same operation restrictions.

::: fatqat.LogicalProgram
    options:
      inherited_members: false
      show_bases: true
