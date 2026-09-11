---
title: "Compiler"
---

# Compiler

Pass a [`Program`][fatqat.Program] or
[`LogicalProgram`][fatqat.LogicalProgram] to the static gate compiler to map and
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

::: fatqat.compiler.to_sc_simulator_program

::: fatqat.compiler.to_na_simulator_program

## Errors

::: fatqat.compiler.PassError
