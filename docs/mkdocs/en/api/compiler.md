---
title: "Compiler"
---

# Compiler

[`LogicalProgram`][fatqat.LogicalProgram] is the editable Python frontend for
FatQat's static gate compiler. Its gate helpers mutate the circuit and return
the same object, so both line-by-line and chained construction are supported.

```python
import fatqat as fq

circuit = (
    fq.LogicalProgram(2, 2)
    .h(0)
    .cx(0, 1)
    .measure_all()
)
```

::: fatqat.LogicalProgram
    options:
      filters:
        - "!^_"
      show_bases: false

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
