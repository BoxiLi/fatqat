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

## Translate compiler output for simulation

::: fatqat.compiler.to_sc_simulator_program

::: fatqat.compiler.to_na_simulator_program

## Results and errors

::: fatqat.compiler.CompilationResult

::: fatqat.compiler.PassError

