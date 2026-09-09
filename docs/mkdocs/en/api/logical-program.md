---
title: "LogicalProgram"
---

# LogicalProgram

[`LogicalProgram`][fatqat.LogicalProgram] inherits the constructor and authoring
methods of [`Program`](program.md). Both types can be passed to
`compile_to_sc()` and `compile_to_na()`.

**Under development:** logical-only operation guards still need to be defined.
For now, `LogicalProgram` accepts all `Program` operations, including resource
operations. Compilation applies the
[static gate and target restrictions](../guide/compiler.md#supported-source-behavior).

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2, metadata={"name": "bell"})
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()
compiled = fq.compiler.compile_to_sc(circuit, fq.simulator.SCQubitSimulator())
```

The inherited `copy()` and `assign_parameters()` methods return `Program`
instances, which remain valid compiler inputs. See [Compiler](compiler.md)
for compilation options and results.

## Reference

::: fatqat.LogicalProgram
    options:
      inherited_members: false
      show_bases: true
