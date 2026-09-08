# Compiler executable interface design

## Goal

Make every public compiler-to-simulator path executable without requiring users
to call a simulator bridge. Superconducting and neutral-atom compilation retain
their target-specific inputs, but produce the same kind of executable result.

The normal workflows become:

```python
sc_compiled = fq.compiler.compile_to_sc(circuit, sc_backend)
sc_result = sc_backend.run(sc_compiled, shots=1000).result()

na_compiled = fq.compiler.compile_to_na(circuit, architecture)
na_result = na_backend.run(na_compiled, shots=1000).result()
```

The same behavior applies to the OpenQASM frontends
`compile_qasm_to_sc()` and `compile_qasm_to_na()`.

## Responsibility boundary

The bridge remains compiler-owned. A public final-target compile function:

1. runs the existing typed pipeline;
2. retains the final compiler IR and pass route;
3. invokes the matching SC or NA bridge once;
4. returns the IR together with the simulator `Program` and
   `ResourceLayout`.

The simulator does not import compiler dialects and does not translate
`SCNativeProgram` or `ZonedPlan`. It only recognizes a small structural
execution contract containing an ordinary `Program` and its layout, then uses
the existing execution path unchanged.

## Public types

Add a dialect-independent runtime protocol outside the compiler package:

```python
@runtime_checkable
class ExecutableProgram(Protocol):
    @property
    def program(self) -> Program: ...

    @property
    def resource_layout(self) -> ResourceLayout: ...
```

Add a compiler result that preserves the current result contract and satisfies
that protocol:

```python
@dataclass(frozen=True, slots=True)
class ExecutableCompilationResult(CompilationResult[TargetT]):
    program: Program
    resource_layout: ResourceLayout
```

The inherited fields remain:

- `output`: the final compiler IR (`SCNativeProgram` or `ZonedPlan`);
- `route`: the pass names that produced it.

The added fields are:

- `program`: the bridge-produced simulator `Program`;
- `resource_layout`: the matching fixed device mapping.

`ExecutableCompilationResult` is immutable. It does not retain a backend
reference or claim that it can execute itself. The same compiled program can
therefore be submitted to any compatible simulator instance, including an
ideal and a noisy instance with the same supported target contract.

## Compile-function behavior

The public helpers return `ExecutableCompilationResult` when they reach their
final simulator-facing IR:

- `compile_to_sc(..., emit=SCNativeProgram.IR_ID)`;
- `compile_qasm_to_sc(..., emit=SCNativeProgram.IR_ID)`;
- `compile_to_na(..., emit=ZonedPlan.IR_ID)`;
- `compile_qasm_to_na(..., emit=ZonedPlan.IR_ID)`.

The default `emit` values already select these boundaries, so ordinary callers
always receive executable results.

An explicit intermediate `emit`, such as `LogicalIR.IR_ID`, `SCProgram.IR_ID`,
or `NAProgram.IR_ID`, continues to return the existing plain
`CompilationResult`. Intermediate IR is deliberately not simulator-executable,
and the compiler does not run later passes merely to manufacture an execution
payload.

The private rotation-profile helper follows the same rule at its private final
IR so internal SC paths do not retain a second execution convention.

The low-level `Compiler.compile()` method remains unchanged and continues to
return `CompilationResult`. Executable packaging belongs only to the public
target-aware helpers, because only they know which bridge applies.

## Simulator behavior

`Simulator.run()` accepts `Program | ExecutableProgram`.

For a direct `Program`, all existing behavior is unchanged. For an
`ExecutableProgram`, `run()` extracts `program` and `resource_layout` before
entering the existing validation and preparation path. No SC- or NA-specific
branch is added.

Supplying `resource_layout=` together with an executable result is rejected
with `ValueError`, because two layouts would otherwise compete. All other run
arguments retain their current meanings.

Backend compatibility remains enforced by existing validation. The executable
does not store a target fingerprint and does not introduce speculative target
matching. A program compiled for one backend may run on another backend when
its operations, dimensions, sites, and layout are actually compatible; normal
backend errors reject incompatible inputs.

This v0.3 change applies to matrix simulators, including
`SCQubitSimulator` and `AtomArraySimulator`. Compiler output is not promised to
run on pulse emulators, `Estimator`, or `run_sweep`; those integrations require
their own semantics for observables, parameters, and physical-state ordering.

## Bridge API and compatibility

`to_sc_simulator_program()` and `to_na_simulator_program()` remain available as
low-level public functions. Existing callers keep working. Guides and examples
stop requiring them in the normal path, but advanced users can still project a
manually constructed or modified final IR.

Existing access to `compiled.output` and `compiled.route` also remains valid.
Code that only checks `isinstance(compiled, CompilationResult)` continues to
work because the executable result is a subclass.

No compiler IR gains simulator-specific fields. No bridge logic moves into a
backend. No existing direct `Program` execution changes.

## Error behavior

- Compilation and bridge failures remain synchronous compiler errors from the
  public compile function.
- Passing an intermediate plain `CompilationResult` to `Simulator.run()` raises
  a clear `TypeError` stating that `Program` or `ExecutableProgram` is required;
  it is not treated as executable.
- Passing an executable result and an explicit `resource_layout` raises
  `ValueError` before backend preparation.
- Unsupported operations, illegal device sites, dimensions, and runtime
  settings continue to use the existing backend validation errors.

## Implementation sequence

1. Introduce and test the dialect-independent `ExecutableProgram` protocol.
2. Extend `Simulator.run()` to unwrap the protocol while preserving direct
   `Program` execution and rejecting a second layout.
3. Add `ExecutableCompilationResult` without changing `Compiler.compile()`.
4. Package final SC results with `to_sc_simulator_program()` in both Python and
   QASM helpers, including the private rotation path.
5. Package final NA results with `to_na_simulator_program()` in both Python and
   QASM helpers.
6. Convert SC and NA end-to-end tests to `backend.run(compiled)` while retaining
   focused bridge unit tests.
7. Update compiler guides, API documentation, homepage example code, and the
   compiler v0.3 design document.
8. Run focused SC/NA tests, the full suite, formatting, lint, and strict
   documentation generation.

## Acceptance criteria

- All four public final-target compilation helpers return executable results.
- `SCQubitSimulator.run(sc_compiled)` and
  `AtomArraySimulator.run(na_compiled)` execute without a manual bridge call.
- SC routing preserves logical measurement results and final layout behavior.
- NA scheduling preserves atom loading, pairing, movement-derived execution,
  and terminal measurement behavior.
- Explicit intermediate `emit` remains inspectable and non-executable.
- Existing direct `Program` calls and low-level bridge calls remain compatible.
- No simulator module imports `fatqat.compiler` or a compiler dialect.

## Alternatives not selected

Moving bridge dispatch into `backend.run()` would make simulators understand
every compiler dialect and reverse the dependency direction. Returning only a
`Program` would hide the final IR, pass route, and physical plan that compiler
users need to inspect and visualize. Storing the layout inside `Program` would
also blur its current role as a backend-independent instruction container. The
executable result preserves all three boundaries with one small shared
protocol.
