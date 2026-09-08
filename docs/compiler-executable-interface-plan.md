# Compiler Executable Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let final SC and NA compiler results run directly on their matrix simulators while preserving compiler IR inspection and all existing low-level APIs.

**Architecture:** A compiler-independent runtime protocol exposes a simulator `Program` and `ResourceLayout`. Public target-aware compiler helpers package final IR, route, program, and layout in an immutable `ExecutableCompilationResult`; `Simulator.run()` unwraps that structural contract and then follows its existing execution path. Intermediate emits, low-level `Compiler.compile()`, direct `Program` execution, and explicit bridge functions remain unchanged.

**Tech Stack:** Python 3.11+, dataclasses, typing protocols, pytest, MkDocs/Sphinx documentation.

**Spec:** `docs/compiler-executable-interface-design.md`

## Global Constraints

- Work only on `codex/compiler-executable-results` in its isolated worktree.
- Do not modify the unrelated `codex/pr3-logical-program-demo` branch or root-worktree `docs/sphinx/` files.
- The simulator must not import `fatqat.compiler` or compiler dialects.
- Do not move bridge logic into backends and do not add target fingerprints or new backend classes.
- Keep `to_sc_simulator_program()` and `to_na_simulator_program()` public.
- Keep `compiled.output`, `compiled.route`, and `isinstance(compiled, CompilationResult)` compatible.
- Limit executable-result support to `Simulator.run()`; do not change `run_sweep`, `Estimator`, or pulse emulators.
- Use test-first red-green cycles and make no unrelated refactors.

---

### Task 1: Define the shared executable result contract

**Files:**
- Create: `src/fatqat/execution.py`
- Modify: `src/fatqat/compiler/core.py`
- Modify: `src/fatqat/compiler/__init__.py`
- Modify: `src/fatqat/__init__.py`
- Test: `tests/compiler/test_core.py`
- Test: `tests/compiler/test_logical_program_frontend.py`

**Interfaces:**
- Consumes: existing `Program`, `ResourceLayout`, and `CompilationResult[TargetT]`.
- Produces: `ExecutableProgram` structural protocol and `ExecutableCompilationResult[TargetT]` with `program` and `resource_layout` fields.

- [ ] **Step 1: Write the failing result-contract test**

Add a test that creates a one-qubit `Program`, a matching `ResourceLayout`, and:

```python
result = ExecutableCompilationResult(
    output="native-ir",
    route=("lower",),
    program=program,
    resource_layout=layout,
)

assert isinstance(result, CompilationResult)
assert isinstance(result, ExecutableProgram)
assert result.output == "native-ir"
assert result.route == ("lower",)
assert result.program is program
assert result.resource_layout is layout
```

Also assert that assigning to `result.program` raises `FrozenInstanceError`.

- [ ] **Step 2: Run the focused test and verify the missing imports fail**

Run: `pytest tests/compiler/test_core.py -q`

Expected: collection fails because `ExecutableProgram` and `ExecutableCompilationResult` do not exist.

- [ ] **Step 3: Implement the minimal neutral protocol and result subclass**

Create `fatqat.execution.ExecutableProgram` as a `@runtime_checkable Protocol` with read-only `program: Program` and `resource_layout: ResourceLayout` properties. Add:

```python
@dataclass(frozen=True, slots=True)
class ExecutableCompilationResult(CompilationResult[TargetT]):
    program: Program
    resource_layout: ResourceLayout
```

Export the protocol from `fatqat` and the result from `fatqat.compiler` without making the top-level package eagerly import the compiler.

- [ ] **Step 4: Run the focused test and existing compiler-core tests**

Run: `pytest tests/compiler/test_core.py tests/compiler/test_logical_program_frontend.py -q -k 'contract or top_level_fatqat'`

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add src/fatqat/execution.py src/fatqat/compiler/core.py src/fatqat/compiler/__init__.py src/fatqat/__init__.py tests/compiler/test_core.py tests/compiler/test_logical_program_frontend.py
git commit -m "feat: define executable compilation result"
```

### Task 2: Teach matrix simulators to unwrap executable inputs

**Files:**
- Modify: `src/fatqat/simulator/simulator.py`
- Test: `tests/simulator/test_simulator.py`

**Interfaces:**
- Consumes: `ExecutableProgram` from `fatqat.execution`.
- Produces: `Simulator.run(program: Program | ExecutableProgram, ...)`, with direct `Program` behavior unchanged.

- [ ] **Step 1: Write failing simulator behavior tests**

Use `ExecutableCompilationResult` around a measured one-qubit `Program` and verify:

```python
actual = simulator.run(executable, shots=8).result().get_counts()
expected = simulator.run(program, shots=8, resource_layout=layout).result().get_counts()
assert actual == expected
```

Add separate tests that an executable plus explicit `resource_layout=` raises `ValueError` with `resource_layout`, and that a plain `CompilationResult` raises `TypeError` mentioning `Program or ExecutableProgram`.

- [ ] **Step 2: Run the three tests and verify the expected failures**

Run: `pytest tests/simulator/test_simulator.py -q -k 'executable or compilation_result'`

Expected: executable input reaches `_instructions` and fails because unwrapping is absent.

- [ ] **Step 3: Add one private input-normalization helper**

Import only `ExecutableProgram` from `fatqat.execution`. Before current run validation:

```python
if isinstance(program, ExecutableProgram):
    if resource_layout is not None:
        raise ValueError("resource_layout cannot be supplied with an ExecutableProgram")
    resource_layout = program.resource_layout
    program = program.program
elif not isinstance(program, Program):
    raise TypeError("program must be a Program or ExecutableProgram")
```

Do not add SC/NA branches or alter `run_sweep()`.

- [ ] **Step 4: Run focused and direct-Program regression tests**

Run: `pytest tests/simulator/test_simulator.py -q`

Expected: PASS.

- [ ] **Step 5: Commit simulator support**

```bash
git add src/fatqat/simulator/simulator.py tests/simulator/test_simulator.py
git commit -m "feat: run executable compiler results"
```

### Task 3: Package final SC compiler results

**Files:**
- Modify: `src/fatqat/compiler/pipelines.py`
- Modify: `tests/compiler/test_pipeline.py`
- Modify: `tests/compiler/test_logical_program_frontend.py`
- Modify: `tests/compiler/test_sc_target_pipeline.py`

**Interfaces:**
- Consumes: `ExecutableCompilationResult` and `to_sc_simulator_program(SCNativeProgram) -> tuple[Program, ResourceLayout]`.
- Produces: executable results from final SC Python, QASM, and private rotation targets; intermediate emits remain plain `CompilationResult`.

- [ ] **Step 1: Write failing SC packaging and direct-run tests**

Assert the default public result is `ExecutableCompilationResult`, retains `SCNativeProgram` in `.output`, and runs as:

```python
compiled = compile_to_sc(_bell_program(), backend)
counts = backend.run(compiled, shots=32, simulation_config={"seed": 3}).result().get_counts()
```

Change the QASM routing test to run `backend.run(result, ...)` for both the public native profile and private rotation profile. Add one assertion that `emit=SCProgram.IR_ID` returns exactly `CompilationResult` and is not an `ExecutableProgram`.

- [ ] **Step 2: Run focused SC tests and verify default results are not executable**

Run: `pytest tests/compiler/test_pipeline.py tests/compiler/test_logical_program_frontend.py tests/compiler/test_sc_target_pipeline.py -q`

Expected: new executable-type/direct-run assertions fail while existing compilation assertions pass.

- [ ] **Step 3: Add final-boundary packaging in SC helpers**

Add a private helper that preserves the exact `output` and `route`, bridges only when `type(result.output)` is `SCNativeProgram` or `_RotationNativeProgram`, and returns:

```python
ExecutableCompilationResult(
    output=result.output,
    route=result.route,
    program=program,
    resource_layout=layout,
)
```

Call it from `compile_to_sc()`, `compile_qasm_to_sc()`, and `_compile_qasm_to_sc_rotation()`. Do not modify `Compiler.compile()`.

- [ ] **Step 4: Run focused SC tests**

Run: `pytest tests/compiler/test_pipeline.py tests/compiler/test_logical_program_frontend.py tests/compiler/test_sc_target_pipeline.py tests/compiler/test_sabre_routing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit SC packaging**

```bash
git add src/fatqat/compiler/pipelines.py tests/compiler/test_pipeline.py tests/compiler/test_logical_program_frontend.py tests/compiler/test_sc_target_pipeline.py
git commit -m "feat: package executable SC compiler output"
```

### Task 4: Package final NA compiler results

**Files:**
- Modify: `src/fatqat/compiler/pipelines.py`
- Modify: `tests/compiler/test_logical_program_frontend.py`
- Modify: `tests/compiler/test_na_end_to_end.py`

**Interfaces:**
- Consumes: `ExecutableCompilationResult` and `to_na_simulator_program(ZonedPlan) -> tuple[Program, ResourceLayout]`.
- Produces: executable results from final NA Python and QASM targets; `NAProgram` and earlier emits stay plain.

- [ ] **Step 1: Write failing NA packaging and direct-run tests**

Change the logical NA end-to-end test to:

```python
compiled = compile_to_na(_bell_program(), architecture)
counts = backend.run(compiled, shots=32, simulation_config={"seed": 3}).result().get_counts()
```

Add a QASM assertion that the default result is executable and can produce a statevector or terminal measurement counts. Keep the existing focused bridge/provenance test that removes terminal measurements and compares statevectors.

- [ ] **Step 2: Run focused NA tests and verify default results are not executable**

Run: `pytest tests/compiler/test_logical_program_frontend.py tests/compiler/test_na_end_to_end.py -q -k 'na'`

Expected: executable-type/direct-run assertions fail.

- [ ] **Step 3: Add final-boundary packaging in NA helpers**

Bridge and wrap only when `type(result.output) is ZonedPlan`. Call the helper from `compile_to_na()` and `compile_qasm_to_na()`. Leave `emit=NAProgram.IR_ID` and earlier boundaries as plain `CompilationResult`.

- [ ] **Step 4: Run the complete compiler test area**

Run: `pytest tests/compiler -q`

Expected: PASS, including unchanged low-level bridge and visualization tests.

- [ ] **Step 5: Commit NA packaging**

```bash
git add src/fatqat/compiler/pipelines.py tests/compiler/test_logical_program_frontend.py tests/compiler/test_na_end_to_end.py
git commit -m "feat: package executable NA compiler output"
```

### Task 5: Document the one-step compiler-to-simulator workflow

**Files:**
- Modify: `docs/mkdocs/en/guide/compiler.md`
- Modify: `docs/mkdocs/en/api/compiler.md`
- Modify: `docs/mkdocs/figure-sources/home_grover_sc.py`
- Modify: `docs/compiler-v0.3-design.md`

**Interfaces:**
- Consumes: final workflows `backend.run(compile_to_sc(...))`, `backend.run(compile_to_na(...))`, `backend.run(compile_qasm_to_sc(...))`, and `backend.run(compile_qasm_to_na(...))`.
- Produces: user-facing documentation that treats bridge functions as advanced low-level projections, not required steps.

- [ ] **Step 1: Update the guide with direct Python and QASM examples**

Show SC and NA compiled results passed directly to their simulators. Explain that `.output` retains the final native IR/plan for inspection and visualization, while intermediate `emit` results are intentionally not runnable.

- [ ] **Step 2: Update API and architecture documentation**

Add `ExecutableCompilationResult` and `ExecutableProgram` to the API explanation, retain bridge reference entries, and record the v0.3 execution boundary in the design document.

- [ ] **Step 3: Update the homepage source example**

Replace the explicit bridge call with `compiler_backend.run(compiled, ...)`, preserving the example's output and seed behavior.

- [ ] **Step 4: Build strict documentation**

Run: `mkdocs build --strict`

Expected: PASS without missing references or warnings.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/mkdocs/en/guide/compiler.md docs/mkdocs/en/api/compiler.md docs/mkdocs/figure-sources/home_grover_sc.py docs/compiler-v0.3-design.md
git commit -m "docs: simplify compiler execution workflow"
```

### Task 6: Verify the complete change

**Files:**
- Verify only; modify a task-owned file only if a check exposes a defect in this feature.

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: evidence that the public workflow, compatibility paths, code quality, and documentation all pass together.

- [ ] **Step 1: Run formatting and lint checks**

Run: `black --check src tests docs/mkdocs/figure-sources`

Run: `pylint src/fatqat tests`

Expected: PASS.

- [ ] **Step 2: Run the full test suite excluding network-sensitive packaging**

Run: `pytest -q --ignore=tests/test_packaging.py`

Expected: all tests pass.

- [ ] **Step 3: Run packaging tests separately**

Run: `pytest tests/test_packaging.py -q`

Expected: all three packaging tests pass; request network permission only if dependency resolution requires it.

- [ ] **Step 4: Rebuild strict documentation after all edits**

Run: `mkdocs build --strict`

Expected: PASS.

- [ ] **Step 5: Inspect scope and commit any verification-only correction**

Run: `git status --short && git diff --check && git log --oneline --decorate -6`

Expected: no unstaged changes, no whitespace errors, and only feature-plan/implementation commits on this branch.
