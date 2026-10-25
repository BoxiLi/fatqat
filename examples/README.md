# Custom loss operation

With this checkout installed, run from the repository root:

```sh
python examples/custom_loss.py
```

`MyLoss` inherits from `CustomOperation`, independently of built-in `Loss`.
Its behavior is registered through the existing `implementation_map`.
`execute(operation, state, context)` returns the updated `EvolutionState`.

The example passes `EvolutionState` through `run(initial_state=...)` to initialize
extra classical data, updates a loss probability from
inside the circuit, and records a counter while discarding lost subsystems.
A final custom operation copies the counter into a classical register:

- `201`: erasure was measured, refill succeeded, and one loss was recorded.
- `000`: the atom survived, ended in ground, and no loss was recorded.

It runs with NumPy and Numba using both statevectors and density matrices.
The loss implementation uses its own NumPy code for quantum updates, independent
of the runtime used for built-in operations.
All imports are public. Extra state is independent per shot; it is available to
later operations but is not automatically included in results.

See [Custom operations](../docs/mkdocs/en/api/simulators/custom-operations.md).
