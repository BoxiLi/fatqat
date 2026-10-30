"""Tests LOSimulator's fixed-photon method against its statevector method."""

import math

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError, ResultFieldUnavailableError
from fatqat.noise import Depolarizing
from fatqat.simulator import BasisState, LOSimulator, Simulator

_STATE_ONLY = {"shots": 0, "result_config": {"counts": False, "final_state": True}}


def _mesh(num_modes, seed=0):
    """Return a program of random beam splitters and phases on every pair."""
    rng = np.random.default_rng(seed)
    modes = fq.ModeRegister(num_modes)
    counts = fq.CountRegister(num_modes)
    program = fq.Program([modes], [counts])
    for layer in range(num_modes):
        for first in range(layer % 2, num_modes - 1, 2):
            program.add(
                ops.BeamSplitter(rng.uniform(0, 1.5), rng.uniform(0, 2 * math.pi)),
                (modes[first], modes[first + 1]),
            )
            program.add(ops.PhaseShifter(rng.uniform(0, 2 * math.pi)), modes[first])
    return program, modes, counts


def _measure_all(program, modes, counts):
    program.measure(
        tuple(modes[i] for i in range(modes.size)),
        tuple(counts[i] for i in range(counts.size)),
    )


def test_fixed_photon_state_is_the_statevector_in_its_sector():
    program, modes, _counts = _mesh(4)
    start = BasisState({modes[0]: 1, modes[1]: 2})

    fixed = LOSimulator("fixed_photon_state").run(
        program, initial_state=start, **_STATE_ONLY
    )
    fixed = fixed.result()
    statevector = (
        LOSimulator("SV", runtime="numpy")
        .run(program, initial_state=start, **_STATE_ONLY)
        .result()
        .get_statevector()
    )

    occupations = fixed.metadata["occupations"]
    # Every occupation of 3 photons in 4 modes, ascending like the statevector.
    assert occupations.shape == (math.comb(6, 3), 4)
    assert (occupations.sum(axis=1) == 3).all()
    rows = np.ravel_multi_index(occupations.T, (4,) * 4)
    assert (np.diff(rows) > 0).all()
    np.testing.assert_allclose(
        fixed.get_fixed_photon_state(), statevector[rows], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(np.delete(statevector, rows), 0, rtol=0, atol=1e-12)
    assert fixed.metadata["method"] == "fixed_photon_state"
    assert fixed.metadata["runtime"] == "numpy"
    with pytest.raises(ResultFieldUnavailableError):
        fixed.get_statevector()


def test_fixed_photon_counts_match_statevector_with_feedforward():
    program, modes, counts = _mesh(3)
    # Hong-Ou-Mandel on modes 0 and 1 leaves 0 or 2 photons in mode 0; the
    # mixing that follows depends on the measured count.
    program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
    program.measure(modes[0], counts[0])
    program.add(
        ops.BeamSplitter(math.pi / 3), (modes[1], modes[2]), condition=((counts[0], 2),)
    )
    _measure_all(program, modes, counts)
    start = BasisState({modes[0]: 1, modes[1]: 1, modes[2]: 1})
    config = {"seed": 3, "shot_parallelism": "serial"}
    options = {"shots": 300, "initial_state": start, "simulation_config": config}

    fixed = LOSimulator("fixed_photon_state").run(program, **options).result()
    dense = LOSimulator("SV", runtime="numpy").run(program, **options).result()

    # Same probabilities in the same order draw the same outcomes.
    assert fixed.get_counts() == dense.get_counts()
    assert all(sum(map(int, key)) == 3 for key in fixed.get_counts())


def test_fixed_photon_runs_where_a_statevector_cannot_be_stored():
    program, modes, _counts = _mesh(12)
    start = BasisState({modes[index]: 1 for index in range(6)})

    result = LOSimulator("fixed_photon_state").run(
        program, initial_state=start, **_STATE_ONLY
    )

    # 12 modes with a cutoff of 6 would need 7**12 (about 1.4e10) amplitudes.
    state = result.result().get_fixed_photon_state()
    assert len(state) == math.comb(17, 6)
    assert np.linalg.norm(state) == pytest.approx(1.0)


def test_fixed_photon_sweep_matches_explicit_runs():
    phi = fq.Parameter("phi")
    modes = fq.ModeRegister(2)
    program = fq.Program([modes])
    program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
    program.add(ops.PhaseShifter(phi), modes[1])
    program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
    values = np.linspace(0.0, math.pi, 3)
    backend = LOSimulator("fixed_photon_state")
    options = {"initial_state": BasisState({modes[0]: 2}), **_STATE_ONLY}

    swept = backend.run_sweep(program, {phi: values}, **options).result()

    for result, value in zip(swept, values, strict=True):
        bound = program.assign_parameters({phi: value})
        expected = backend.run(bound, **options).result().get_fixed_photon_state()
        np.testing.assert_allclose(
            result.get_fixed_photon_state(), expected, rtol=0, atol=1e-12
        )


def test_fixed_photon_rejects_what_changes_the_photon_number():
    flip = fq.Program(1)
    flip.add(ops.X, 0)
    with pytest.raises(BackendValidationError, match="conserve the photon number"):
        LOSimulator("fixed_photon_state").run(flip, **_STATE_ONLY).result()

    program, modes, _counts = _mesh(2)
    noise = fq.NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=ops.BeamSplitter)
    with pytest.raises(BackendValidationError, match="non-unitary"):
        LOSimulator("fixed_photon_state", noise=noise).run(
            program, initial_state=BasisState({modes[0]: 1}), **_STATE_ONLY
        )

    with pytest.raises(BackendValidationError, match="not a dense initial_state"):
        LOSimulator("fixed_photon_state").run(program, initial_state=np.eye(4)[1])


def test_fixed_photon_is_a_numpy_method_of_losimulator_only():
    assert LOSimulator("fixed_photon_state").method == "fixed_photon_state"
    with pytest.raises(BackendValidationError, match="only runtime='numpy'"):
        LOSimulator("fixed_photon_state", runtime="numba")
    with pytest.raises(BackendValidationError, match="unsupported method"):
        Simulator("fixed_photon_state")
