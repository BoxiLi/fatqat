"""NumPy numerics for states of one fixed total photon number.

Passive optics never changes the total photon number ``N``, so a state only
needs one amplitude per occupation that sums to ``N``: ``C(M + N - 1, N)``
amplitudes for ``M`` modes instead of ``(N + 1)**M``. Occupations are stored
in engine subsystem order, and rows ascend like the engine's flat index, so
the amplitudes are a statevector's in-sector entries in the same order.

A gate whose matrix conserves the photon number of its targets maps each
group of rows that agree outside the targets onto itself. Rows with the same
local photon number ``s`` hold the same local configurations, so each ``s``
is one small matrix applied to every group at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from ....._backends.steps import ApplyMatrixStep
from .....errors import BackendValidationError
from ...state import _LoweredBasisState

# Entries this small in a conserving gate's off-sector blocks are round-off.
_LEAK_TOLERANCE = 1e-10


class _FixedPhotonBasis:
    """Occupations with one total photon number, and per-target row blocks."""

    def __init__(self, dims: Sequence[int], photons: int):
        self.dims = tuple(dims)
        self.photons = photons
        # Enumerating in public order (last engine subsystem first) gives
        # rows in ascending engine flat-index order.
        public = _occupations(tuple(reversed(self.dims)), photons)
        self.occupations = np.ascontiguousarray(public[:, ::-1])
        self.occupations.flags.writeable = False
        self._blocks: dict[
            tuple[int, ...], tuple[tuple[np.ndarray, np.ndarray], ...]
        ] = {}

    def __len__(self) -> int:
        return len(self.occupations)

    def index(self, levels: Sequence[int]) -> int:
        """Return the row of one occupation in engine order."""
        return int(np.flatnonzero((self.occupations == levels).all(axis=1))[0])

    def blocks(
        self, targets: tuple[int, ...]
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Return ``(local indices, rows)`` for each local photon number.

        ``rows[g, j]`` is the row of group ``g`` whose targets hold local
        configuration ``local[j]``; local indices put ``targets[0]`` first,
        like an `ApplyMatrixStep` matrix.
        """
        cached = self._blocks.get(targets)
        if cached is not None:
            return cached
        target_levels = self.occupations[:, targets]
        local = np.ravel_multi_index(
            target_levels.T, tuple(self.dims[t] for t in targets)
        )
        local_photons = target_levels.sum(axis=1)
        outside = np.delete(self.occupations, targets, axis=1)
        blocks = []
        for photons in np.unique(local_photons):
            rows = np.flatnonzero(local_photons == photons)
            # Group rows by their occupation outside the targets (lexsort's
            # last key is primary), then order each group by local index.
            rows = rows[np.lexsort((local[rows], *outside[rows].T))]
            configurations = np.unique(local[rows])
            rows = rows.reshape(-1, len(configurations))
            assert (local[rows] == configurations).all(), "groups share configurations"
            blocks.append((configurations, rows))
        self._blocks[targets] = tuple(blocks)
        return self._blocks[targets]


class _FixedPhotonState(NamedTuple):
    """Amplitudes over the occupations of a `_FixedPhotonBasis`."""

    amplitudes: np.ndarray
    basis: _FixedPhotonBasis


def _occupations(dims: tuple[int, ...], photons: int) -> np.ndarray:
    """Return every occupation of ``dims`` summing to ``photons``, ascending."""
    # Memoized per call: tails repeat across levels, and a module cache
    # would keep every basis alive.
    tails: dict[tuple[int, int], np.ndarray] = {}

    def tail(first: int, remaining: int) -> np.ndarray:
        key = (first, remaining)
        if key not in tails:
            if first == len(dims):
                tails[key] = np.zeros((1 if remaining == 0 else 0, 0), np.int64)
            else:
                parts = []
                for level in range(min(dims[first] - 1, remaining) + 1):
                    rest = tail(first + 1, remaining - level)
                    parts.append(np.column_stack([np.full(len(rest), level), rest]))
                tails[key] = np.concatenate(parts).astype(np.int64)
        return tails[key]

    return tail(0, photons)


def _allocate_fixed_photon(
    basis: _FixedPhotonBasis, start: _LoweredBasisState
) -> _FixedPhotonState:
    """Return the basis state ``start`` in ``basis``, its photon-number sector."""
    assert isinstance(start, _LoweredBasisState), "fixed-photon runs start from levels"
    assert sum(start.levels) == basis.photons, "the basis holds the start's sector"
    amplitudes = np.zeros(len(basis), dtype=complex)
    amplitudes[basis.index(start.levels)] = 1.0
    return _FixedPhotonState(amplitudes, basis)


def _apply_matrix_fixed_photon(
    state: _FixedPhotonState, step: ApplyMatrixStep, dims: Sequence[int]
) -> _FixedPhotonState:
    """Apply a photon-number-conserving local matrix block by block."""
    matrix = step.matrix
    targets = tuple(step.target_indices)
    local_dims = tuple(dims[t] for t in targets)
    local_photons = np.indices(local_dims).reshape(len(targets), -1).sum(axis=0)
    # A matrix entry linking different local photon numbers would move
    # amplitude out of the sector this state stores.
    leaks = local_photons[:, None] != local_photons[None, :]
    if np.abs(matrix[leaks]).max(initial=0.0) > _LEAK_TOLERANCE:
        raise BackendValidationError(
            "method='fixed_photon_state' needs gates that conserve the photon number; "
            "a gate matrix changes it"
        )
    amplitudes = state.amplitudes.copy()
    for configurations, rows in state.basis.blocks(targets):
        block = matrix[np.ix_(configurations, configurations)]
        amplitudes[rows] = amplitudes[rows] @ block.T
    return _FixedPhotonState(amplitudes, state.basis)


def _fixed_photon_probabilities(state: _FixedPhotonState) -> np.ndarray:
    """Return normalized occupation probabilities."""
    probabilities = np.abs(state.amplitudes) ** 2
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def _project_fixed_photon(
    state: _FixedPhotonState,
    row: int,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
) -> _FixedPhotonState:
    """Keep the rows that match ``row`` on the measured subsystems."""
    del dims  # rows are not flat indices; the basis decodes them
    measured = list(measured_subsystems)
    occupations = state.basis.occupations
    keep = (occupations[:, measured] == occupations[row, measured]).all(axis=1)
    amplitudes = np.where(keep, state.amplitudes, 0.0)
    norm = np.linalg.norm(amplitudes)
    return _FixedPhotonState(amplitudes / norm if norm > 0 else amplitudes, state.basis)


def _fixed_photon_levels(
    state: _FixedPhotonState, rows: np.ndarray, subsystems: Sequence[int]
) -> np.ndarray:
    """Return the measured subsystems' levels for sampled rows."""
    return state.basis.occupations[np.ix_(np.asarray(rows, dtype=int), subsystems)]
