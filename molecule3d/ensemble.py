"""Analysis across a set of structures, e.g. the models of an NMR ensemble."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .molecule import Molecule


def align_all(models: list[Molecule], reference: Molecule | None = None) -> list[Molecule]:
    """Kabsch-superpose every model onto ``reference`` (default: the first model)."""
    ref = reference if reference is not None else models[0]
    return [m.superpose(ref) for m in models]


def average(models: list[Molecule], align: bool = True) -> Molecule:
    """Average structure over the ensemble (atoms matched by index)."""
    _check_consistent(models)
    aligned = align_all(models) if align else models
    coords = np.mean([m.coords for m in aligned], axis=0)
    return replace(models[0], coords=coords, name=f"{models[0].name} (average)")


def rmsf(models: list[Molecule], align: bool = True) -> np.ndarray:
    """Per-atom root-mean-square fluctuation about the mean position."""
    _check_consistent(models)
    aligned = align_all(models) if align else models
    stack = np.array([m.coords for m in aligned])      # (n_models, n_atoms, 3)
    mean = stack.mean(axis=0)
    return np.sqrt(((stack - mean) ** 2).sum(axis=2).mean(axis=0))


def rmsd_matrix(models: list[Molecule], align: bool = True) -> np.ndarray:
    """Symmetric ``(M, M)`` matrix of pairwise RMSDs between models."""
    _check_consistent(models)
    n = len(models)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = models[i].rmsd(models[j], align=align)
    return mat


def _check_consistent(models: list[Molecule]) -> None:
    if not models:
        raise ValueError("no models given")
    sizes = {len(m) for m in models}
    if len(sizes) != 1:
        raise ValueError(f"models have differing atom counts: {sorted(sizes)}")
