"""Contact maps and residue-level contact analysis.

A contact map records which pairs of atoms (or residues) are within a distance
cutoff. Residue contact maps are a staple of protein-folding intuition, peptide
and NMR-ensemble comparison, and coarse-graining validation.

    cmap = mol.contact_map(cutoff=8.0, level="residue")   # -> ContactMap
    cmap.matrix                                           # (R, R) array
    cmap.plot()                                           # heatmap
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import elements
from .molecule import Molecule


@dataclass
class ContactMap:
    """A square contact (or contact-frequency) matrix with axis labels.

    ``matrix`` holds booleans-as-floats for a single structure, or values in
    ``[0, 1]`` for an ensemble frequency map. ``labels`` name the rows/columns
    (e.g. ``"A:LYS8"``); ``resids`` are the residue numbers for a residue map.
    """

    matrix: np.ndarray
    level: str
    cutoff: float
    labels: list[str] = field(default_factory=list)
    resids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def is_frequency(self) -> bool:
        """True if the matrix holds fractional values (an ensemble map)."""
        vals = np.unique(self.matrix)
        return not np.all(np.isin(vals, (0.0, 1.0)))

    def plot(self, **kwargs):
        """Draw the contact map as a heatmap. See :func:`molecule3d.plotting.plot_contact_map`."""
        from .plotting import plot_contact_map

        return plot_contact_map(self, **kwargs)


def contact_map(molecule: Molecule, cutoff: float = 8.0, level: str = "residue",
                method: str = "ca") -> ContactMap:
    """Compute a contact map for one structure (see :class:`ContactMap`)."""
    if level == "atom":
        dist = molecule.distance_matrix()
        mat = (dist < cutoff).astype(float)
        np.fill_diagonal(mat, 0.0)
        return ContactMap(mat, level="atom", cutoff=cutoff)

    if level != "residue":
        raise ValueError(f"level must be 'atom' or 'residue', got {level!r}")

    groups = list(molecule.residue_groups())
    if not groups:
        raise ValueError("residue contact map needs residue information")
    labels = [_label(chain, resname, resid) for _, resname, resid, chain in groups]
    resids = np.array([resid for _, _, resid, _ in groups], dtype=int)
    mat = _residue_contacts(molecule, groups, cutoff, method)
    return ContactMap(mat, level="residue", cutoff=cutoff, labels=labels, resids=resids)


def _residue_contacts(molecule, groups, cutoff, method) -> np.ndarray:
    if method in ("ca", "com"):
        reps = np.array([_representative(molecule, idx, method) for idx, *_ in groups])
        deltas = reps[:, None, :] - reps[None, :, :]
        dist = np.sqrt((deltas ** 2).sum(axis=-1))
        mat = (dist < cutoff).astype(float)
    elif method == "min":
        mat = _min_distance_contacts(molecule, groups, cutoff)
    else:
        raise ValueError(f"method must be 'ca', 'com' or 'min', got {method!r}")
    np.fill_diagonal(mat, 0.0)
    return mat


def _representative(molecule, idx, method) -> np.ndarray:
    if method == "ca" and molecule.atom_names:
        ca = [i for i in idx if molecule.atom_names[i] == "CA"]
        if ca:
            return molecule.coords[ca[0]]
    if method == "com":
        w = np.array([elements.mass(molecule.elements[i]) for i in idx])
        return (w[:, None] * molecule.coords[idx]).sum(axis=0) / w.sum()
    return molecule.coords[idx].mean(axis=0)  # CA fallback: residue centroid


def _min_distance_contacts(molecule, groups, cutoff) -> np.ndarray:
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        def cdist(a, b):
            d = a[:, None, :] - b[None, :, :]
            return np.sqrt((d ** 2).sum(axis=-1))

    n = len(groups)
    coords = [molecule.coords[idx] for idx, *_ in groups]
    mat = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            if cdist(coords[a], coords[b]).min() < cutoff:
                mat[a, b] = mat[b, a] = 1.0
    return mat


def _label(chain, resname, resid) -> str:
    base = f"{resname or 'RES'}{resid}"
    return f"{chain}:{base}" if chain else base
