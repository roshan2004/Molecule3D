"""Coarse-graining: map an atomistic structure onto a smaller set of beads.

The result is an ordinary :class:`~molecule3d.molecule.Molecule` whose "atoms"
are beads, so it plots, transforms and exports to a graph like any other. Bead
positions are the mass-weighted centre (or geometric centroid) of their member
atoms, and explicit CG bonds are attached (intra-residue plus a backbone chain
between consecutive residues).

Built-in modes:

- ``"residue_com"``      — one bead per residue at its centre of mass
- ``"residue_centroid"`` — one bead per residue at its geometric centroid
- ``"martini"``          — a simplified backbone + side-chain (BB/SC) model

Or pass a custom ``{resname: {bead_name: [atom_names]}}`` mapping, e.g.::

    {"ALA": {"BB": ["N", "CA", "C", "O"], "SC": ["CB"]}}

This is intended for teaching and prototyping CG mappings, not as a substitute
for production Martini parameters.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import elements
from .molecule import Molecule

_BACKBONE = ("N", "CA", "C", "O", "OXT")


def coarse_grain(molecule: Molecule, mapping="residue_com", weighted: bool = True) -> Molecule:
    """Coarse-grain ``molecule``; see the module docstring for ``mapping`` options."""
    if len(molecule.resids) == 0:
        raise ValueError(
            "coarse-graining needs residue information; read a PDB/mmCIF file "
            "(an .xyz has no residues)"
        )

    centroid_mode = mapping == "residue_centroid"
    use_mass = weighted and not centroid_mode

    bead_coords: list[np.ndarray] = []
    bead_names: list[str] = []
    bead_resnames: list[str] = []
    bead_resids: list[int] = []
    bead_chains: list[str] = []
    # Per residue: indices into the bead lists, so we can wire up bonds.
    residue_beads: list[list[int]] = []

    for atom_idx, resname, resid, chain in _iter_residues(molecule):
        beads = _residue_beads(molecule, atom_idx, resname, mapping)
        local = []
        for bead_name, members in beads:
            if not members:
                continue
            bead_coords.append(_reduce(molecule, members, use_mass))
            bead_names.append(bead_name)
            bead_resnames.append(resname)
            bead_resids.append(resid)
            bead_chains.append(chain)
            local.append(len(bead_coords) - 1)
        if local:
            residue_beads.append(local)

    if not bead_coords:
        raise ValueError("mapping produced no beads")

    bonds = _cg_bonds(residue_beads, bead_chains)
    return Molecule(
        np.array(bead_coords, dtype=float),
        elements=[""] * len(bead_coords),
        name=f"{molecule.name} (CG)",
        atom_names=bead_names,
        resnames=bead_resnames,
        resids=np.array(bead_resids, dtype=int),
        chains=bead_chains,
        bond_index=bonds,
    )


def _iter_residues(molecule: Molecule):
    """Yield ``(atom_indices, resname, resid, chain)`` for each residue, in order."""
    chains = molecule.chains or [""] * len(molecule)
    resnames = molecule.resnames or [""] * len(molecule)
    key_prev = object()
    start = 0
    for i in range(len(molecule) + 1):
        key = (chains[i], int(molecule.resids[i])) if i < len(molecule) else None
        if i == 0:
            key_prev = key
            continue
        if key != key_prev or i == len(molecule):
            idx = list(range(start, i))
            yield idx, resnames[start], int(molecule.resids[start]), chains[start]
            start = i
            key_prev = key


def _residue_beads(molecule: Molecule, atom_idx, resname, mapping):
    """Return ``[(bead_name, [atom_index, ...]), ...]`` for one residue."""
    if mapping in ("residue_com", "residue_centroid"):
        return [(resname or "BEAD", atom_idx)]

    if mapping == "martini":
        return _backbone_sidechain(molecule, atom_idx)

    if isinstance(mapping, dict):
        spec = mapping.get(resname)
        if spec is None:
            warnings.warn(
                f"no mapping for residue {resname!r}; collapsing it to one bead",
                stacklevel=3,
            )
            return [(resname or "BEAD", atom_idx)]
        names = {molecule.atom_names[i]: i for i in atom_idx}
        return [
            (bead, [names[a] for a in atoms if a in names])
            for bead, atoms in spec.items()
        ]

    raise ValueError(f"unknown coarse-grain mapping {mapping!r}")


def _backbone_sidechain(molecule: Molecule, atom_idx):
    """Simplified Martini-like split: a backbone bead and a side-chain bead."""
    bb = [i for i in atom_idx if molecule.atom_names[i] in _BACKBONE]
    sc = [
        i for i in atom_idx
        if molecule.atom_names[i] not in _BACKBONE and molecule.elements[i] != "H"
    ]
    beads = []
    if bb:
        beads.append(("BB", bb))
    if sc:
        beads.append(("SC", sc))
    return beads


def _reduce(molecule: Molecule, members, use_mass: bool) -> np.ndarray:
    coords = molecule.coords[members]
    if use_mass:
        w = np.array([elements.mass(molecule.elements[i]) for i in members])
        return (w[:, None] * coords).sum(axis=0) / w.sum()
    return coords.mean(axis=0)


def _cg_bonds(residue_beads, bead_chains) -> np.ndarray:
    """Bonds within each residue (sequential) plus a chain between residues."""
    bonds: list[tuple[int, int]] = []
    for beads in residue_beads:
        for a, b in zip(beads, beads[1:]):
            bonds.append((a, b))
    # Link the anchor (first) bead of consecutive residues in the same chain.
    for prev, curr in zip(residue_beads, residue_beads[1:]):
        if bead_chains[prev[0]] == bead_chains[curr[0]]:
            bonds.append((prev[0], curr[0]))
    return np.array(bonds, dtype=int).reshape(-1, 2)
