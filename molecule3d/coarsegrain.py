"""Coarse-graining: map an atomistic structure onto a smaller set of beads.

The result is an ordinary :class:`~molecule3d.molecule.Molecule` whose "atoms"
are beads, so it plots, transforms and exports to a graph like any other. Bead
positions are the mass-weighted centre (or geometric centroid) of their member
atoms, and explicit CG bonds are attached.

Built-in modes:

- ``"residue_com"``      — one bead per residue at its centre of mass
- ``"residue_centroid"`` — one bead per residue at its geometric centroid
- ``"martini"``          — a simplified backbone + side-chain (BB/SC) model

Custom mappings come in two forms:

- **By residue/atom name** (needs PDB/mmCIF metadata)::

      {"ALA": {"BB": ["N", "CA", "C", "O"], "SC": ["CB"]}}

  Bonds are generated automatically (sequential within a residue, plus a
  backbone chain between consecutive residues) unless you pass ``bonds=``.

- **By atom index** (works on any structure, including ``.xyz``)::

      {"head": [0, 1, 2, 3], "tail": [4, 5, 6, 7]}

  No bonds are added unless you pass ``bonds=`` (see below).

``bonds`` lets you define the bead network explicitly as pairs of bead names or
bead indices, e.g. ``bonds=[("head", "tail")]`` or ``bonds=[(0, 1)]``.

Intended for teaching and prototyping CG mappings, not as a substitute for
production Martini parameters.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import elements
from .molecule import Molecule

_BACKBONE = ("N", "CA", "C", "O", "OXT")


def coarse_grain(molecule: Molecule, mapping="residue_com", weighted: bool = True,
                 bonds=None) -> Molecule:
    """Coarse-grain ``molecule``; see the module docstring for the options."""
    if _is_index_mapping(mapping):
        return _by_index(molecule, mapping, weighted, bonds)

    if len(molecule.resids) == 0:
        raise ValueError(
            "coarse-graining by residue needs residue information; for a file "
            "without it (e.g. .xyz) use an index mapping {bead: [atom_indices]}"
        )
    return _by_residue(molecule, mapping, weighted, bonds)


def _is_index_mapping(mapping) -> bool:
    """An index mapping is a dict whose values are index lists, not sub-dicts."""
    if not isinstance(mapping, dict) or not mapping:
        return False
    return not isinstance(next(iter(mapping.values())), dict)


def _by_index(molecule: Molecule, mapping: dict, weighted: bool, bonds) -> Molecule:
    bead_names: list[str] = []
    bead_coords: list[np.ndarray] = []
    assigned: set[int] = set()
    for name, members in mapping.items():
        try:
            members = [int(i) for i in members]
        except (TypeError, ValueError):
            raise ValueError(
                f"bead {name!r}: an index mapping expects integer atom indices. "
                "For atom-name beads use a residue mapping {resname: {bead: [names]}}."
            ) from None
        if not members:
            continue
        assigned.update(members)
        bead_coords.append(_reduce(molecule, members, weighted))
        bead_names.append(name)

    if not bead_coords:
        raise ValueError("mapping produced no beads")
    _warn_dropped(len(molecule), assigned)
    return Molecule(
        np.array(bead_coords, dtype=float), elements=[""] * len(bead_coords),
        name=f"{molecule.name} (CG)", atom_names=bead_names,
        bond_index=_resolve_bonds(bonds, bead_names),
    )


def _by_residue(molecule: Molecule, mapping, weighted: bool, bonds) -> Molecule:
    use_mass = weighted and mapping != "residue_centroid"
    bead_coords, bead_names = [], []
    bead_resnames, bead_resids, bead_chains = [], [], []
    residue_beads: list[list[int]] = []
    assigned: set[int] = set()

    for atom_idx, resname, resid, chain in _iter_residues(molecule):
        local = []
        for bead_name, members in _residue_beads(molecule, atom_idx, resname, mapping):
            if not members:
                continue
            assigned.update(members)
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
    if isinstance(mapping, dict):  # only custom mappings can leave atoms unassigned
        _warn_dropped(len(molecule), assigned)

    if bonds is not None:
        bond_index = _resolve_bonds(bonds, bead_names)
    else:
        bond_index = _cg_bonds(residue_beads, bead_chains)
    return Molecule(
        np.array(bead_coords, dtype=float), elements=[""] * len(bead_coords),
        name=f"{molecule.name} (CG)", atom_names=bead_names, resnames=bead_resnames,
        resids=np.array(bead_resids, dtype=int), chains=bead_chains,
        bond_index=bond_index,
    )


def _residue_beads(molecule: Molecule, atom_idx, resname, mapping):
    """Return ``[(bead_name, [atom_index, ...]), ...]`` for one residue."""
    if mapping in ("residue_com", "residue_centroid"):
        return [(resname or "BEAD", atom_idx)]
    if mapping == "martini":
        return _backbone_sidechain(molecule, atom_idx)

    spec = mapping.get(resname)
    if spec is None:
        warnings.warn(
            f"no mapping for residue {resname!r}; collapsing it to one bead",
            stacklevel=4,
        )
        return [(resname or "BEAD", atom_idx)]
    names = {molecule.atom_names[i]: i for i in atom_idx}
    return [
        (bead, [names[a] for a in atoms if a in names])
        for bead, atoms in spec.items()
    ]


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
            yield list(range(start, i)), resnames[start], int(molecule.resids[start]), chains[start]
            start = i
            key_prev = key


def _reduce(molecule: Molecule, members, use_mass: bool) -> np.ndarray:
    coords = molecule.coords[members]
    if use_mass:
        w = np.array([elements.mass(molecule.elements[i]) for i in members])
        return (w[:, None] * coords).sum(axis=0) / w.sum()
    return coords.mean(axis=0)


def _resolve_bonds(bonds, bead_names) -> np.ndarray | None:
    """Turn user bond pairs (bead names or indices) into an (E, 2) index array."""
    if bonds is None:
        return None
    name_to_idx = {}
    for i, n in enumerate(bead_names):
        name_to_idx.setdefault(n, i)
    pairs = []
    for a, b in bonds:
        ai = name_to_idx[a] if isinstance(a, str) else int(a)
        bi = name_to_idx[b] if isinstance(b, str) else int(b)
        pairs.append((ai, bi))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _cg_bonds(residue_beads, bead_chains) -> np.ndarray:
    """Bonds within each residue (sequential) plus a chain between residues."""
    bonds: list[tuple[int, int]] = []
    for beads in residue_beads:
        bonds.extend(zip(beads, beads[1:]))
    for prev, curr in zip(residue_beads, residue_beads[1:]):
        if bead_chains[prev[0]] == bead_chains[curr[0]]:
            bonds.append((prev[0], curr[0]))
    return np.array(bonds, dtype=int).reshape(-1, 2)


def _warn_dropped(n_atoms: int, assigned: set) -> None:
    dropped = n_atoms - len(assigned)
    if dropped:
        warnings.warn(
            f"{dropped} atom(s) were not assigned to any bead and were dropped",
            stacklevel=3,
        )
