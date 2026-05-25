"""Coarse-graining: map an atomistic structure onto a smaller set of beads.

The result is an ordinary :class:`~molscope.molecule.Molecule` whose "atoms"
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

``bonds`` lets you define the bead network explicitly as pairs of bead indices,
or by bead name when names are unique, e.g. ``bonds=[("head", "tail")]`` or
``bonds=[(0, 1)]``. Repeated names such as ``BB``/``SC`` in Martini-like residue
mappings are ambiguous; use bead indices for those.

Intended for teaching and prototyping CG mappings, not as a substitute for
production Martini parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import elements
from .molecule import Molecule

_BACKBONE = ("N", "CA", "C", "O", "OXT")


@dataclass(frozen=True)
class BeadMapping:
    """One coarse-grained bead and the atom names assigned to it."""

    name: str
    atom_names: list[str]
    reduction: str
    resname: str = ""
    resid: Optional[int] = None
    chain: str = ""


@dataclass(frozen=True)
class DroppedAtom:
    """An atom omitted by a custom coarse-graining mapping."""

    name: str
    element: str
    resname: str = ""
    resid: Optional[int] = None
    chain: str = ""


@dataclass(frozen=True)
class BondMapping:
    """One generated or user-defined CG bond."""

    a: str
    b: str
    reason: str


@dataclass(frozen=True)
class CoarseGrainReport:
    """Human-readable explanation of a coarse-graining operation."""

    mapping: str
    beads: list[BeadMapping] = field(default_factory=list)
    dropped_atoms: list[DroppedAtom] = field(default_factory=list)
    bonds: list[BondMapping] = field(default_factory=list)

    def __str__(self) -> str:
        return self.format()

    def format(self) -> str:
        lines = [f"Mapping: {self.mapping}", "", "Beads:"]
        if self.beads:
            for bead in self.beads:
                prefix = _residue_label(bead.resid, bead.resname, bead.chain)
                atoms = ", ".join(bead.atom_names) if bead.atom_names else "(none)"
                lines.append(f"  {prefix}:")
                lines.append(f"    {bead.name} bead: {atoms} -> {bead.reduction}")
        else:
            lines.append("  (none)")

        lines.extend(["", "Dropped atoms:"])
        if self.dropped_atoms:
            for atom in self.dropped_atoms:
                label = _residue_label(atom.resid, atom.resname, atom.chain)
                name = atom.name or atom.element or "(unnamed)"
                lines.append(f"  {label}: {name}")
        else:
            lines.append("  (none)")

        lines.extend(["", "Generated bonds:"])
        if self.bonds:
            for bond in self.bonds:
                lines.append(f"  {bond.a}-{bond.b} {bond.reason}")
        else:
            lines.append("  (none)")
        return "\n".join(lines)


def coarse_grain(molecule: Molecule, mapping="residue_com", weighted: bool = True,
                 bonds=None, return_report: bool = False):
    """Coarse-grain ``molecule``; see the module docstring for the options."""
    if _is_index_mapping(mapping):
        cg, report = _by_index(molecule, mapping, weighted, bonds)
        return (cg, report) if return_report else cg

    if len(molecule.resids) == 0:
        raise ValueError(
            "coarse-graining by residue needs residue information; for a file "
            "without it (e.g. .xyz) use an index mapping {bead: [atom_indices]}"
        )
    cg, report = _by_residue(molecule, mapping, weighted, bonds)
    return (cg, report) if return_report else cg


def _is_index_mapping(mapping) -> bool:
    """An index mapping is a dict whose values are index lists, not sub-dicts."""
    if not isinstance(mapping, dict) or not mapping:
        return False
    return not isinstance(next(iter(mapping.values())), dict)


def _by_index(molecule: Molecule, mapping: dict, weighted: bool, bonds) -> Molecule:
    bead_names: list[str] = []
    bead_coords: list[np.ndarray] = []
    bead_report: list[BeadMapping] = []
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
        bead_report.append(
            BeadMapping(
                name=name,
                atom_names=_atom_names(molecule, members),
                reduction=_reduction_name(weighted),
            )
        )

    if not bead_coords:
        raise ValueError("mapping produced no beads")
    dropped = _dropped_atoms(molecule, assigned)
    _warn_dropped(len(molecule), assigned)
    bond_index, bond_report = _resolve_bonds(bonds, bead_names)
    report = CoarseGrainReport(
        mapping="index",
        beads=bead_report,
        dropped_atoms=dropped,
        bonds=bond_report,
    )
    cg = Molecule(
        np.array(bead_coords, dtype=float), elements=[""] * len(bead_coords),
        name=f"{molecule.name} (CG)", atom_names=bead_names,
        bond_index=bond_index, _mapping_report=report,
    )
    return cg, report


def _by_residue(molecule: Molecule, mapping, weighted: bool, bonds) -> Molecule:
    use_mass = weighted and mapping != "residue_centroid"
    bead_coords, bead_names = [], []
    bead_resnames, bead_resids, bead_chains = [], [], []
    residue_beads: list[list[int]] = []
    bead_report: list[BeadMapping] = []
    assigned: set[int] = set()

    for atom_idx, resname, resid, chain in molecule.residue_groups():
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
            bead_report.append(
                BeadMapping(
                    name=bead_name,
                    atom_names=_atom_names(molecule, members),
                    reduction=_reduction_name(use_mass),
                    resname=resname,
                    resid=resid,
                    chain=chain,
                )
            )
        if local:
            residue_beads.append(local)

    if not bead_coords:
        raise ValueError("mapping produced no beads")
    dropped = _dropped_atoms(molecule, assigned)
    if isinstance(mapping, dict):  # only custom mappings can leave atoms unassigned
        _warn_dropped(len(molecule), assigned)

    if bonds is not None:
        bond_index, bond_report = _resolve_bonds(bonds, bead_names)
    else:
        bond_index, bond_report = _cg_bonds(residue_beads, bead_chains, bead_names)
    report = CoarseGrainReport(
        mapping=_mapping_name(mapping),
        beads=bead_report,
        dropped_atoms=dropped if isinstance(mapping, dict) or mapping == "martini" else [],
        bonds=bond_report,
    )
    cg = Molecule(
        np.array(bead_coords, dtype=float), elements=[""] * len(bead_coords),
        name=f"{molecule.name} (CG)", atom_names=bead_names, resnames=bead_resnames,
        resids=np.array(bead_resids, dtype=int), chains=bead_chains,
        bond_index=bond_index, _mapping_report=report,
    )
    return cg, report


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


def _reduce(molecule: Molecule, members, use_mass: bool) -> np.ndarray:
    coords = molecule.coords[members]
    if use_mass:
        w = np.array([elements.mass(molecule.elements[i]) for i in members])
        return (w[:, None] * coords).sum(axis=0) / w.sum()
    return coords.mean(axis=0)


def _resolve_bonds(bonds, bead_names):
    """Turn user bond pairs (bead names or indices) into an (E, 2) index array."""
    if bonds is None:
        return None, []
    name_to_idx = _unique_name_index(bead_names)
    pairs = []
    report = []
    for a, b in bonds:
        ai = _resolve_bond_endpoint(a, bead_names, name_to_idx)
        bi = _resolve_bond_endpoint(b, bead_names, name_to_idx)
        pairs.append((ai, bi))
        report.append(BondMapping(bead_names[ai], bead_names[bi], "(user-defined)"))
    return np.array(pairs, dtype=int).reshape(-1, 2), report


def _resolve_bond_endpoint(endpoint, bead_names, name_to_idx) -> int:
    if not isinstance(endpoint, str):
        return int(endpoint)
    if endpoint in name_to_idx:
        return name_to_idx[endpoint]
    if endpoint in bead_names:
        raise ValueError(
            f"bead name {endpoint!r} is repeated and cannot identify one bead; "
            "use bead indices for user-defined bonds"
        )
    raise ValueError(f"unknown bead name {endpoint!r}")


def _unique_name_index(bead_names):
    """Map unique bead names to indices; repeated names cannot identify one bead."""
    counts = {}
    for name in bead_names:
        counts[name] = counts.get(name, 0) + 1

    name_to_idx = {}
    for i, name in enumerate(bead_names):
        if counts[name] == 1:
            name_to_idx[name] = i
    return name_to_idx


def _cg_bonds(residue_beads, bead_chains, bead_names):
    """Bonds within each residue (sequential) plus a chain between residues."""
    bonds: list[tuple[int, int]] = []
    report: list[BondMapping] = []
    for beads in residue_beads:
        for a, b in zip(beads, beads[1:]):
            bonds.append((a, b))
            report.append(BondMapping(bead_names[a], bead_names[b], "within residue"))
    for prev, curr in zip(residue_beads, residue_beads[1:]):
        if bead_chains[prev[0]] == bead_chains[curr[0]]:
            bonds.append((prev[0], curr[0]))
            report.append(
                BondMapping(bead_names[prev[0]], bead_names[curr[0]], "between residues")
            )
    return np.array(bonds, dtype=int).reshape(-1, 2), report


def _warn_dropped(n_atoms: int, assigned: set) -> None:
    dropped = n_atoms - len(assigned)
    if dropped:
        warnings.warn(
            f"{dropped} atom(s) were not assigned to any bead and were dropped",
            stacklevel=3,
        )


def _atom_names(molecule: Molecule, atom_indices) -> list[str]:
    names = []
    for i in atom_indices:
        if molecule.atom_names:
            names.append(molecule.atom_names[i])
        elif molecule.elements:
            names.append(molecule.elements[i])
        else:
            names.append(str(i))
    return names


def _dropped_atoms(molecule: Molecule, assigned: set) -> list[DroppedAtom]:
    dropped = []
    chains = molecule.chains or [""] * len(molecule)
    resnames = molecule.resnames or [""] * len(molecule)
    resids = molecule.resids if len(molecule.resids) else [None] * len(molecule)
    atom_names = molecule.atom_names or [""] * len(molecule)
    for i in range(len(molecule)):
        if i in assigned:
            continue
        dropped.append(
            DroppedAtom(
                name=atom_names[i],
                element=molecule.elements[i] if molecule.elements else "",
                resname=resnames[i],
                resid=None if resids[i] is None else int(resids[i]),
                chain=chains[i],
            )
        )
    return dropped


def _reduction_name(weighted: bool) -> str:
    return "centre of mass" if weighted else "centroid"


def _mapping_name(mapping) -> str:
    return mapping if isinstance(mapping, str) else "custom residue"


def _residue_label(resid: Optional[int], resname: str, chain: str) -> str:
    parts = ["Residue"]
    if resid is not None:
        parts.append(str(resid))
    if resname:
        parts.append(resname)
    if chain:
        parts.append(chain)
    return " ".join(parts) if len(parts) > 1 else "Molecule"
