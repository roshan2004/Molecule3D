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

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import elements
from .molecule import Molecule

_MAPPING_FORMAT = "molscope-cg-mapping"
_MAPPING_VERSION = 1

_BACKBONE = ("N", "CA", "C", "O", "OXT")


@dataclass(frozen=True)
class BeadMapping:
    """One coarse-grained bead and the atoms assigned to it.

    ``atom_indices`` are positions into the *source* (atomistic) molecule, in
    the same order as ``atom_names``. They drive mapping visualisation and
    export; ``atom_names`` stays the human-readable view.
    """

    name: str
    atom_names: list[str]
    reduction: str
    resname: str = ""
    resid: Optional[int] = None
    chain: str = ""
    atom_indices: list[int] = field(default_factory=list)


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

    @property
    def n_beads(self) -> int:
        """Number of coarse-grained beads."""
        return len(self.beads)

    @property
    def n_assigned(self) -> int:
        """Number of atoms folded into a bead."""
        return sum(len(bead.atom_indices) for bead in self.beads)

    @property
    def n_dropped(self) -> int:
        """Number of atoms left unassigned (and dropped)."""
        return len(self.dropped_atoms)

    def __str__(self) -> str:
        return self.format()

    def coverage(self) -> str:
        """One-line summary of how much of the structure the mapping covered."""
        total = self.n_assigned + self.n_dropped
        beads = f"{self.n_beads} bead{'' if self.n_beads == 1 else 's'}"
        atoms = f"{self.n_assigned}/{total} atom{'' if total == 1 else 's'}"
        line = f"{beads} from {atoms}"
        if self.n_dropped:
            line += f" ({self.n_dropped} dropped)"
        return line

    def format(self) -> str:
        lines = [f"Mapping: {self.mapping}", self.coverage(), "", "Beads:"]
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
                atom_indices=[int(i) for i in members],
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
                    atom_indices=[int(i) for i in members],
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


def bead_label(bead: BeadMapping, index: Optional[int] = None) -> str:
    """A short, unique-ish label for a bead, e.g. ``"BB (12 ALA A)"``.

    Falls back to ``name#index`` when no residue metadata is available, so beads
    from index mappings still get distinct labels.
    """
    residue = " ".join(
        str(p) for p in (bead.resid, bead.resname, bead.chain) if p not in (None, "")
    )
    if residue:
        return f"{bead.name} ({residue})"
    if index is not None:
        return f"{bead.name}#{index}"
    return bead.name


# -- mapping export / round-trip -------------------------------------------


def mapping_to_dict(cg: Molecule) -> dict:
    """Serialise a coarse-grained molecule's mapping to a plain ``dict``.

    The record captures, per bead, the source atom indices and names, the
    reduction used, residue metadata and the bead position, plus the bead-index
    bond network and any dropped atoms. It is JSON-serialisable and can be
    re-applied with :func:`apply_mapping`.
    """
    report = cg.coarse_grain_report
    beads = []
    for index, bead in enumerate(report.beads):
        beads.append({
            "index": index,
            "name": bead.name,
            "resname": bead.resname,
            "resid": bead.resid,
            "chain": bead.chain,
            "reduction": bead.reduction,
            "atom_indices": list(bead.atom_indices),
            "atom_names": list(bead.atom_names),
            "position": [float(x) for x in cg.coords[index]],
        })
    # Only explicit CG bonds are part of the mapping; never geometry-infer beads.
    bonds = (
        [[int(i), int(j)] for i, j in cg.bond_index]
        if cg.bond_index is not None else []
    )
    dropped = [
        {
            "name": atom.name,
            "element": atom.element,
            "resname": atom.resname,
            "resid": atom.resid,
            "chain": atom.chain,
        }
        for atom in report.dropped_atoms
    ]
    return {
        "format": _MAPPING_FORMAT,
        "version": _MAPPING_VERSION,
        "name": cg.name,
        "mapping": report.mapping,
        "n_beads": report.n_beads,
        "n_atoms_assigned": report.n_assigned,
        "n_dropped": report.n_dropped,
        "beads": beads,
        "bonds": bonds,
        "dropped_atoms": dropped,
    }


def write_mapping(cg: Molecule, path: str) -> str:
    """Write a coarse-grained mapping to a JSON file (see :func:`mapping_to_dict`)."""
    path = os.fspath(path)
    with open(path, "w") as fh:
        json.dump(mapping_to_dict(cg), fh, indent=2)
        fh.write("\n")
    return path


def read_mapping(path: str) -> dict:
    """Read a mapping JSON written by :func:`write_mapping`.

    Returns the record ``dict``; apply it to a structure with
    :func:`apply_mapping`.
    """
    with open(os.fspath(path)) as fh:
        record = json.load(fh)
    fmt = record.get("format")
    if fmt != _MAPPING_FORMAT:
        raise ValueError(
            f"{path}: not a molscope CG mapping (format={fmt!r})"
        )
    return record


def apply_mapping(molecule: Molecule, record: dict) -> Molecule:
    """Re-apply a saved mapping ``record`` to ``molecule``.

    Beads are rebuilt from the stored atom indices (so repeated bead names such
    as ``BB``/``SC`` round-trip cleanly) and reduced exactly as recorded. The
    structure must expose every referenced atom index, so apply a mapping to the
    structure it was built from, or one with the same atom ordering.
    """
    beads = record.get("beads", [])
    if not beads:
        raise ValueError("mapping record has no beads")

    n = len(molecule)
    coords, names, resnames, resids, chains = [], [], [], [], []
    bead_report: list[BeadMapping] = []
    assigned: set[int] = set()
    has_residues = False
    for bead in beads:
        members = [int(i) for i in bead["atom_indices"]]
        out_of_range = [i for i in members if not 0 <= i < n]
        if out_of_range:
            raise ValueError(
                f"bead {bead.get('name')!r} references atom index "
                f"{out_of_range[0]} but the molecule has {n} atoms"
            )
        assigned.update(members)
        use_mass = bead.get("reduction") == "centre of mass"
        coords.append(_reduce(molecule, members, use_mass))
        names.append(bead["name"])
        resname, resid, chain = bead.get("resname", ""), bead.get("resid"), bead.get("chain", "")
        resnames.append(resname)
        resids.append(0 if resid is None else int(resid))
        chains.append(chain)
        has_residues = has_residues or resid is not None
        bead_report.append(
            BeadMapping(
                name=bead["name"],
                atom_names=list(bead.get("atom_names", _atom_names(molecule, members))),
                reduction=bead.get("reduction", _reduction_name(use_mass)),
                resname=resname,
                resid=resid,
                chain=chain,
                atom_indices=members,
            )
        )

    bonds = record.get("bonds") or []
    bond_index = np.array(bonds, dtype=int).reshape(-1, 2) if bonds else None
    bond_report = [
        BondMapping(names[i], names[j], "(from mapping file)") for i, j in (bonds or [])
    ]
    dropped = _dropped_atoms(molecule, assigned)
    report = CoarseGrainReport(
        mapping=record.get("mapping", "custom"),
        beads=bead_report,
        dropped_atoms=dropped,
        bonds=bond_report,
    )
    kwargs = dict(
        elements=[""] * len(coords),
        name=record.get("name") or f"{molecule.name} (CG)",
        atom_names=names,
        bond_index=bond_index,
        _mapping_report=report,
    )
    if has_residues:
        kwargs.update(
            resnames=resnames,
            resids=np.array(resids, dtype=int),
            chains=chains,
        )
    return Molecule(np.array(coords, dtype=float), **kwargs)


def write_index(cg: Molecule, path: str, per_line: int = 15) -> str:
    """Write a GROMACS-style ``.ndx`` index file, one group per bead.

    Each group lists the 1-based serial numbers of the source atoms folded into
    that bead (serial = atom index + 1). Handy for inspecting a mapping in tools
    that read index files. This is an educational convenience, not a validated
    GROMACS topology.
    """
    path = os.fspath(path)
    report = cg.coarse_grain_report
    used: dict[str, int] = {}
    lines = [
        f"; molscope coarse-grain index ({report.mapping})",
        f"; {report.coverage()}",
    ]
    for index, bead in enumerate(report.beads):
        group = _index_group_name(bead, index, used)
        lines.append(f"[ {group} ]")
        serials = [i + 1 for i in bead.atom_indices]
        for start in range(0, len(serials), per_line):
            lines.append(" ".join(str(s) for s in serials[start:start + per_line]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _index_group_name(bead: BeadMapping, index: int, used: dict[str, int]) -> str:
    """Build a unique, whitespace-free group name for one bead."""
    parts = [bead.name]
    if bead.resid is not None:
        parts.append(str(bead.resid))
    if bead.resname:
        parts.append(bead.resname)
    if bead.chain:
        parts.append(bead.chain)
    base = re.sub(r"\s+", "_", "_".join(parts)) or f"bead{index}"
    count = used.get(base, 0)
    used[base] = count + 1
    return base if count == 0 else f"{base}_{count + 1}"
