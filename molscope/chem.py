"""Optional RDKit-backed chemical perception.

MolScope keeps cheminformatics out of the core dependency set. When RDKit is
installed, this module can annotate a :class:`~molscope.molecule.Molecule` with
formal charges, valence, aromaticity and sanitized bond-order information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ChemicalFeatures:
    """Per-atom and per-bond features from RDKit sanitization.

    Bond arrays are aligned with the bond list used to build the RDKit molecule:
    explicit ``molecule.bond_index`` when present, otherwise geometrically
    inferred bonds when ``infer_bonds=True``.
    """

    formal_charges: np.ndarray
    total_valences: np.ndarray
    aromatic_atoms: np.ndarray
    bond_index: np.ndarray
    bond_orders: np.ndarray
    aromatic_bonds: np.ndarray


def chemical_features(molecule, *, sanitize: bool = True, infer_bonds: bool = True):
    """Return RDKit-backed chemical features for ``molecule``.

    Explicit SDF/MOL bond orders and formal charges are used when available. If
    the molecule has only coordinates, geometrically inferred bonds are passed
    to RDKit as single bonds. MolScope does not attempt general bond-order
    inference from raw coordinates.
    """
    rdmol, bond_index = to_rdkit(molecule, sanitize=sanitize, infer_bonds=infer_bonds)
    atoms = list(rdmol.GetAtoms())
    bond_orders, aromatic_bonds = [], []
    for i, j in bond_index:
        bond = rdmol.GetBondBetweenAtoms(int(i), int(j))
        if bond is None:
            bond_orders.append(0.0)
            aromatic_bonds.append(False)
        else:
            bond_orders.append(float(bond.GetBondTypeAsDouble()))
            aromatic_bonds.append(bool(bond.GetIsAromatic()))

    return ChemicalFeatures(
        formal_charges=np.array([a.GetFormalCharge() for a in atoms], dtype=int),
        total_valences=np.array([a.GetTotalValence() for a in atoms], dtype=int),
        aromatic_atoms=np.array([a.GetIsAromatic() for a in atoms], dtype=bool),
        bond_index=bond_index,
        bond_orders=np.array(bond_orders, dtype=float),
        aromatic_bonds=np.array(aromatic_bonds, dtype=bool),
    )


def rdkit_descriptors(
    molecule,
    *,
    names: Optional[list[str]] = None,
    prefix: str = "rdkit_",
    sanitize: bool = True,
    infer_bonds: bool = True,
    errors: str = "nan",
) -> dict[str, float]:
    """Return scalar RDKit molecular descriptors as a flat dictionary.

    ``names`` can restrict the descriptor set to RDKit descriptor names such as
    ``"MolWt"`` or ``"TPSA"``. Keys are prefixed by default to avoid collisions
    with MolScope-native descriptor names. ``errors="nan"`` records failing
    descriptor calculations as NaN; pass ``errors="raise"`` to surface them.
    """
    if errors not in ("nan", "raise"):
        raise ValueError("errors must be 'nan' or 'raise'")

    Chem, _ = _require_rdkit()
    from rdkit.Chem import Descriptors

    rdmol, _ = to_rdkit(molecule, sanitize=sanitize, infer_bonds=infer_bonds)
    if sanitize:
        # Keep descriptor preconditions explicit when to_rdkit was called with
        # an already-sanitized molecule; this is cheap and avoids stale caches.
        Chem.SanitizeMol(rdmol)
    else:
        rdmol.UpdatePropertyCache(strict=False)

    desc_map = dict(Descriptors._descList)
    selected = list(desc_map) if names is None else list(names)
    unknown = [name for name in selected if name not in desc_map]
    if unknown:
        raise ValueError(f"unknown RDKit descriptor(s): {', '.join(unknown)}")

    out: dict[str, float] = {}
    for name in selected:
        try:
            value = desc_map[name](rdmol)
            out[f"{prefix}{name}"] = _numeric_descriptor(value)
        except Exception:
            if errors == "raise":
                raise
            out[f"{prefix}{name}"] = float("nan")
    return out


def to_rdkit(molecule, *, sanitize: bool = True, infer_bonds: bool = True):
    """Build an RDKit molecule from a MolScope molecule.

    Returns ``(rdmol, bond_index)`` where ``bond_index`` is the edge list used to
    populate the RDKit molecule.
    """
    Chem, Point3D = _require_rdkit()
    rw = Chem.RWMol()
    charges = molecule.formal_charges if len(molecule.formal_charges) else None
    for i, symbol in enumerate(molecule.elements):
        atom = _rdkit_atom(Chem, symbol)
        if charges is not None:
            atom.SetFormalCharge(int(charges[i]))
        rw.AddAtom(atom)

    if molecule.bond_index is not None:
        bond_index = molecule.bond_index
        orders = molecule.bond_order_array()
    elif infer_bonds:
        bond_index = molecule.bonds()
        orders = np.ones(len(bond_index), dtype=float)
    else:
        bond_index = np.empty((0, 2), dtype=int)
        orders = np.empty(0, dtype=float)

    for (i, j), order in zip(bond_index, orders):
        bond_type = _rdkit_bond_type(Chem, order)
        rw.AddBond(int(i), int(j), bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            rw.GetAtomWithIdx(int(i)).SetIsAromatic(True)
            rw.GetAtomWithIdx(int(j)).SetIsAromatic(True)
            rw.GetBondBetweenAtoms(int(i), int(j)).SetIsAromatic(True)

    rdmol = rw.GetMol()
    conf = Chem.Conformer(len(molecule))
    for i, (x, y, z) in enumerate(molecule.coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    rdmol.AddConformer(conf)

    try:
        if sanitize:
            Chem.SanitizeMol(rdmol)
        else:
            rdmol.UpdatePropertyCache(strict=False)
    except Exception as exc:
        raise ValueError(
            "RDKit could not sanitize this molecule; check bond orders, valence "
            "and formal charges, or call chemical_features(sanitize=False)"
        ) from exc
    return rdmol, np.asarray(bond_index, dtype=int).reshape(-1, 2)


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "RDKit is required for chemical perception; install it with "
            'pip install "molscope[chem]"'
        ) from exc
    return Chem, Point3D


def _rdkit_atom(Chem, symbol: str):
    raw = (symbol or "").strip()
    if not raw:
        return Chem.Atom(0)
    normalized = raw[0].upper() + raw[1:].lower()
    try:
        return Chem.Atom(normalized)
    except RuntimeError:
        return Chem.Atom(0)


def _rdkit_bond_type(Chem, order: float):
    if np.isclose(order, 1.5) or np.isclose(order, 4.0):
        return Chem.BondType.AROMATIC
    rounded = int(round(float(order)))
    if rounded == 1:
        return Chem.BondType.SINGLE
    if rounded == 2:
        return Chem.BondType.DOUBLE
    if rounded == 3:
        return Chem.BondType.TRIPLE
    return Chem.BondType.UNSPECIFIED


def _numeric_descriptor(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"RDKit descriptor returned non-scalar value {value!r}")
