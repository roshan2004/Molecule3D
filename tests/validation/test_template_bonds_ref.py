"""Tier 2 validation: residue-template bond perception vs known residue chemistry.

``bond_perception="template"`` routes a protein PDB through RDKit's residue-aware
reader and maps the perceived bonds back to MolScope atom indices. The honest
correctness check is not "does it equal RDKit" (that would be circular) but "does
the result match the *known chemistry* of the standard residues": the number of
aromatic ring atoms is fixed per residue type, so the perceived aromatic-atom
total must equal the sum over residues of those known counts. This catches a
mis-mapped atom correspondence, a broken Kekule round-trip, or dropped rings.

Skipped cleanly when RDKit is not installed.
"""

import os
from collections import Counter

import numpy as np
import pytest

import molscope as ms

pytestmark = pytest.mark.validation

pytest.importorskip("rdkit")

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "examples", "data")

# Aromatic heavy atoms per standard residue: Phe/Tyr six-membered ring (6),
# Trp indole (9, two fused rings sharing two atoms), His imidazole (5).
AROMATIC_ATOMS_PER_RESIDUE = {"PHE": 6, "TYR": 6, "TRP": 9, "HIS": 5}

PROTEINS = ["1ubq.pdb", "3ptb.pdb", "1shg.pdb"]


def _residue_type_counts(mol) -> Counter:
    return Counter(group.residue_id.resname for group in mol.residue_groups())


@pytest.mark.parametrize("pdb", PROTEINS, ids=[p.split(".")[0] for p in PROTEINS])
def test_aromatic_atoms_match_known_residue_chemistry(pdb):
    mol = ms.read(os.path.join(DATA, pdb), bond_perception="template")
    feats = mol.chemical_features()

    perceived = int(feats.aromatic_atoms.sum())
    counts = _residue_type_counts(mol)
    expected = sum(AROMATIC_ATOMS_PER_RESIDUE.get(rn, 0) * n for rn, n in counts.items())

    print(f"\n{pdb}: aromatic atoms perceived={perceived} expected={expected}")
    assert perceived == expected


@pytest.mark.parametrize("pdb", PROTEINS, ids=[p.split(".")[0] for p in PROTEINS])
def test_aromatic_atoms_only_in_aromatic_residues(pdb):
    mol = ms.read(os.path.join(DATA, pdb), bond_perception="template")
    aromatic = mol.chemical_features().aromatic_atoms
    flagged = {mol.resnames[i] for i, is_arom in enumerate(aromatic) if is_arom}
    assert flagged <= set(AROMATIC_ATOMS_PER_RESIDUE)  # no spurious aromatic atoms


@pytest.mark.parametrize("pdb", PROTEINS, ids=[p.split(".")[0] for p in PROTEINS])
def test_template_bond_index_is_well_formed(pdb):
    from molscope.chem import pdb_template_bonds

    path = os.path.join(DATA, pdb)
    mol = ms.read(path)
    bond_index, bond_orders, charges = pdb_template_bonds(path, mol)

    assert bond_index.shape[1] == 2
    assert bond_index.min() >= 0 and bond_index.max() < len(mol)  # valid atom indices
    assert (bond_index[:, 0] != bond_index[:, 1]).all()          # no self-bonds
    assert len(bond_orders) == len(bond_index)
    # Kekule orders only: no aromatic 1.5. (0 can appear for metal/dative bonds,
    # e.g. the Ca(2+) coordination in 3ptb.)
    assert 1.5 not in set(np.unique(bond_orders))
    assert set(np.unique(bond_orders)).issubset({0.0, 1.0, 2.0, 3.0})
    assert len(charges) == len(mol)


def test_standard_protonation_matches_residue_balance():
    """Net charge = (#Lys + #Arg) - (#Asp + #Glu), His neutral, termini uncharged."""
    path = os.path.join(DATA, "3ptb.pdb")
    mol = ms.read(path, bond_perception="template", protonation="standard")
    counts = _residue_type_counts(mol)
    expected = (counts["LYS"] + counts["ARG"]) - (counts["ASP"] + counts["GLU"])
    net = int(mol.chemical_features().formal_charges.sum())
    print(f"\n3ptb: standard-protonation net charge={net} expected={expected}")
    assert net == expected
