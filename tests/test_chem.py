"""Tests for optional RDKit-backed chemical perception."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import ChemicalFeatures, Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def test_pdb_template_bonds_perceive_aromatic_rings():
    """Residue templates recover aromaticity that geometric bonds miss."""
    pytest.importorskip("rdkit")
    path = os.path.join(DATA, "1ubq.pdb")

    geometric = ms.read(path)
    template = ms.read(path, bond_perception="template")
    assert template.bond_index is not None

    geo_arom = int(sum(bool(a) for a in geometric.chemical_features().aromatic_atoms))
    tpl_arom = int(sum(bool(a) for a in template.chemical_features().aromatic_atoms))
    assert geo_arom == 0  # geometric single bonds carry no aromatic perception
    assert tpl_arom >= 20  # Phe/Tyr/His rings of ubiquitin


def test_pdb_template_bonds_returns_aligned_arrays():
    pytest.importorskip("rdkit")
    from molscope.chem import pdb_template_bonds

    path = os.path.join(DATA, "1ubq.pdb")
    bond_index, bond_orders, charges = pdb_template_bonds(path, ms.read(path))
    assert bond_index.shape[1] == 2
    assert len(bond_orders) == len(bond_index)
    assert len(charges) == 660
    assert set(np.unique(bond_orders)).issubset({1.0, 2.0, 3.0})  # Kekule, no 1.5


def test_chemical_features_require_rdkit():
    pytest.importorskip("rdkit")
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ["C", "O"],
        bond_index=[[0, 1]],
        bond_orders=[2],
    )
    features = mol.chemical_features()
    assert isinstance(features, ChemicalFeatures)
    np.testing.assert_array_equal(features.bond_orders, [2.0])
    np.testing.assert_array_equal(features.formal_charges, [0, 0])


def test_chemical_features_reports_aromaticity():
    pytest.importorskip("rdkit")
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    coords = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1)
    bonds = [[i, (i + 1) % 6] for i in range(6)]
    mol = Molecule(coords, ["C"] * 6, bond_index=bonds, bond_orders=[1.5] * 6)
    features = mol.chemical_features()
    assert features.aromatic_atoms.all()
    assert features.aromatic_bonds.all()


def test_rdkit_descriptors_by_name():
    pytest.importorskip("rdkit")
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ["C", "O"],
        bond_index=[[0, 1]],
        bond_orders=[2],
    )
    desc = mol.rdkit_descriptors(names=["MolWt", "TPSA"])
    assert desc["rdkit_MolWt"] > 0.0
    assert desc["rdkit_TPSA"] >= 0.0


def test_rdkit_descriptors_reject_unknown_name():
    pytest.importorskip("rdkit")
    mol = Molecule(np.zeros((1, 3)), ["C"])
    with pytest.raises(ValueError, match="unknown RDKit descriptor"):
        mol.rdkit_descriptors(names=["not_a_descriptor"])
