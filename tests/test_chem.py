"""Tests for optional RDKit-backed chemical perception."""

import numpy as np
import pytest

from molscope import ChemicalFeatures, Molecule


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
