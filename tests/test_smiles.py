"""Tests for ``read_smiles`` (SMILES -> Molecule via an RDKit-generated conformer).

Skipped without RDKit, which the feature requires.
"""

import numpy as np
import pytest

import molscope as ms

pytest.importorskip("rdkit")


def test_read_smiles_builds_molecule_with_topology():
    mol = ms.read_smiles("CCO")  # ethanol
    assert len(mol) == 9  # C2 O1 + 6 H
    assert mol.coords.shape == (9, 3)
    assert sorted(set(mol.elements)) == ["C", "H", "O"]
    assert mol.bond_index is not None and len(mol.bond_index) == 8
    assert "CCO" in mol.name  # provenance recorded


def test_read_smiles_perceives_aromaticity():
    mol = ms.read_smiles("c1ccccc1")  # benzene
    aromatic = int(mol.chemical_features().aromatic_atoms.sum())
    assert aromatic == 6


def test_read_smiles_bond_orders_are_kekule():
    mol = ms.read_smiles("c1ccccc1")
    assert set(np.unique(mol.bond_orders)).issubset({1.0, 2.0})  # no aromatic 1.5


def test_read_smiles_feeds_the_graph_workflow():
    graph = ms.read_smiles("CC(=O)O").to_graph()  # acetic acid
    assert graph.n_atoms == 8
    assert graph.n_bonds > 0


def test_read_smiles_add_hs_false_drops_hydrogens():
    mol = ms.read_smiles("c1ccccc1", add_hs=False)
    assert len(mol) == 6  # heavy atoms only
    assert "H" not in mol.elements


def test_read_smiles_is_reproducible_with_seed():
    a = ms.read_smiles("CCO", seed=7)
    b = ms.read_smiles("CCO", seed=7)
    np.testing.assert_allclose(a.coords, b.coords)


def test_read_smiles_formal_charges_carry_over():
    mol = ms.read_smiles("CC(=O)[O-]")  # acetate
    assert int(mol.formal_charges.sum()) == -1


def test_read_smiles_rejects_invalid():
    with pytest.raises(ValueError, match="invalid SMILES"):
        ms.read_smiles("not-a-smiles$$")
