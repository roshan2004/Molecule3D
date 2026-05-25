"""Tests for explicit bonds and coarse-graining."""

import os

import numpy as np
import pytest

import molecule3d as m3d
from molecule3d import Molecule

DATA = os.path.dirname(os.path.dirname(__file__))


def two_alanines(second_chain="A"):
    """Two ALA residues (N, CA, C, O, CB each) in one or two chains."""
    names = ["N", "CA", "C", "O", "CB"] * 2
    els = ["N", "C", "C", "O", "C"] * 2
    return Molecule(
        np.arange(30).reshape(10, 3).astype(float),
        els,
        name="dialanine",
        atom_names=names,
        resnames=["ALA"] * 10,
        resids=np.array([1] * 5 + [2] * 5),
        chains=["A"] * 5 + [second_chain] * 5,
    )


# -- explicit bonds ---------------------------------------------------------


def test_explicit_bonds_returned():
    m = Molecule(np.zeros((3, 3)), ["C", "C", "C"], bond_index=[[0, 1], [1, 2]])
    np.testing.assert_array_equal(m.bonds(), [[0, 1], [1, 2]])


def test_explicit_bonds_survive_transforms():
    m = Molecule(np.eye(3), ["C", "C", "C"], bond_index=[[0, 1]])
    np.testing.assert_array_equal(m.translate((1, 1, 1)).bonds(), [[0, 1]])
    np.testing.assert_array_equal(m.rotate("z", 30).bonds(), [[0, 1]])


def test_explicit_bonds_remap_on_subset():
    m = Molecule(np.arange(12).reshape(4, 3), ["C"] * 4,
                 bond_index=[[0, 1], [1, 2], [2, 3]])
    sub = m.take([1, 2, 3])  # drop atom 0; (0,1) gone, (1,2)->(0,1), (2,3)->(1,2)
    np.testing.assert_array_equal(sub.bonds(), [[0, 1], [1, 2]])


# -- coarse-graining --------------------------------------------------------


def test_residue_com_one_bead_per_residue():
    cg = two_alanines().coarse_grain("residue_com")
    assert len(cg) == 2
    assert cg.atom_names == ["ALA", "ALA"]
    np.testing.assert_array_equal(cg.bonds(), [[0, 1]])  # backbone chain link


def test_com_is_mass_weighted_centroid_is_not():
    from molecule3d import elements

    mol = Molecule(
        np.array([[0.0, 0, 0], [2.0, 0, 0]]), ["O", "H"],
        atom_names=["O", "H"], resnames=["HOH", "HOH"],
        resids=np.array([1, 1]), chains=["A", "A"],
    )
    com = mol.coarse_grain("residue_com").coords[0, 0]
    centroid = mol.coarse_grain("residue_centroid").coords[0, 0]
    expected_com = 2.0 * elements.mass("H") / (elements.mass("O") + elements.mass("H"))
    assert com == pytest.approx(expected_com)   # pulled toward the heavy O
    assert centroid == pytest.approx(1.0)
    assert com < centroid


def test_martini_backbone_and_sidechain():
    cg = two_alanines().coarse_grain("martini")
    assert cg.atom_names == ["BB", "SC", "BB", "SC"]
    # intra-residue BB-SC bonds plus a BB-BB link between the two residues
    pairs = {frozenset(b) for b in cg.bonds()}
    assert pairs == {frozenset({0, 1}), frozenset({2, 3}), frozenset({0, 2})}


def test_custom_mapping():
    mapping = {"ALA": {"BB": ["N", "CA", "C", "O"], "SC": ["CB"]}}
    cg = two_alanines().coarse_grain(mapping)
    assert len(cg) == 4
    assert cg.atom_names == ["BB", "SC", "BB", "SC"]


def test_unmapped_residue_warns_and_collapses():
    with pytest.warns(UserWarning):
        cg = two_alanines().coarse_grain({"GLY": {"BB": ["CA"]}})
    assert len(cg) == 2  # both ALA residues collapse to one bead each


def test_separate_chains_not_linked():
    cg = two_alanines(second_chain="B").coarse_grain("residue_com")
    assert len(cg) == 2
    assert len(cg.bonds()) == 0  # different chains -> no backbone bond


def test_coarse_grain_requires_residues():
    xyz = m3d.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.raises(ValueError):
        xyz.coarse_grain()


def test_cg_result_is_a_usable_molecule():
    mol = m3d.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    cg = mol.coarse_grain("residue_com")
    assert isinstance(cg, Molecule)
    assert len(cg) == 226              # one bead per residue
    g = cg.to_graph()                  # the CG model graphs like any molecule
    assert g.n_atoms == 226
    assert g.n_bonds == len(cg.bonds())
