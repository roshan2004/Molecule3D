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


def alanine_with_hydrogens():
    """One ALA residue with hydrogens that Martini intentionally drops."""
    names = ["N", "H", "CA", "HA", "C", "O", "CB", "HB1", "HB2", "HB3"]
    els = ["N", "H", "C", "H", "C", "O", "C", "H", "H", "H"]
    return Molecule(
        np.arange(30).reshape(10, 3).astype(float),
        els,
        name="alanine",
        atom_names=names,
        resnames=["ALA"] * 10,
        resids=np.array([1] * 10),
        chains=["A"] * 10,
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


def test_mapping_report_explains_martini_mapping():
    cg = alanine_with_hydrogens().coarse_grain("martini")
    report = cg.mapping_report()
    assert "Mapping: martini" in report
    assert "BB bead: N, CA, C, O -> centre of mass" in report
    assert "SC bead: CB -> centre of mass" in report
    assert "Residue 1 ALA A: H" in report
    assert "Residue 1 ALA A: HB3" in report
    assert "BB-SC within residue" in report


def test_custom_mapping():
    mapping = {"ALA": {"BB": ["N", "CA", "C", "O"], "SC": ["CB"]}}
    cg = two_alanines().coarse_grain(mapping)
    assert len(cg) == 4
    assert cg.atom_names == ["BB", "SC", "BB", "SC"]


def test_return_report_gives_structured_report_object():
    mapping = {"ALA": {"BB": ["N", "CA", "C", "O"]}}
    with pytest.warns(UserWarning, match="not assigned"):
        cg, report = alanine_with_hydrogens().coarse_grain(mapping, return_report=True)
    assert isinstance(cg, Molecule)
    assert isinstance(report, m3d.CoarseGrainReport)
    assert report.beads[0].name == "BB"
    assert report.beads[0].atom_names == ["N", "CA", "C", "O"]
    assert [atom.name for atom in report.dropped_atoms] == ["H", "HA", "CB", "HB1", "HB2", "HB3"]
    assert cg.mapping_report() == report.format()


def test_mapping_report_missing_on_plain_or_subset_molecule():
    mol = two_alanines()
    with pytest.raises(ValueError):
        mol.mapping_report()
    cg = mol.coarse_grain("martini")
    assert "Mapping: martini" in cg.mapping_report()
    with pytest.raises(ValueError):
        cg.take([0]).mapping_report()


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


def test_index_mapping_works_without_residues():
    helix = m3d.read(os.path.join(DATA, "helix_201.xyz"))  # no residue/atom names
    cg = helix.coarse_grain({"A": list(range(100)), "B": list(range(100, 201))})
    assert len(cg) == 2
    assert cg.atom_names == ["A", "B"]
    # bead A sits at the centroid of the first 100 atoms (equal masses here)
    np.testing.assert_allclose(cg.coords[0], helix.coords[:100].mean(axis=0))


def test_user_defined_bonds_by_name_and_index():
    helix = m3d.read(os.path.join(DATA, "helix_201.xyz"))
    by_name = helix.coarse_grain(
        {"A": [0, 1], "B": [2, 3], "C": [4, 5]}, bonds=[("A", "B"), ("B", "C")]
    )
    np.testing.assert_array_equal(by_name.bonds(), [[0, 1], [1, 2]])
    by_index = helix.coarse_grain(
        {"A": [0, 1], "B": [2, 3], "C": [4, 5]}, bonds=[(0, 2)]
    )
    np.testing.assert_array_equal(by_index.bonds(), [[0, 2]])


def test_user_defined_bonds_reject_repeated_bead_names():
    with pytest.raises(ValueError, match="repeated"):
        two_alanines().coarse_grain("martini", bonds=[("BB", "SC")])


def test_user_defined_bonds_reject_unknown_bead_name():
    helix = m3d.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.raises(ValueError, match="unknown bead name"):
        helix.coarse_grain({"A": [0, 1], "B": [2, 3]}, bonds=[("A", "missing")])


def test_index_mapping_rejects_atom_names():
    helix = m3d.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.raises(ValueError):
        helix.coarse_grain({"BB": ["N", "CA"]})  # names, not indices


def test_warns_on_unassigned_atoms():
    # The mapping omits CB, so one atom per residue is dropped.
    with pytest.warns(UserWarning, match="not assigned"):
        cg = two_alanines().coarse_grain({"ALA": {"BB": ["N", "CA", "C", "O"]}})
    assert len(cg) == 2


def test_user_defined_bonds_override_residue_topology():
    cg = two_alanines().coarse_grain("residue_com", bonds=[(0, 1)])
    np.testing.assert_array_equal(cg.bonds(), [[0, 1]])


def test_cg_result_is_a_usable_molecule():
    mol = m3d.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    cg = mol.coarse_grain("residue_com")
    assert isinstance(cg, Molecule)
    assert len(cg) == 226              # one bead per residue
    g = cg.to_graph()                  # the CG model graphs like any molecule
    assert g.n_atoms == 226
    assert g.n_bonds == len(cg.bonds())
