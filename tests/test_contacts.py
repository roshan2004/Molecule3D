"""Tests for chain interfaces and ligand-binding-site contacts."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def two_chain():
    """Chain A (res 1 at x=0, res 2 at x=4) and chain B (res 1 at x=6).

    A-res2 and B-res1 sit 2 A apart; A-res1 is 6 A from B.
    """
    return Molecule(
        np.array([[0.0, 0, 0], [4.0, 0, 0], [6.0, 0, 0]]), ["C", "C", "C"],
        atom_names=["CA", "CA", "CA"], resnames=["ALA", "ALA", "GLU"],
        resids=np.array([1, 2, 1]), chains=["A", "A", "B"],
    )


def protein_with_ligand():
    """Two protein residues + a LIG ligand atom + a water, with hetero flags.

    Protein res 1 (x=0) is 2 A from the ligand (x=2); res 2 (x=10) is far.
    """
    return Molecule(
        np.array([[0.0, 0, 0], [10.0, 0, 0], [2.0, 0, 0], [20.0, 0, 0]]),
        ["C", "C", "N", "O"],
        atom_names=["CA", "CA", "N1", "O"],
        resnames=["ALA", "GLY", "LIG", "HOH"],
        resids=np.array([1, 2, 100, 200]),
        chains=["A", "A", "A", "A"],
        hetero=[False, False, True, True],
    )


# -- interfaces -------------------------------------------------------------


def test_interface_residues_finds_contacting_residues():
    iface = two_chain().interface("A", "B", cutoff=5.0)
    assert iface.n_atom_contacts == 1
    assert [r.resid for r in iface.residues_a] == [2]      # only A-res2 is close
    assert [r.resid for r in iface.residues_b] == [1]
    assert iface.contacts == [(1, 2)]                       # atom 1 (A) - atom 2 (B)


def test_interface_unknown_chain_raises():
    with pytest.raises(ValueError):
        two_chain().interface("A", "Z")


def test_interface_requires_chain_metadata():
    bare = Molecule(np.zeros((2, 3)), ["C", "C"])
    with pytest.raises(ValueError):
        bare.interface("A", "B")


def test_chain_contact_matrix_is_symmetric_counts():
    ccm = two_chain().chain_contacts(cutoff=5.0)
    assert ccm.chains == ["A", "B"]
    assert ccm.matrix.shape == (2, 2)
    assert np.array_equal(ccm.matrix, ccm.matrix.T)
    assert np.diag(ccm.matrix).sum() == 0
    assert ccm.count("A", "B") == 1


# -- ligands & binding sites ------------------------------------------------


def test_ligands_excludes_water_and_ions():
    mol = protein_with_ligand()
    ligs = mol.ligands()
    assert [g.resname for g in ligs] == ["LIG"]             # HOH filtered out
    assert mol.ligands(exclude_water=False)                 # water kept when asked
    assert any(g.resname == "HOH" for g in mol.ligands(exclude_water=False))


def test_binding_site_synthetic():
    bs = protein_with_ligand().binding_site(cutoff=4.5)
    assert bs.ligand.resname == "LIG"
    assert [r.resid for r in bs.residues] == [1]            # only res 1 is within 4.5 A
    assert bs.min_distances[0] == pytest.approx(2.0)
    assert all(p < q for p, q in zip(bs.min_distances, bs.min_distances[1:]))
    assert bs.n_atom_contacts == 1
    assert bs.residue_contact_counts == [1]
    assert bs.to_records() == [{
        "chain": "A",
        "resid": 1,
        "resname": "ALA",
        "min_distance": 2.0,
        "n_atom_contacts": 1,
    }]


def test_binding_site_can_select_full_site_residues():
    mol = Molecule(
        np.array([[0.0, 0, 0], [8.0, 0, 0], [10.0, 0, 0], [2.0, 0, 0]]),
        ["C", "C", "C", "N"],
        atom_names=["CA", "CB", "CA", "N1"],
        resnames=["ALA", "ALA", "GLY", "LIG"],
        resids=np.array([1, 1, 2, 100]),
        chains=["A", "A", "A", "A"],
        hetero=[False, False, False, True],
    )
    bs = mol.binding_site(cutoff=4.5)
    assert bs.contact_atom_indices == [0]
    assert bs.protein_atom_indices == [0, 1]
    assert len(bs.to_molecule(mol)) == 2
    assert len(bs.to_molecule(mol, include_ligand=True)) == 3


def test_binding_site_pocket_basic_descriptors_are_fixed_size():
    mol = protein_with_ligand()
    bs = mol.binding_site(cutoff=4.5)
    desc = bs.descriptors(mol)
    assert set(desc) == set(ms.pocket_descriptor_feature_names("pocket-basic"))
    assert desc["pocket_n_atoms"] == 1.0
    assert desc["pocket_n_residues"] == 1.0
    assert desc["ligand_n_atoms"] == 1.0
    assert desc["pocket_atom_contact_count"] == 1.0
    assert desc["pocket_residue_count_ALA"] == 1.0
    assert desc["ligand_distance_min"] == pytest.approx(2.0)
    assert desc["ligand_contact_distance_mean"] == pytest.approx(2.0)


def test_binding_site_plot_shortcut_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    mol = protein_with_ligand()
    ax = mol.binding_site(cutoff=4.5).plot(mol, show=False)
    assert ax is not None


def test_binding_site_ambiguous_requires_explicit_ligand():
    mol = Molecule(
        np.array([[0.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]),
        ["C", "N", "N"], atom_names=["CA", "N1", "N1"],
        resnames=["ALA", "LIG", "FAD"], resids=np.array([1, 100, 101]),
        chains=["A", "A", "A"], hetero=[False, True, True],
    )
    with pytest.raises(ValueError):
        mol.binding_site()                                  # two candidate ligands
    bs = mol.binding_site(ligand="LIG", cutoff=4.5)
    assert bs.ligand.resname == "LIG"


def test_binding_site_no_hetatm_raises():
    bare = Molecule(
        np.zeros((2, 3)), ["C", "C"], atom_names=["CA", "CA"],
        resnames=["ALA", "ALA"], resids=np.array([1, 2]), chains=["A", "A"],
    )
    with pytest.raises(ValueError):
        bare.binding_site()


def test_binding_site_on_trypsin_benzamidine():
    mol = ms.read(os.path.join(DATA, "3ptb.pdb"))
    ligs = mol.ligands()                                    # CA ion + waters excluded
    assert [g.resname for g in ligs] == ["BEN"]
    bs = mol.binding_site(cutoff=4.5)
    resids = {r.resid for r in bs.residues}
    # The benzamidine S1 pocket: Asp189 (specificity), Ser190, Gly219, Ser195.
    assert {189, 190, 195, 219} <= resids
    assert bs.min_distances == sorted(bs.min_distances)
