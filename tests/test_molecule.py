import sys

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule


def water():
    coords = [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    return Molecule(np.array(coords), ["O", "H", "H"], name="water")


@pytest.fixture
def no_scipy(monkeypatch):
    """Force the dense bond path by making scipy.spatial unimportable."""
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)


def test_construction_validates_element_count():
    with pytest.raises(ValueError):
        Molecule(np.zeros((3, 3)), ["O", "H"])


def test_construction_validates_formal_charge_count():
    with pytest.raises(ValueError):
        Molecule(np.zeros((3, 3)), ["O", "H", "H"], formal_charges=[0, 0])


def test_equality_by_value():
    a = Molecule(np.zeros((3, 3)), ["O", "H", "H"])
    b = Molecule(np.zeros((3, 3)), ["O", "H", "H"])
    c = Molecule(np.ones((3, 3)), ["O", "H", "H"])
    assert a == b
    assert a != c
    assert a != "not a molecule"


def test_take_subsets_formal_charges():
    mol = Molecule(np.zeros((3, 3)), ["N", "C", "O"], formal_charges=[1, 0, -1])
    np.testing.assert_array_equal(mol.take([0, 2]).formal_charges, [1, -1])


def test_instances_are_unhashable():
    with pytest.raises(TypeError):
        hash(water())


def test_translate_is_pure():
    mol = water()
    moved = mol.translate((1, 2, -1))
    np.testing.assert_allclose(moved.coords[0], [1, 2, -1])
    np.testing.assert_allclose(mol.coords[0], [0, 0, 0])  # original untouched


def test_centered_puts_centroid_at_origin():
    np.testing.assert_allclose(water().centered().centroid, [0, 0, 0], atol=1e-12)


def test_center_of_mass_pulled_toward_heavy_atom():
    mol = water()  # O sits at the origin, both H at y = 0.59
    assert mol.center_of_mass[1] < mol.centroid[1]


def test_radius_of_gyration_is_positive():
    assert water().radius_of_gyration > 0


def test_rotation_preserves_distances():
    mol = water()
    rotated = mol.rotate("z", 90)
    d_before = np.linalg.norm(mol.coords[1] - mol.coords[2])
    d_after = np.linalg.norm(rotated.coords[1] - rotated.coords[2])
    assert d_after == pytest.approx(d_before)


def test_full_turn_is_identity():
    mol = water()
    np.testing.assert_allclose(mol.rotate("y", 360).coords, mol.coords, atol=1e-9)


def test_rmsd_zero_for_identical():
    assert water().rmsd(water()) == pytest.approx(0.0)


def test_rmsd_align_recovers_rigid_motion():
    mol = water()
    moved = mol.translate((10, 5, -3)).rotate("y", 73)
    assert mol.rmsd(moved) > 1e-6                       # plain RMSD sees the motion
    assert mol.rmsd(moved, align=True) == pytest.approx(0.0, abs=1e-9)


def test_superpose_aligns_onto_reference():
    mol = water()
    moved = mol.translate((10, 5, -3)).rotate("y", 90)
    np.testing.assert_allclose(moved.superpose(mol).coords, mol.coords, atol=1e-9)


def test_rmsd_atom_count_mismatch():
    with pytest.raises(ValueError):
        water().rmsd(Molecule(np.zeros((2, 3)), ["H", "H"]))


def test_bonds_finds_the_two_oh_bonds():
    bonds = water().bonds()
    pairs = {frozenset(b) for b in bonds}
    assert pairs == {frozenset({0, 1}), frozenset({0, 2})}


def test_bonds_dense_path_matches(no_scipy):
    pairs = {frozenset(b) for b in water().bonds()}
    assert pairs == {frozenset({0, 1}), frozenset({0, 2})}


def test_contacts_path_matches(no_scipy):
    pairs = {frozenset(b) for b in water().contacts(cutoff=1.1)}
    assert pairs == {frozenset({0, 1}), frozenset({0, 2})}


def test_contact_count_path(no_scipy):
    assert water().contact_count(cutoff=1.1) == 2


def test_bonds_works_on_large_molecules_without_scipy(no_scipy):
    # Previously this would have failed with O(n^2) guard; now it uses O(n) cell list.
    big = Molecule(np.zeros((9000, 3)), ["C"] * 9000)
    bonds = big.bonds()
    # All at origin, so all are bonded (cutoff 1.2 * (0.77+0.77) = 1.848)
    # n*(n-1)/2 bonds.
    assert len(bonds) == 9000 * 8999 // 2



# -- rich residue identity --------------------------------------------------


def insertion_mol():
    """Two residues sharing chain/resid 100 but differing by insertion code."""
    return Molecule(
        np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]),
        ["C", "C", "C", "C"],
        atom_names=["CA", "CB", "CA", "CB"],
        resnames=["SER", "SER", "THR", "THR"],
        resids=np.array([100, 100, 100, 100]),
        icodes=["A", "A", "B", "B"],
        chains=["A", "A", "A", "A"],
    )


def no_icode_mol():
    """Residue metadata but no insertion codes."""
    return Molecule(
        np.array([[0.0, 0, 0], [1.0, 0, 0]]), ["C", "C"],
        atom_names=["CA", "CA"], resnames=["ALA", "GLY"],
        resids=np.array([1, 2]), chains=["A", "A"],
    )


def test_residue_id_label_str_and_ordering():
    rid = ms.ResidueId("A", 100, "B", "THR")
    assert rid.icode == "B"
    assert rid.label() == "A:THR100B"
    assert str(rid) == "A:THR100B"
    # No chain drops the leading "A:"; missing resname falls back to "RES".
    assert ms.ResidueId("", 5, "", "ALA").label() == "ALA5"
    assert ms.ResidueId("A", 7).label() == "A:RES7"
    # ``order=True`` makes ResidueId sortable.
    assert ms.ResidueId("A", 1) < ms.ResidueId("A", 2)


def test_residue_group_exposes_rich_id_and_tuple_api():
    groups = list(insertion_mol().residue_groups())
    assert len(groups) == 2
    first = groups[0]
    assert (first.resname, first.resid, first.chain) == ("SER", 100, "A")
    assert first.insertion_code == "A" and first.icode == "A"
    assert first.as_tuple() == ([0, 1], "SER", 100, "A")
    # Backwards-compatible unpacking / len / indexing.
    idx, resname, resid, chain = first
    assert idx == [0, 1] and resname == "SER" and resid == 100 and chain == "A"
    assert len(first) == 4
    assert first[1] == "SER"


def test_residue_ids_and_residue_id_lookup():
    mol = insertion_mol()
    assert [r.label() for r in mol.residue_ids] == [
        "A:SER100A", "A:SER100A", "A:THR100B", "A:THR100B",
    ]
    assert mol.residue_id(2).label() == "A:THR100B"
    # No residue metadata -> empty list and a clear error on single lookup.
    assert water().residue_ids == []
    with pytest.raises(ValueError, match="no residue-id"):
        water().residue_id(0)


def test_select_by_icode():
    mol = insertion_mol()
    assert len(mol.select(icode="A")) == 2
    assert mol.select(icode="B").resnames == ["THR", "THR"]
    assert len(mol.select(icode=["A", "B"])) == 4
    # A molecule without icodes: empty-string icode keeps everything.
    bare = no_icode_mol()
    assert len(bare.select(icode="")) == len(bare)
    with pytest.raises(ValueError, match="no insertion-code"):
        bare.select(icode="A")


def test_select_by_residue_id_selectors():
    mol = insertion_mol()
    # ResidueId and (chain, resid) tuple.
    assert len(mol.select(residue_id=ms.ResidueId("A", 100, "B"))) == 2
    assert len(mol.select(residue_id=("A", 100))) == 4
    # Tuple with icode and with icode + resname.
    assert len(mol.select(residue_id=("A", 100, "A"))) == 2
    assert len(mol.select(residue_id=("A", 100, "B", "THR"))) == 2
    assert len(mol.select(residue_id=("A", 100, "B", "SER"))) == 0
    # Dict selectors, including the ``icode``/``insertion_code`` aliases.
    assert len(mol.select(residue_id={"chain": "A", "resid": 100, "icode": "A"})) == 2
    assert len(
        mol.select(residue_id={"chain": "A", "resid": 100, "insertion_code": "B"})
    ) == 2
    # Resname match is case-insensitive.
    assert len(mol.select(residue_id={"chain": "A", "resid": 100, "resname": "thr"})) == 2
    # A list of selectors unions the matches.
    assert len(mol.select(residue_id=[("A", 100, "A"), ("A", 100, "B")])) == 4
    # Any object exposing ``.residue_id`` (e.g. a ResidueGroup) works.
    group = list(mol.residue_groups())[0]
    assert len(mol.select(residue_id=group)) == 2


def test_select_by_residue_id_rejects_bad_input():
    mol = insertion_mol()
    with pytest.raises(ValueError, match="require 'chain' and 'resid'"):
        mol.select(residue_id={"chain": "A"})
    with pytest.raises(ValueError, match="residue_id expects"):
        mol.select(residue_id=12345)
    # Selecting residue_id when the molecule lacks residue metadata.
    with pytest.raises(ValueError, match="no residue-id"):
        water().select(residue_id=("A", 1))
