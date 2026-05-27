import sys

import numpy as np
import pytest

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

