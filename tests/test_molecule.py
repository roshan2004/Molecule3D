import numpy as np
import pytest

from molecule3d import Molecule


def water():
    coords = [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    return Molecule(np.array(coords), ["O", "H", "H"], name="water")


def test_construction_validates_element_count():
    with pytest.raises(ValueError):
        Molecule(np.zeros((3, 3)), ["O", "H"])


def test_translate_is_pure():
    mol = water()
    moved = mol.translate((1, 2, -1))
    np.testing.assert_allclose(moved.coords[0], [1, 2, -1])
    # original is untouched
    np.testing.assert_allclose(mol.coords[0], [0, 0, 0])


def test_centered_puts_centroid_at_origin():
    mol = water().centered()
    np.testing.assert_allclose(mol.centroid, [0, 0, 0], atol=1e-12)


def test_rotation_preserves_distances():
    mol = water()
    rotated = mol.rotate("z", 90)
    d_before = np.linalg.norm(mol.coords[1] - mol.coords[2])
    d_after = np.linalg.norm(rotated.coords[1] - rotated.coords[2])
    assert d_after == pytest.approx(d_before)


def test_full_turn_is_identity():
    mol = water()
    np.testing.assert_allclose(mol.rotate("y", 360).coords, mol.coords, atol=1e-9)


def test_bonds_finds_the_two_oh_bonds():
    bonds = water().bonds()
    assert len(bonds) == 2
    assert {0, 1} in [set(b) for b in bonds]
    assert {0, 2} in [set(b) for b in bonds]


def test_bonds_guards_against_huge_molecules():
    big = Molecule(np.random.rand(10, 3), ["C"] * 10)
    with pytest.raises(ValueError):
        big.bonds(max_atoms=5)
