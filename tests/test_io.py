import os

import numpy as np
import pytest

import molecule3d as m3d
from molecule3d.io import read_pdb, read_xyz

DATA = os.path.dirname(os.path.dirname(__file__))

PDB_ATOM = (
    "ATOM      1  N   ASP A   1       8.950   5.340  -7.914  1.00  0.00           N  \n"
)


def test_read_xyz_bare_coordinates():
    mol = read_xyz(os.path.join(DATA, "helix_201.xyz"))
    assert len(mol) == 201
    assert mol.coords.shape == (201, 3)
    np.testing.assert_allclose(mol.coords[0], [0.3517846, -0.7869986, -2.873479])


def test_read_standard_xyz_with_elements(tmp_path):
    f = tmp_path / "water.xyz"
    f.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.76 0.59 0.0\nH -0.76 0.59 0.0\n")
    mol = read_xyz(str(f))
    assert len(mol) == 3
    assert mol.elements == ["O", "H", "H"]


def test_read_pdb_fixed_columns(tmp_path):
    f = tmp_path / "one.pdb"
    f.write_text(PDB_ATOM)
    mol = read_pdb(str(f))
    assert len(mol) == 1
    np.testing.assert_allclose(mol.coords[0], [8.950, 5.340, -7.914])
    assert mol.elements == ["N"]


def test_pdb_columns_beat_whitespace_split(tmp_path):
    """Touching coordinate fields fool str.split but not column slicing."""
    line = (
        "ATOM      1  CA  ARG A 100    -123.456-789.012  12.345  1.00  0.00           C  \n"
    )
    f = tmp_path / "tight.pdb"
    f.write_text(line)
    mol = read_pdb(str(f))
    # Whitespace splitting would merge "-123.456-789.012" into one bad token.
    np.testing.assert_allclose(mol.coords[0], [-123.456, -789.012, 12.345])


def test_read_pdb_single_model_from_multimodel():
    mol = read_pdb(os.path.join(DATA, "1aml.pdb"), model=1)
    # 1aml has 20 NMR models; model 1 must be a slice of the 11960 total atoms.
    assert 0 < len(mol) < 11960


def test_read_dispatches_on_extension():
    assert len(m3d.read(os.path.join(DATA, "1fqy.pdb"))) == 1661
    with pytest.raises(ValueError):
        m3d.read("structure.mol2")
