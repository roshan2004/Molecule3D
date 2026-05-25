import os

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule
from molscope.io import read_pdb, read_pdb_models, read_xyz, write_pdb, write_xyz

DATA = os.path.dirname(os.path.dirname(__file__))


def _atom_line(serial, altloc, x, y, z, element="N"):
    """Build a fixed-column ATOM record with an explicit altLoc at column 17."""
    line = list(" " * 80)

    def put(value, start):
        line[start - 1:start - 1 + len(value)] = value

    put("ATOM", 1)
    put(f"{serial:>5}", 7)
    put(" N  ", 13)
    put(altloc, 17)
    put("ASP", 18)
    put("A", 22)
    put(f"{1:>4}", 23)
    put(f"{x:8.3f}", 31)
    put(f"{y:8.3f}", 39)
    put(f"{z:8.3f}", 47)
    put(f"{element:>2}", 77)
    return "".join(line).rstrip() + "\n"

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
    assert len(ms.read(os.path.join(DATA, "1fqy.pdb"))) == 1661
    with pytest.raises(ValueError):
        ms.read("structure.mol2")


def test_pdb_skips_alternate_conformations(tmp_path):
    f = tmp_path / "alt.pdb"
    f.write_text(
        _atom_line(1, "A", 1.0, 2.0, 3.0) + _atom_line(2, "B", 9.0, 9.0, 9.0)
    )
    mol = read_pdb(str(f))
    assert len(mol) == 1  # only the primary "A" conformation is kept
    np.testing.assert_allclose(mol.coords[0], [1.0, 2.0, 3.0])


def test_read_all_nmr_models():
    models = read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    assert len(models) == 20
    assert {len(m) for m in models} == {len(models[0])}  # consistent atom count
    assert models[1].name.endswith("#2")


def test_rmsd_between_nmr_models():
    models = read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    assert models[0].rmsd(models[1], align=True) >= 0.0


def test_read_pdb_model_out_of_range(tmp_path):
    f = tmp_path / "one.pdb"
    f.write_text(PDB_ATOM)
    with pytest.raises(ValueError):
        read_pdb(str(f), model=2)


def test_xyz_round_trip(tmp_path):
    mol = Molecule(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ["O", "H"], name="t")
    path = str(tmp_path / "rt.xyz")
    write_xyz(mol, path)
    back = read_xyz(path)
    np.testing.assert_allclose(back.coords, mol.coords)
    assert back.elements == ["O", "H"]


def test_pdb_round_trip(tmp_path):
    mol = Molecule(np.array([[1.234, -5.678, 9.012], [0.0, 0.0, 0.0]]), ["C", "O"])
    path = str(tmp_path / "rt.pdb")
    write_pdb(mol, path)
    back = read_pdb(path)
    np.testing.assert_allclose(back.coords, mol.coords, atol=1e-3)
    assert back.elements == ["C", "O"]
