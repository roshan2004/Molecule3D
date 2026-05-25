"""Tests for metadata/selections, geometry, ensemble analysis, I/O and viz."""

import gzip
import os

import numpy as np
import pytest

import molecule3d as m3d
from molecule3d import Molecule, ensemble

DATA = os.path.dirname(os.path.dirname(__file__))


# -- selections / metadata --------------------------------------------------


def test_pdb_carries_metadata():
    mol = m3d.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    assert mol.has_topology
    assert mol.atom_names[0] == "N"
    assert mol.resnames[0] == "LYS"
    assert mol.chains[0] == "A"
    assert mol.resids[0] == 8


def test_select_by_element_and_atom_name():
    mol = m3d.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    carbons = mol.select(element="C")
    assert all(e == "C" for e in carbons.elements)
    assert 0 < len(carbons) < len(mol)

    ca = mol.alpha_carbons()
    assert all(n == "CA" for n in ca.atom_names)
    assert 0 < len(ca) <= len(mol.backbone())


def test_select_without_metadata_raises():
    xyz = m3d.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.raises(ValueError):
        xyz.select(chain="A")


def test_getitem_subsets():
    mol = m3d.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    first10 = mol[np.arange(10)]
    assert len(first10) == 10
    assert first10.atom_names == mol.atom_names[:10]


def test_ca_rmsd_smaller_atom_count_than_all_atom():
    models = m3d.read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    ca0, ca1 = models[0].alpha_carbons(), models[1].alpha_carbons()
    assert len(ca0) < len(models[0])
    assert ca0.rmsd(ca1, align=True) >= 0.0


# -- geometry ---------------------------------------------------------------


def water():
    return Molecule(
        np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        ["O", "H", "H"],
    )


def test_distance_angle_dihedral():
    mol = water()
    assert mol.distance(0, 1) == pytest.approx(0.96)
    assert mol.angle(1, 0, 2) == pytest.approx(104.0, abs=1.0)
    square = Molecule(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float),
        ["C"] * 4,
    )
    assert abs(square.dihedral(0, 1, 2, 3)) == pytest.approx(90.0, abs=1e-6)


def test_distance_matrix_and_contacts():
    mol = water()
    dm = mol.distance_matrix()
    assert dm.shape == (3, 3)
    assert np.allclose(np.diag(dm), 0)
    assert len(mol.contacts(cutoff=2.0)) == 3  # all three pairs are within 2 A


def test_dimensions_and_formula():
    mol = Molecule(np.array([[0, 0, 0], [3, 0, 0]], dtype=float), ["O", "H"])
    np.testing.assert_allclose(mol.dimensions, [3, 0, 0])
    assert water().formula == "H2 O"


def test_summary_runs():
    assert "atoms" in m3d.read_pdb(os.path.join(DATA, "1fqy.pdb")).summary()


# -- ensemble ---------------------------------------------------------------


def test_ensemble_average_and_rmsf():
    models = m3d.read_pdb_models(os.path.join(DATA, "1aml.pdb"))[:5]
    avg = ensemble.average(models)
    assert len(avg) == len(models[0])
    fluct = ensemble.rmsf(models)
    assert fluct.shape == (len(models[0]),)
    assert (fluct >= 0).all()


def test_ensemble_rmsd_matrix():
    models = m3d.read_pdb_models(os.path.join(DATA, "1aml.pdb"))[:4]
    mat = ensemble.rmsd_matrix(models)
    assert mat.shape == (4, 4)
    assert np.allclose(np.diag(mat), 0)
    assert np.allclose(mat, mat.T)


# -- I/O: gzip, cif, sdf, multi-frame xyz, fetch ----------------------------


def test_gzip_transparent_read(tmp_path):
    src = os.path.join(DATA, "1fqy.pdb")
    gz = tmp_path / "1fqy.pdb.gz"
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        fout.write(fin.read())
    assert len(m3d.read(str(gz))) == 1661


CIF = """\
data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 N N ALA A 1 1.000 2.000 3.000
ATOM 2 C CA ALA A 1 4.000 5.000 6.000
#
"""


def test_read_cif(tmp_path):
    f = tmp_path / "t.cif"
    f.write_text(CIF)
    mol = m3d.read(str(f))
    assert len(mol) == 2
    assert mol.elements == ["N", "C"]
    assert mol.atom_names == ["N", "CA"]
    np.testing.assert_allclose(mol.coords[1], [4, 5, 6])


SDF = """\
water
  example

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0
    0.9600    0.0000    0.0000 H   0  0
   -0.2400    0.9300    0.0000 H   0  0
  1  2  1  0
  1  3  1  0
M  END
$$$$
"""


def test_read_sdf(tmp_path):
    f = tmp_path / "w.sdf"
    f.write_text(SDF)
    mol = m3d.read(str(f))
    assert mol.elements == ["O", "H", "H"]
    np.testing.assert_allclose(mol.coords[1], [0.96, 0.0, 0.0])


def test_multi_frame_xyz(tmp_path):
    frame = "2\nframe\nH 0 0 0\nH 0 0 1\n"
    f = tmp_path / "traj.xyz"
    f.write_text(frame * 3)
    frames = m3d.read_xyz_frames(str(f))
    assert len(frames) == 3
    assert all(len(fr) == 2 for fr in frames)


def test_fetch_uses_downloader(tmp_path, monkeypatch):
    import molecule3d.io as mio

    with open(os.path.join(DATA, "1fqy.pdb"), "rb") as fh:
        sample = fh.read()

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return sample

    monkeypatch.setattr(mio, "urlopen", lambda url: FakeResp())
    mol = m3d.fetch("1fqy", cache_dir=str(tmp_path))
    assert len(mol) == 1661


# -- visualization ----------------------------------------------------------


def test_py3dmol_view_builds():
    mol = water()
    viewer = mol.view()
    assert viewer is not None


def test_spin_gif(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from molecule3d.plotting import spin_gif

    out = str(tmp_path / "spin.gif")
    spin_gif(water(), out, frames=4, fps=5)
    assert os.path.getsize(out) > 0
