"""Tests for optional Gemmi-backed CIF validation."""

import numpy as np
import pytest

import molscope as ms

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


def test_validate_cif_with_gemmi(tmp_path):
    pytest.importorskip("gemmi")
    path = tmp_path / "ok.cif"
    path.write_text(CIF)

    report = ms.validate_cif(str(path))
    assert report.valid
    assert report.syntax_ok
    assert report.atom_site_ok
    assert report.n_blocks == 1
    assert report.n_atom_site_rows == 2
    assert not report.dictionary_checked


def test_validate_cif_reports_missing_atom_site_columns(tmp_path):
    pytest.importorskip("gemmi")
    path = tmp_path / "bad.cif"
    path.write_text("""\
data_BAD
loop_
_atom_site.group_PDB
_atom_site.Cartn_x
_atom_site.Cartn_y
ATOM 1 2
#
""")

    report = ms.validate_cif(str(path))
    assert not report.valid
    assert report.syntax_ok
    assert not report.atom_site_ok
    assert any("missing atom-site column" in error for error in report.errors)


def test_read_cif_with_gemmi_parser(tmp_path):
    pytest.importorskip("gemmi")
    path = tmp_path / "ok.cif"
    path.write_text(CIF)

    mol = ms.read_cif(str(path), parser="gemmi")
    assert len(mol) == 2
    assert mol.atom_names == ["N", "CA"]
    np.testing.assert_allclose(mol.coords[1], [4.0, 5.0, 6.0])
    assert mol.hetero == [False, False]


CIF_WITH_HETATM = """\
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
HETATM 2 O O HOH A 2 4.000 5.000 6.000
#
"""


def test_read_cif_builtin_parser_reads_group_pdb(tmp_path):
    path = tmp_path / "het.cif"
    path.write_text(CIF_WITH_HETATM)

    mol = ms.read_cif(str(path))  # builtin parser, no gemmi needed
    assert mol.hetero == [False, True]
    assert mol.hetero_atoms().resnames == ["HOH"]


def test_read_cif_gemmi_parser_reads_group_pdb(tmp_path):
    pytest.importorskip("gemmi")
    path = tmp_path / "het.cif"
    path.write_text(CIF_WITH_HETATM)

    mol = ms.read_cif(str(path), parser="gemmi")
    assert mol.hetero == [False, True]


CIF_WITH_CELL = """\
data_TEST
_cell.length_a 10.000
_cell.length_b 11.000
_cell.length_c 12.000
_cell.angle_alpha 90.000
_cell.angle_beta 90.000
_cell.angle_gamma 120.000
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
#
"""


def test_read_cif_gemmi_parser_reads_unit_cell(tmp_path):
    pytest.importorskip("gemmi")
    path = tmp_path / "cell.cif"
    path.write_text(CIF_WITH_CELL)

    mol = ms.read_cif(str(path), parser="gemmi")
    assert mol.unit_cell is not None
    np.testing.assert_allclose(
        [mol.unit_cell.a, mol.unit_cell.b, mol.unit_cell.c], [10.0, 11.0, 12.0]
    )
    assert mol.unit_cell.gamma == 120.0


def test_read_cif_gemmi_parser_without_cell_leaves_none(tmp_path):
    pytest.importorskip("gemmi")
    path = tmp_path / "ok.cif"
    path.write_text(CIF)  # no _cell.* items

    mol = ms.read_cif(str(path), parser="gemmi")
    assert mol.unit_cell is None
