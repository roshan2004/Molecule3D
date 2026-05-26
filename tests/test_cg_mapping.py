"""Tests for the coarse-graining workflow: bead assignment, mapping
visualisation, and mapping export / round-trip."""

import json
import os

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


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


# -- bead assignment --------------------------------------------------------


def test_beads_record_their_source_atom_indices():
    cg, report = two_alanines().coarse_grain("martini", return_report=True)
    # martini -> BB(N,CA,C,O) + SC(CB) per residue, in atom order
    assert [b.atom_indices for b in report.beads] == [[0, 1, 2, 3], [4], [5, 6, 7, 8], [9]]
    # atom_names stay aligned with atom_indices
    assert report.beads[0].atom_names == ["N", "CA", "C", "O"]


def test_index_mapping_records_indices():
    helix = ms.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.warns(UserWarning, match="not assigned"):  # most atoms unmapped here
        cg = helix.coarse_grain({"A": [0, 1, 2], "B": [3, 4]})
    assert [b.atom_indices for b in cg.coarse_grain_report.beads] == [[0, 1, 2], [3, 4]]


def test_report_coverage_counts():
    with pytest.warns(UserWarning, match="not assigned"):
        cg = two_alanines().coarse_grain({"ALA": {"BB": ["N", "CA", "C", "O"]}})
    report = cg.coarse_grain_report
    assert (report.n_beads, report.n_assigned, report.n_dropped) == (2, 8, 2)
    assert report.coverage() == "2 beads from 8/10 atoms (2 dropped)"
    assert report.coverage() in report.format()


def test_coarse_grain_report_property_requires_cg_molecule():
    with pytest.raises(ValueError):
        _ = two_alanines().coarse_grain_report


# -- mapping export / round-trip -------------------------------------------


def test_mapping_to_dict_shape():
    cg = two_alanines().coarse_grain("martini")
    rec = ms.cg_mapping_to_dict(cg)
    assert rec["format"] == "molscope-cg-mapping"
    assert rec["mapping"] == "martini"
    assert rec["n_beads"] == 4 and rec["n_atoms_assigned"] == 10
    first = rec["beads"][0]
    assert first["name"] == "BB" and first["atom_indices"] == [0, 1, 2, 3]
    assert len(first["position"]) == 3
    assert [tuple(b) for b in rec["bonds"]] == [(0, 1), (2, 3), (0, 2)]


def test_mapping_export_omits_inferred_bead_bonds():
    # An index mapping with no explicit bonds must export none, not geometry.
    helix = ms.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.warns(UserWarning, match="not assigned"):
        cg = helix.coarse_grain({"a": [0, 1], "b": [2, 3]})
    assert cg.bond_index is None
    assert ms.cg_mapping_to_dict(cg)["bonds"] == []


def test_write_then_read_mapping_round_trips_on_disk(tmp_path):
    cg = two_alanines().coarse_grain("martini")
    path = tmp_path / "map.json"
    ms.write_cg_mapping(cg, path)
    record = ms.read_cg_mapping(path)
    assert record == json.loads(path.read_text())
    assert record["mapping"] == "martini"


def test_read_mapping_rejects_foreign_json(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "something-else"}')
    with pytest.raises(ValueError, match="not a molscope CG mapping"):
        ms.read_cg_mapping(path)


def test_apply_mapping_reproduces_the_model(tmp_path):
    mol = two_alanines()
    cg = mol.coarse_grain("martini")
    record = ms.read_cg_mapping(ms.write_cg_mapping(cg, tmp_path / "m.json"))
    rebuilt = ms.apply_cg_mapping(mol, record)
    np.testing.assert_allclose(rebuilt.coords, cg.coords)
    assert rebuilt.atom_names == cg.atom_names  # repeated BB/SC names preserved
    np.testing.assert_array_equal(rebuilt.bonds(), cg.bonds())
    assert rebuilt.coarse_grain_report.coverage() == cg.coarse_grain_report.coverage()


def test_apply_mapping_round_trips_a_real_structure(tmp_path):
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    cg = mol.coarse_grain("residue_com")
    rebuilt = ms.apply_cg_mapping(mol, ms.cg_mapping_to_dict(cg))
    np.testing.assert_allclose(rebuilt.coords, cg.coords)
    np.testing.assert_array_equal(rebuilt.bonds(), cg.bonds())


def test_apply_mapping_rejects_out_of_range_atom_index():
    mol = two_alanines()
    record = ms.cg_mapping_to_dict(mol.coarse_grain("martini"))
    with pytest.raises(ValueError, match="references atom index"):
        ms.apply_cg_mapping(mol.take([0, 1, 2]), record)  # too few atoms


def test_apply_mapping_keeps_centroid_reduction():
    mol = two_alanines()
    record = ms.cg_mapping_to_dict(mol.coarse_grain("residue_centroid"))
    rebuilt = ms.apply_cg_mapping(mol, record)
    np.testing.assert_allclose(rebuilt.coords, mol.coarse_grain("residue_centroid").coords)


# -- GROMACS-style index export --------------------------------------------


def test_write_index_groups_and_serials(tmp_path):
    cg = two_alanines().coarse_grain("martini")
    path = cg.write_index(tmp_path / "map.ndx")
    lines = [ln for ln in open(path).read().splitlines() if ln and not ln.startswith(";")]
    groups = [ln for ln in lines if ln.startswith("[")]
    # one group per bead, names disambiguated by residue, serials are 1-based
    assert groups == ["[ BB_1_ALA_A ]", "[ SC_1_ALA_A ]", "[ BB_2_ALA_A ]", "[ SC_2_ALA_A ]"]
    assert lines[1] == "1 2 3 4"   # BB of residue 1 -> atoms 0..3 -> serials 1..4
    assert lines[3] == "5"         # SC of residue 1 -> atom 4 -> serial 5


def test_write_index_disambiguates_repeated_group_names(tmp_path):
    # Two residue-less beads sharing a name must still get distinct ndx groups.
    helix = ms.read(os.path.join(DATA, "helix_201.xyz"))
    record = {
        "format": "molscope-cg-mapping",
        "mapping": "custom",
        "beads": [
            {"name": "X", "atom_indices": [0, 1], "reduction": "centroid"},
            {"name": "X", "atom_indices": [2, 3], "reduction": "centroid"},
        ],
        "bonds": [],
    }
    cg = ms.apply_cg_mapping(helix, record)
    path = cg.write_index(tmp_path / "dup.ndx")
    groups = [ln for ln in open(path).read().splitlines() if ln.startswith("[")]
    assert groups == ["[ X ]", "[ X_2 ]"]


# -- mapping visualisation --------------------------------------------------


def _agg():
    import matplotlib

    matplotlib.use("Agg")


def test_plot_mapping_returns_axes():
    _agg()
    mol = two_alanines()
    cg = mol.coarse_grain("martini")
    ax = ms.plot_mapping(mol, cg, show=False)
    assert ax is not None
    assert "mapping" in ax.get_title()


def test_plot_mapping_method_matches_function():
    _agg()
    mol = two_alanines()
    cg = mol.coarse_grain("martini")
    assert cg.plot_mapping(mol, show=False) is not None


def test_plot_mapping_handles_dropped_atoms_and_index_mappings():
    _agg()
    helix = ms.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.warns(UserWarning, match="not assigned"):
        cg = helix.coarse_grain({"head": [0, 1, 2], "tail": [3, 4, 5]})
    assert ms.plot_mapping(helix, cg, show=False) is not None


def test_plot_mapping_rejects_mismatched_structure():
    _agg()
    mol = two_alanines()
    cg = mol.coarse_grain("martini")
    with pytest.raises(ValueError, match="atoms"):
        ms.plot_mapping(mol.take([0, 1, 2]), cg, show=False)
