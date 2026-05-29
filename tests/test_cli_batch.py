"""Tests for the batch CLI subcommands (`analyze`, `export`).

The parallel (`--jobs 2`) cases are regression tests: the worker functions must
be importable at module level so they pickle under the ``spawn`` multiprocessing
start method used on macOS and Windows. A nested-closure worker passes on Linux
(``fork``) but raises ``AttributeError: Can't get local object`` elsewhere.
"""

import csv
import os

import pytest

from molscope.cli import main

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
PDBS = [os.path.join(DATA, "1fqy.pdb"), os.path.join(DATA, "3ptb.pdb")]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_analyze_serial(tmp_path):
    out = tmp_path / "serial.csv"
    rc = main(["analyze", *PDBS, "--out", str(out), "--jobs", "1"])
    assert rc == 0
    rows = _read_csv(out)
    assert len(rows) == len(PDBS)
    assert {os.path.basename(r["file"]) for r in rows} == {"1fqy.pdb", "3ptb.pdb"}
    assert all(float(r["n_atoms"]) > 0 for r in rows)


def test_analyze_parallel_matches_serial(tmp_path):
    """--jobs 2 must not crash on spawn platforms and must match serial output."""
    serial = tmp_path / "serial.csv"
    parallel = tmp_path / "parallel.csv"
    assert main(["analyze", *PDBS, "--out", str(serial), "--jobs", "1"]) == 0
    assert main(["analyze", *PDBS, "--out", str(parallel), "--jobs", "2"]) == 0

    def by_file(rows):
        return {os.path.basename(r["file"]): r for r in rows}

    assert by_file(_read_csv(serial)) == by_file(_read_csv(parallel))


def test_export_nx_parallel(tmp_path):
    """Export worker must also pickle under spawn (--jobs 2)."""
    pytest.importorskip("networkx")
    out_dir = tmp_path / "graphs"
    rc = main(["export", *PDBS, "--to", "nx", "--out-dir", str(out_dir), "--jobs", "2"])
    assert rc == 0
    assert sorted(p.name for p in out_dir.glob("*.json")) == ["1fqy.json", "3ptb.json"]


def test_binding_site_writes_residue_csv_and_descriptors(tmp_path):
    out = tmp_path / "site.csv"
    descriptors_out = tmp_path / "pocket.csv"
    rc = main([
        "binding-site",
        os.path.join(DATA, "3ptb.pdb"),
        "--out",
        str(out),
        "--cutoff",
        "4.5",
        "--descriptors-out",
        str(descriptors_out),
    ])
    assert rc == 0

    rows = _read_csv(out)
    assert len(rows) == 13
    assert rows[0]["ligand_resname"] == "BEN"
    assert {"ASP", "SER", "TRP"} <= {row["resname"] for row in rows}
    assert any(row["resid"] == "189" and row["resname"] == "ASP" for row in rows)
    assert all(float(row["min_distance"]) <= 4.5 for row in rows)

    desc_rows = _read_csv(descriptors_out)
    assert len(desc_rows) == 1
    assert float(desc_rows[0]["pocket_n_residues"]) == 13.0
    assert float(desc_rows[0]["pocket_atom_contact_count"]) == 107.0


def test_binding_site_unknown_ligand_reports_error(tmp_path, capsys):
    out = tmp_path / "site.csv"
    rc = main([
        "binding-site",
        os.path.join(DATA, "3ptb.pdb"),
        "--out",
        str(out),
        "--ligand",
        "ZZZ",
    ])
    assert rc == 2
    assert not out.exists()
    assert "Binding-site analysis failed" in capsys.readouterr().err
