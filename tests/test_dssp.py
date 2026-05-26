"""Tests for the simplified DSSP secondary-structure assignment."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import SecondaryStructure, dssp
from molscope.molecule import Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def aquaporin():
    return ms.read(os.path.join(DATA, "1fqy.pdb"))


def test_assign_returns_one_code_per_residue():
    mol = aquaporin()
    ss = mol.secondary_structure()
    assert isinstance(ss, SecondaryStructure)
    n_residues = sum(1 for _ in mol.residue_groups())
    assert len(ss) == n_residues
    assert len(ss.string) == n_residues
    assert set(ss.string) <= set("HGIEBTS-")


def test_aquaporin_is_helix_rich():
    # Aquaporin-1 is an all-alpha membrane protein: helix-dominated, no sheets.
    summary = aquaporin().secondary_structure().summary()
    assert summary["helix_fraction"] > 0.4
    assert summary["strand"] == 0
    counts = summary["helix"] + summary["strand"] + summary["coil"]
    assert counts == summary["residues"]


def test_summary_fractions_sum_to_one():
    summary = aquaporin().secondary_structure().summary()
    total = (
        summary["helix_fraction"]
        + summary["strand_fraction"]
        + summary["coil_fraction"]
    )
    assert total == pytest.approx(1.0)


def test_per_atom_ss_aligns_with_atoms():
    mol = aquaporin()
    per_atom = dssp.per_atom_ss(mol)
    assert len(per_atom) == len(mol)
    assert set(per_atom) <= set("HGIEBTS-")


def test_plot_color_by_ss(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    mol = aquaporin()
    ax = mol.plot(color_by="ss", show=False)
    assert ax is not None


def test_requires_backbone_metadata():
    # A bare coordinate molecule (no atom names / resids) cannot be assigned.
    bare = Molecule(np.zeros((3, 3)), ["C", "C", "C"])
    with pytest.raises(ValueError):
        bare.secondary_structure()
