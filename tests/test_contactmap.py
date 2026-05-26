"""Tests for contact maps and ensemble contact frequency."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import ContactMap, Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def ca_chain():
    """Three residues, one CA each, at x = 0, 5, 12."""
    return Molecule(
        np.array([[0.0, 0, 0], [5.0, 0, 0], [12.0, 0, 0]]), ["C", "C", "C"],
        atom_names=["CA", "CA", "CA"], resnames=["ALA", "ALA", "ALA"],
        resids=np.array([1, 2, 3]), chains=["A", "A", "A"],
    )


def test_atom_contact_map():
    m = Molecule(np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]]), ["C", "C", "C"])
    cm = m.contact_map(cutoff=2.0, level="atom")
    assert isinstance(cm, ContactMap)
    assert cm.matrix.shape == (3, 3)
    assert cm.matrix[0, 1] == 1 and cm.matrix[0, 2] == 0
    assert np.diag(cm.matrix).sum() == 0          # no self-contacts


def test_residue_ca_contact_map():
    cm = ca_chain().contact_map(cutoff=8.0, level="residue", method="ca")
    assert cm.matrix.shape == (3, 3)
    # 0-1 (5 A) and 1-2 (7 A) in contact; 0-2 (12 A) not
    assert cm.matrix[0, 1] == 1 and cm.matrix[1, 2] == 1 and cm.matrix[0, 2] == 0
    assert np.array_equal(cm.matrix, cm.matrix.T)
    assert cm.labels == ["A:ALA1", "A:ALA2", "A:ALA3"]
    assert not cm.is_frequency


def test_residue_methods_all_run():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    for method in ("ca", "com", "min"):
        cutoff = 4.5 if method == "min" else 8.0
        cm = mol.contact_map(cutoff=cutoff, level="residue", method=method)
        assert cm.matrix.shape == (226, 226)
        assert np.array_equal(cm.matrix, cm.matrix.T)
        assert np.diag(cm.matrix).sum() == 0


def test_residue_map_requires_residues():
    xyz = ms.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.raises(ValueError):
        xyz.contact_map(level="residue")


def test_invalid_level_and_method():
    with pytest.raises(ValueError):
        ca_chain().contact_map(level="bogus")
    with pytest.raises(ValueError):
        ca_chain().contact_map(level="residue", method="bogus")


def test_ensemble_contact_frequency():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
    n = len(list(models[0].residue_groups()))
    assert freq.matrix.shape == (n, n)
    assert freq.is_frequency
    assert (freq.matrix >= 0).all() and (freq.matrix <= 1).all()
    assert np.array_equal(freq.matrix, freq.matrix.T)
    assert np.diag(freq.matrix).sum() == 0
    # the whole point: some pairs are in contact in only a fraction of models
    assert ((freq.matrix > 0) & (freq.matrix < 1)).any()


def test_plot_contact_map(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    ax = ca_chain().contact_map(level="residue").plot(show=False)
    assert ax is not None
    freq = ms.ensemble_contact_frequency(
        ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))[:5]
    )
    assert freq.plot(show=False) is not None
