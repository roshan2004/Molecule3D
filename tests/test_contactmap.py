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


def test_atom_contact_map_numpy_backend_matches_default():
    m = Molecule(np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]]), ["C", "C", "C"])
    np.testing.assert_array_equal(
        m.contact_map(cutoff=2.0, level="atom").matrix,
        m.contact_map(cutoff=2.0, level="atom", backend="numpy").matrix,
    )


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


def test_residue_contact_map_torch_backend_if_installed():
    pytest.importorskip("torch")
    mol = ca_chain()
    np.testing.assert_array_equal(
        mol.contact_map(cutoff=8.0, level="residue", backend="torch", device="cpu").matrix,
        mol.contact_map(cutoff=8.0, level="residue").matrix,
    )


def test_residue_map_requires_residues():
    xyz = ms.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.raises(ValueError):
        xyz.contact_map(level="residue")


def test_invalid_level_and_method():
    with pytest.raises(ValueError):
        ca_chain().contact_map(level="bogus")
    with pytest.raises(ValueError):
        ca_chain().contact_map(level="residue", method="bogus")


def two_chain():
    """Chains A (resid 1,2) and B (resid 1): A2 and B1 are 4 A apart."""
    return Molecule(
        np.array([[0.0, 0, 0], [5.0, 0, 0], [9.0, 0, 0]]), ["C", "C", "C"],
        atom_names=["CA", "CA", "CA"], resnames=["ALA", "ALA", "ALA"],
        resids=np.array([1, 2, 1]), chains=["A", "A", "B"],
    )


def test_contact_metrics():
    cm = ca_chain().contact_map(cutoff=8.0, level="residue", method="ca")
    assert cm.n_contacts == 2                       # 0-1 and 1-2
    # mean |i-j| over contacts = 1; normalised by R = 3.
    assert cm.contact_order() == pytest.approx(1.0 / 3.0)
    empty = ca_chain().contact_map(cutoff=1.0, level="residue", method="ca")
    assert empty.n_contacts == 0 and empty.contact_order() == 0.0


def test_min_seq_sep_drops_local_contacts():
    full = ca_chain().contact_map(cutoff=8.0, level="residue", method="ca")
    sep = ca_chain().contact_map(cutoff=8.0, level="residue", method="ca", min_seq_sep=2)
    assert full.n_contacts == 2
    assert sep.n_contacts == 0                       # both contacts are i,i+1


def test_chain_mode_splits_intra_and_inter():
    mol = two_chain()
    allc = mol.contact_map(cutoff=8.0, level="residue", method="ca", chain_mode="all")
    intra = mol.contact_map(cutoff=8.0, level="residue", method="ca", chain_mode="intra")
    inter = mol.contact_map(cutoff=8.0, level="residue", method="ca", chain_mode="inter")
    assert allc.n_contacts == intra.n_contacts + inter.n_contacts
    assert intra.n_contacts == 1                     # A1-A2 (5 A)
    assert inter.n_contacts == 1                     # A2-B1 (4 A)
    with pytest.raises(ValueError):
        mol.contact_map(level="residue", chain_mode="bogus")


def test_min_method_matches_brute_force():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    groups = list(mol.residue_groups())[:30]         # a contiguous slice for speed
    sub = mol.take(np.concatenate([np.array(idx) for idx, *_ in groups]))
    new = sub.contact_map(cutoff=4.5, level="residue", method="min").matrix
    blocks = [sub.coords[idx] for idx, *_ in sub.residue_groups()]
    ref = np.zeros_like(new)
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            d = np.linalg.norm(blocks[a][:, None] - blocks[b][None, :], axis=-1).min()
            ref[a, b] = ref[b, a] = 1.0 if d < 4.5 else 0.0
    np.testing.assert_array_equal(new, ref)


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


def test_plot_distance_matrix():
    import matplotlib

    matplotlib.use("Agg")
    ax = ca_chain().plot_distance_matrix(show=False)
    assert ax is not None
