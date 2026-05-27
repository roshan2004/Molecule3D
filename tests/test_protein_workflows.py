"""Protein-analysis workflows over bundled teaching structures."""

import os

import molscope as ms

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def test_1fqy_backbone_contacts_and_simplified_dssp():
    mol = ms.read(os.path.join(DATA, "1fqy.pdb"))

    assert mol.chain_ids() == ["A"]
    assert sum(1 for _ in mol.residue_groups()) == 226
    assert len(mol.backbone()) == 904
    assert len(mol.alpha_carbons()) == 226

    cmap = mol.contact_map(cutoff=8.0, level="residue", method="ca", min_seq_sep=4)
    assert cmap.matrix.shape == (226, 226)
    assert cmap.n_contacts == 448

    ss = mol.secondary_structure()
    summary = ss.summary()
    assert summary["residues"] == 226
    assert summary["helix_fraction"] > 0.6
    assert summary["strand"] == 0


def test_1aml_nmr_ensemble_contact_frequency():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    assert len(models) == 20
    assert all(len(model.alpha_carbons()) == 40 for model in models)

    freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
    assert freq.matrix.shape == (40, 40)
    assert freq.n_contacts == 189
    assert freq.matrix.min() >= 0.0
    assert freq.matrix.max() <= 1.0


def test_3ptb_ligands_waters_binding_site_and_secondary_structure():
    mol = ms.read(os.path.join(DATA, "3ptb.pdb"))

    assert mol.chain_ids() == ["A"]
    assert len(mol.protein()) == 1629
    assert len(mol.hetero_atoms()) == 72
    assert len(mol.select(resname="HOH")) == 62
    assert [ligand.resname for ligand in mol.ligands()] == ["BEN"]

    site = mol.binding_site(cutoff=4.5)
    assert site.ligand.resname == "BEN"
    assert len(site.residues) == 13
    assert {189, 190, 195, 219} <= {res.resid for res in site.residues}
    assert site.min_distances == sorted(site.min_distances)

    summary = mol.secondary_structure().summary()
    assert summary["residues"] == 220
    assert summary["strand"] > summary["helix"]
