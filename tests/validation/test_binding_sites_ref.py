"""Validation smoke tests for binding sites across real protein-ligand PDBs.

The remote cases are intentionally opt-in because they download structures from
RCSB. Run them with:

    MOLSCOPE_RUN_REMOTE_PDB=1 uv run pytest tests/validation/test_binding_sites_ref.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

import molscope as ms

DATA = Path(__file__).resolve().parents[2] / "examples" / "data"
PANEL = ("3ptb", "1stp", "1iep", "3ert", "1hsg", "4hvp", "2br1")

pytestmark = pytest.mark.validation


def _load_panel_structure(pdb_id: str):
    if pdb_id == "3ptb":
        return ms.read(str(DATA / "3ptb.pdb"))
    if os.environ.get("MOLSCOPE_RUN_REMOTE_PDB") != "1":
        pytest.skip("set MOLSCOPE_RUN_REMOTE_PDB=1 to fetch the remote binding-site panel")
    return ms.fetch(pdb_id)


@pytest.mark.parametrize("pdb_id", PANEL)
def test_binding_site_panel_has_records_and_pocket_descriptors(pdb_id):
    mol = _load_panel_structure(pdb_id)
    ligands = mol.ligands()
    assert ligands, f"{pdb_id} should contain at least one non-solvent ligand"

    ligand = max(ligands, key=len)
    site = mol.binding_site(ligand=ligand, cutoff=4.5)
    records = site.to_records()
    desc = site.descriptors(mol)

    assert len(records) == len(site.residues)
    assert site.n_atom_contacts > 0
    assert len(site.to_molecule(mol)) == desc["pocket_n_atoms"]
    assert desc["pocket_n_residues"] == len(site.residues)
    assert desc["ligand_n_atoms"] == len(ligand)
    assert desc["pocket_contact_residue_count"] == len(site.residues)
    assert set(desc) == set(ms.pocket_descriptor_feature_names("pocket-basic"))
    assert np.isfinite(list(desc.values())).all()
