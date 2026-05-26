"""Tier 2 validation: distance-based bond perception vs RDKit topology.

For a molecule with a clean 3D geometry, RDKit's bond graph is the ground truth.
We build small molecules from SMILES, embed and minimise a 3D conformer with
RDKit, then check that molscope's purely geometric ``bonds()`` recovers exactly
that connectivity. Scored as per-molecule recall and precision over the bond
set. Skips when RDKit is not installed.
"""

import numpy as np
import pytest

import molscope as ms

pytestmark = pytest.mark.validation

# name -> SMILES; a small spread of hybridisations, rings and heteroatoms.
PANEL = {
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "glycine": "NCC(=O)O",
    "toluene": "Cc1ccccc1",
    "acetic_acid": "CC(=O)O",
}


def _embed(smiles: str):
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=7) != 0:
        pytest.skip(f"RDKit could not embed {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    coords = np.asarray(mol.GetConformer().GetPositions())
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    truth = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    return coords, elements, truth


@pytest.mark.parametrize("name", list(PANEL))
def test_distance_bonds_recover_rdkit_topology(name):
    coords, elements, truth = _embed(PANEL[name])
    perceived = {frozenset(map(int, p)) for p in ms.Molecule(coords, elements).bonds(tolerance=1.2)}

    shared = len(truth & perceived)
    recall = shared / len(truth)
    precision = shared / len(perceived)
    print(f"\n{name}: recall={recall:.3f} precision={precision:.3f} "
          f"(rdkit={len(truth)}, perceived={len(perceived)})")

    # Measured at 1.000/1.000 across the panel; keep a small margin for future
    # molecules without letting a real perception regression slip through.
    assert recall >= 0.98
    assert precision >= 0.98
