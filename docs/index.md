# MolScope documentation

MolScope is a lightweight Python toolkit for molecular coordinate analysis,
visualization, coarse-graining prototypes, and graph-based molecular
representations.

It is designed for teaching, exploratory research, and early-stage molecular
machine-learning workflows where users want a simple path from PDB, XYZ, CIF, or
SDF files to geometric descriptors, coarse-grained beads, and graph-ready data.

```python
import molscope as ms

mol = ms.read("1fqy.pdb")
print(mol.summary())

cg = mol.coarse_grain("residue_com")
G = cg.to_networkx()
```

## What it does

- Read `.pdb`, `.xyz`, `.cif` atom-site loops, and `.sdf` files, preserving
  explicit SDF/PDB bonds where present.
- Validate CIF/mmCIF syntax and atom-site columns with optional Gemmi support.
- Select atoms by element, chain, residue name, atom name, and residue id.
- Compute geometry, RMSD, contacts, contact maps, ensembles, and descriptors.
- Preserve SDF formal charges and expose optional RDKit-backed chemical features
  and descriptors.
- Visualize molecules with Matplotlib or py3Dmol.
- Export molecular graphs to NetworkX, PyTorch Geometric, or DGL.
- Prototype interpretable coarse-grained mappings.

## Install

```bash
pip install molscope
```

For development from the repository:

```bash
uv sync
uv run pytest
```
