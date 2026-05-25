# Molecule3D documentation

Molecule3D is a lightweight Python toolkit for molecular coordinate analysis,
visualization, coarse-graining prototypes, and graph-based molecular
representations.

It is designed for teaching, exploratory research, and early-stage molecular
machine-learning workflows where users want a simple path from PDB, XYZ, CIF, or
SDF files to geometric descriptors, coarse-grained beads, and graph-ready data.

```python
import molecule3d as m3d

mol = m3d.read("1fqy.pdb")
print(mol.summary())

cg = mol.coarse_grain("residue_com")
G = cg.to_networkx()
```

## What it does

- Read `.pdb`, `.xyz`, basic `.cif` atom-site loops, and `.sdf` files.
- Select atoms by element, chain, residue name, atom name, and residue id.
- Compute geometry, RMSD, contacts, contact maps, ensembles, and descriptors.
- Visualize molecules with Matplotlib or py3Dmol.
- Export molecular graphs to NetworkX, PyTorch Geometric, or DGL.
- Prototype interpretable coarse-grained mappings.

## Install

```bash
pip install molecule3d
```

For development from the repository:

```bash
uv sync
uv run pytest
```
