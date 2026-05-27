# MolScope documentation

MolScope is a lightweight Python toolkit for molecular coordinate analysis,
graph-based molecular representations, and coarse-graining prototypes.

It is designed around three core workflows for teaching, exploratory research,
and early-stage molecular machine learning:

| Workflow | Output |
| --- | --- |
| [PDB to descriptors](tutorials/pdb-to-descriptors.md) | Fixed-width structural and optional RDKit-backed feature tables. |
| [PDB to graph/GNN](tutorials/pdb-to-graph-gnn.md) | Atom/bond, residue-contact, and PyTorch Geometric-ready graph data. |
| [PDB to coarse-grained beads](tutorials/pdb-to-coarse-grained-beads.md) | Residue, simplified Martini-style, custom, and virtual-site bead models. |

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")
print(mol.summary())

cg = mol.coarse_grain("residue_com")
G = cg.to_networkx()
```

## What supports those workflows

- Read `.pdb`, `.xyz`, `.cif` atom-site loops, and `.sdf` files, preserving
  explicit SDF/PDB bonds where present.
- Validate CIF/mmCIF syntax and atom-site columns with optional Gemmi support.
- Select atoms by element, chain, residue name, atom name, and residue id.
- Compute geometry, RMSD, contacts, contact maps, ensembles, and descriptors.
- Analyze protein structures through backbone/alpha-carbon selections, ligands,
  waters, binding sites, contact maps, and simplified DSSP-style secondary
  structure.
- Preserve SDF formal charges and expose optional RDKit-backed chemical features
  and descriptors.
- Visualize molecules with Matplotlib or py3Dmol.
- Export atom/bond and residue-contact graphs to NetworkX, PyTorch Geometric,
  or DGL.
- Prototype interpretable coarse-grained mappings for teaching, inspection, and
  graph representations without claiming production simulation readiness.
- Document scientific validation against MDAnalysis, RDKit, `mkdssp`, and
  invariant checks with explicit assumptions and tolerances.

## Install

```bash
pip install molscope
```

For development from the repository:

```bash
uv sync
uv run pytest
```
