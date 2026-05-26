# From PDB to graph and coarse-grained beads

This short workflow starts with the bundled Aquaporin-1 PDB file, builds an
ML-ready molecular graph, and then maps the same structure to coarse-grained
residue beads.

Run it from the repository root:

```python
from pathlib import Path

import molscope as ms

structure = Path("examples/data/1fqy.pdb")
mol = ms.read(structure)

print(mol.summary())
print(f"chains: {sorted(set(mol.chains))}")
print(f"alpha carbons: {len(mol.alpha_carbons())}")
```

## Build a molecular graph

```python
graph = mol.to_graph()

print(graph.n_atoms, "nodes")
print(graph.n_bonds, "edges")
print(graph.node_features("ml").shape)
print(graph.edge_features("ml").shape)
```

The dependency-free `MolecularGraph` stores atom-level coordinates, element
identity, masses, formal charges, topology metadata, and bond distances. Install
optional graph backends only when you need them:

```bash
pip install "molscope[graph]"  # NetworkX
pip install "molscope[pyg]"    # PyTorch Geometric
pip install "molscope[dgl]"    # DGL
```

Then export without changing the structure workflow:

```python
G = mol.to_networkx()
data = mol.to_pyg_data(node_preset="ml", edge_preset="ml")
```

## Coarse-grain to residue beads

```python
cg = mol.coarse_grain("residue_com")

print(len(mol), "atomistic atoms")
print(len(cg), "coarse-grained beads")
print(len(cg.bonds()), "coarse-grained bonds")
print(cg.mapping_report())
```

The coarse-grained result is still a `Molecule`, so plotting, transforms, and
graph export use the same API:

```python
cg.plot(scale=200)
cg_graph = cg.to_graph()
```

For quick prototyping you can compare the residue-level mapping with the
simplified Martini-style mapping:

```python
bb_sc = mol.coarse_grain("martini")
print(len(bb_sc), "backbone/side-chain beads")
```

These mappings are intended for teaching, inspection, and graph prototyping.
They are not production Martini parameter generation.
