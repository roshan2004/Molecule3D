# Molecular Graphs

The dependency-free graph layer:

```python
g = mol.to_graph()
g.n_atoms, g.n_bonds
g.node_features()
g.node_features("ml")
g.edge_features("ml")
g.feature_matrices(return_names=True)
g.edge_types  # bond orders, 1.0 when inferred or unknown
g.formal_charges
```

Feature presets:

- Node `default`: atomic number and mass.
- Node `basic`: atomic number, mass, formal charge.
- Node `ml`: fixed element one-hot features, atomic number, mass, formal charge, aromatic flag.
- Edge `default`: distance.
- Edge `basic`: distance and bond order.
- Edge `ml`: distance, bond order, aromatic flag.

Use optional RDKit-backed aromaticity when building the graph:

```python
g = mol.to_graph(include_chemical_features=True)
```

NetworkX export:

```python
G = mol.to_networkx()
```

PyTorch Geometric export:

```python
data = mol.to_pyg_data()
data = mol.to_pyg_data(node_preset="ml", edge_preset="ml")
```

DGL export:

```python
dglg = mol.to_dgl_graph()
dglg = mol.to_dgl_graph(node_preset="ml", edge_preset="ml")
```

Install the exporter backend you need:

```bash
pip install "molscope[graph]"  # NetworkX
pip install "molscope[pyg]"    # PyTorch Geometric
pip install "molscope[dgl]"    # DGL
pip install "molscope[gnn]"    # all graph backends
```

For custom CUDA, ROCm, Apple Silicon, or cluster builds, install the matching
PyTorch stack from the backend project's instructions first.
