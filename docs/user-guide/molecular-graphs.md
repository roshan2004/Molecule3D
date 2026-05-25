# Molecular Graphs

The dependency-free graph layer:

```python
g = mol.to_graph()
g.n_atoms, g.n_bonds
g.node_features()
```

NetworkX export:

```python
G = mol.to_networkx()
```

PyTorch Geometric export:

```python
data = mol.to_pyg_data()
```

DGL export:

```python
dglg = mol.to_dgl_graph()
```

The `[graph]` extra installs only NetworkX. Install PyTorch Geometric and DGL
manually after choosing a PyTorch build that matches your platform.
