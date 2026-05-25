# Build A Molecular Graph

```python
import molecule3d as m3d

mol = m3d.read("1fqy.pdb")
graph = mol.to_graph()

print(graph.n_atoms, graph.n_bonds)
print(graph.node_features().shape)
```

For NetworkX:

```python
G = mol.to_networkx()
print(G.number_of_nodes(), G.number_of_edges())
```
