# Build A Molecular Graph

```python
import molscope as ms

mol = ms.read("1fqy.pdb")
graph = mol.to_graph()

print(graph.n_atoms, graph.n_bonds)
print(graph.node_features().shape)
```

For NetworkX:

```python
G = mol.to_networkx()
print(G.number_of_nodes(), G.number_of_edges())
```
