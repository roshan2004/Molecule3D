# Build A Molecular Graph

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")
graph = mol.to_graph()

print(graph.n_atoms, graph.n_bonds)
print(graph.node_features().shape)
```

For NetworkX:

```python
G = mol.to_networkx()
print(G.number_of_nodes(), G.number_of_edges())
```

For a residue-level spatial graph:

```python
rg = mol.to_residue_contact_graph(cutoff=8.0, method="ca", min_seq_sep=4)
print(rg.n_residues, rg.n_contacts)
```
