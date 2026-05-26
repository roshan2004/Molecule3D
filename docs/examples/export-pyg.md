# Export To PyTorch Geometric

Install PyTorch and PyTorch Geometric manually for your platform first.

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")
data = mol.to_pyg_data()

print(data.x.shape)
print(data.edge_index.shape)
print(data.pos.shape)
```

See `examples/graph_to_gnn.py` in the repository for a full GNN forward pass.
For a graph-level classifier/regressor example, see
`docs/examples/pdb-to-pyg-ml.md` and `examples/pdb_to_pyg_ml.py`.
