# Export To PyTorch Geometric

Install PyTorch and PyTorch Geometric manually for your platform first.

```python
import molscope as ms

mol = ms.read("1fqy.pdb")
data = mol.to_pyg_data()

print(data.x.shape)
print(data.edge_index.shape)
print(data.pos.shape)
```

See `examples/graph_to_gnn.py` in the repository for a full GNN forward pass.
