# Quickstart

```python
import molscope as ms

mol = ms.read("1fqy.pdb")
print(mol.summary())
mol.plot()
```

Transformations return new molecules:

```python
moved = mol.centered().rotate("z", 90).translate((1, 2, -1))
```

Read all models from an NMR PDB file:

```python
models = ms.read_pdb_models("1aml.pdb")
matrix = ms.ensemble.rmsd_matrix(models[:5])
```

Create ML-ready descriptors:

```python
features = mol.descriptors()
X, names = ms.featurize_many(["a.pdb", "b.pdb", "c.xyz"], return_names=True)
```

Create a graph:

```python
g = mol.to_graph()
G = mol.to_networkx()
```

Coarse-grain a structure:

```python
cg = mol.coarse_grain("residue_com")
print(cg.mapping_report())
```
