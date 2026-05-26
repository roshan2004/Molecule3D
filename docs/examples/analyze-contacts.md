# Analyze Contacts

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")

pairs = mol.contacts(cutoff=5.0)
print(len(pairs))

cmap = mol.contact_map(cutoff=8.0, level="residue")
print(cmap.matrix.shape)
cmap.plot()
```

For ensembles:

```python
models = ms.read_pdb_models("examples/data/1aml.pdb")
freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
freq.plot()
```
