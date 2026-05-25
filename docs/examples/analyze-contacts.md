# Analyze Contacts

```python
import molecule3d as m3d

mol = m3d.read("1fqy.pdb")

pairs = mol.contacts(cutoff=5.0)
print(len(pairs))

cmap = mol.contact_map(cutoff=8.0, level="residue")
print(cmap.matrix.shape)
cmap.plot()
```

For ensembles:

```python
models = m3d.read_pdb_models("1aml.pdb")
freq = m3d.ensemble_contact_frequency(models, cutoff=8.0)
freq.plot()
```
