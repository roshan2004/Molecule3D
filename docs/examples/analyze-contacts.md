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

Dense distance and contact-map backends:

```python
ca = mol.alpha_carbons()
D = ca.distance_matrix(backend="numpy")
D_gpu = ca.distance_matrix(backend="torch", device="cuda")  # if PyTorch CUDA is installed

cmap_gpu = mol.contact_map(
    cutoff=8.0,
    level="residue",
    backend="torch",
    device="cuda",
)
```

Use `backend="auto"` to prefer an available GPU backend and otherwise fall back
to NumPy.

For ensembles:

```python
models = ms.read_pdb_models("examples/data/1aml.pdb")
freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
freq.plot()
```

See the full guide: [Contact maps and distance matrices](../user-guide/contact-maps.md).
