# Ensemble Analysis

Read all models from an NMR PDB file:

```python
models = m3d.read_pdb_models("1aml.pdb")
```

Compute ensemble descriptors:

```python
from molecule3d import ensemble

aligned = ensemble.align_all(models)
avg = ensemble.average(models)
rmsf = ensemble.rmsf(models)
matrix = ensemble.rmsd_matrix(models)
```

Cluster structures by RMSD:

```python
result = ensemble.cluster(models, n_clusters=3)
result.labels
result.representatives()
```

Contact frequency across models:

```python
freq = m3d.ensemble_contact_frequency(models, cutoff=8.0)
freq.plot()
```
