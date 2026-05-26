# Ensemble Analysis

Read all models from an NMR PDB file:

```python
models = ms.read_pdb_models("examples/data/1aml.pdb")
```

Compute ensemble descriptors:

```python
from molscope import ensemble

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
freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
freq.plot()
```
