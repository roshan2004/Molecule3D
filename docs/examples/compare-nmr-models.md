# Compare NMR Models

```python
import molscope as ms
from molscope import ensemble

models = ms.read_pdb_models("examples/data/1aml.pdb")

matrix = ensemble.rmsd_matrix(models[:10])
clusters = ensemble.cluster(models[:10], n_clusters=3)

print(matrix)
print(clusters.labels)
print(clusters.representatives())
```
