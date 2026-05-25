# Compare NMR Models

```python
import molecule3d as m3d
from molecule3d import ensemble

models = m3d.read_pdb_models("1aml.pdb")

matrix = ensemble.rmsd_matrix(models[:10])
clusters = ensemble.cluster(models[:10], n_clusters=3)

print(matrix)
print(clusters.labels)
print(clusters.representatives())
```
