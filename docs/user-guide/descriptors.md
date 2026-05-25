# Structural Descriptors

`mol.descriptors()` returns a fixed-size descriptor dictionary for quick ML
feature tables:

```python
features = mol.descriptors()
features["radius_of_gyration"]
features["principal_moments"]
features["distance_histogram"]
```

Batch featurization:

```python
X, names = m3d.featurize_many(
    ["a.pdb", "b.pdb", "c.xyz"],
    return_names=True,
)
```

Included features:

- atom and residue counts
- element counts
- molecular mass
- centroid and center of mass
- radius of gyration
- bounding-box dimensions and volume
- inertia tensor
- principal moments and axes
- shape anisotropy
- compactness
- distance histogram
- bond length summary statistics
- atom and residue contact summaries

Full contact matrices remain available through `mol.contact_map(...)`.
