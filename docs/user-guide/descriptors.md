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
X, names = ms.featurize_many(
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
Distance histograms and atom contact counts are computed in coordinate blocks
instead of a full pairwise distance array:

```python
features = mol.descriptors(distance_chunk_size=2048)
```

Stable presets are available when you need reproducible feature columns:

```python
features = mol.descriptors(preset="native-basic")
X, names = ms.featurize_many(paths, preset="native-3d", return_names=True)
names = ms.descriptor_feature_names("native-3d")
```

Preset options:

- `native-basic`: counts, mass, size, compactness, bond summaries, and contact summaries.
- `native-3d`: `native-basic` plus centres, inertia, principal axes/moments, and distance histograms.
- `rdkit-basic`: `native-basic` plus a stable subset of RDKit scalar descriptors.

## RDKit descriptors

Install the optional chemical backend to access RDKit's scalar descriptor set:

```bash
pip install "molscope[chem]"
```

Use RDKit descriptors directly:

```python
rdkit_features = mol.rdkit_descriptors(names=["MolWt", "TPSA", "NumHDonors"])
```

Or merge selected RDKit descriptors into the standard MolScope descriptor
dictionary:

```python
features = mol.descriptors(
    include_rdkit=True,
    rdkit_descriptor_names=["MolWt", "TPSA", "NumHDonors"],
)
```

When `rdkit_descriptor_names` is omitted, all scalar RDKit descriptors available
in the installed RDKit version are included with an `rdkit_` prefix.
