# PDB to Descriptors

This tutorial turns a few PDB structures into a stable, numeric feature table.
The result is a CSV you can load into scikit-learn, pandas, a notebook, or a
plain spreadsheet.

You will build:

- one row per input structure,
- stable descriptor columns from the `native-3d` preset,
- a lightweight CSV export with no optional dependencies.

## Start with bundled PDB files

MolScope includes three useful teaching structures:

```python
from pathlib import Path

paths = [
    Path("examples/data/1fqy.pdb"),  # Aquaporin-1, single model
    Path("examples/data/1aml.pdb"),  # amyloid-beta NMR ensemble, first model via read()
    Path("examples/data/3ptb.pdb"),  # trypsin with ligand/waters/calcium
]
```

`read()` returns one `Molecule`. For a multi-model PDB such as `1aml.pdb`, it
uses the first model; use `read_pdb_models()` when you want one row per model in
an ensemble.

## Inspect one structure

```python
import molscope as ms

mol = ms.read(paths[0])

print(mol.summary())
print("chains:", sorted(set(mol.chains)))
print("alpha carbons:", len(mol.alpha_carbons()))
print("radius of gyration:", round(mol.radius_of_gyration, 2), "A")
```

Descriptors are most useful when you know what biological or geometric signal
you expect. For `1fqy`, useful table-level signals include size, chain count,
shape, contact density, and residue contacts.

## Compute one descriptor dictionary

```python
features = mol.descriptors(preset="native-3d")

print(features["n_atoms"])
print(features["n_residues"])
print(features["radius_of_gyration"])
print(features["principal_moments"])
print(features["distance_histogram"])
```

The dictionary intentionally mixes scalar values and short vector values. Use
`featurize_many()` when you want a flat numeric matrix with stable columns.

## Build a feature matrix

```python
X, names = ms.featurize_many(
    paths,
    preset="native-3d",
    return_names=True,
)

print(f"{X.shape[0]} structures x {X.shape[1]} descriptor columns")
print(names[:8])
```

Expected shape for the bundled PDBs:

```text
3 structures x 71 descriptor columns
['n_atoms', 'n_residues', 'molecular_mass', 'count_H', ...]
```

The `native-3d` preset includes counts, mass, bounding-box dimensions,
compactness, bond summaries, contact summaries, centers, inertia, principal
axes/moments, shape anisotropy, and a fixed-length pairwise-distance histogram.

## Write a CSV

```python
import csv

out = Path("pdb_descriptors.csv")

with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", *names])
    for path, row in zip(paths, X):
        writer.writerow([path.name, *row])

print(f"wrote {out}")
```

You can also use the CLI for batch descriptor generation:

```bash
molscope analyze examples/data/*.pdb --out pdb_descriptors.csv --preset native-3d --jobs 4
```

## Choose the right preset

| Preset | Best for | Notes |
| --- | --- | --- |
| `native-basic` | Fast, compact tables | Counts, mass, size, compactness, bond/contact summaries. |
| `native-3d` | Shape-aware ML baselines | Adds centers, inertia, principal axes/moments, and distance histograms. |
| `rdkit-basic` | Small-molecule chemistry tables | Requires the RDKit-backed `chem` extra. |

Descriptor values are not normalized. For ML, standardize columns after export
and keep the fitted scaler with your model.

## Ensemble variant

To featurize every model in `1aml.pdb`, read all models first and flatten each
descriptor dictionary:

```python
from molscope.descriptors import flatten_descriptors

models = ms.read_pdb_models("examples/data/1aml.pdb")
names = ms.descriptor_feature_names("native-3d")

rows = []
for model_id, model in enumerate(models, start=1):
    flat = flatten_descriptors(model.descriptors(preset="native-3d"))
    rows.append([model_id, *[flat[name] for name in names]])
```

That gives one row per conformer, which is useful for ensemble clustering,
conformer classification, or toy graph-level labels.
