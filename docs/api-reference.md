# API Reference

## Top-level functions

- `molscope.read(path)`: read a molecule by extension.
- `molscope.fetch(pdb_id, fmt="pdb")`: download from RCSB and read.
- `molscope.read_pdb(path)`, `read_pdb_models(path)`, `read_xyz(path)`, `read_xyz_frames(path)`, `read_cif(path)`, `read_sdf(path)`.
- `molscope.write_pdb(molecule, path)`, `write_xyz(molecule, path)`.
- `molscope.featurize_many(paths, return_names=False)`: build an ML feature matrix.

## Molecule

Construction:

```python
mol = ms.Molecule(coords, elements, name="example")
```

Common methods:

- `select(...)`, `backbone()`, `alpha_carbons()`
- `translate(...)`, `centered(...)`, `rotate(...)`, `superpose(...)`
- `distance(...)`, `angle(...)`, `dihedral(...)`
- `distance_matrix()`, `contacts(...)`, `contact_map(...)`
- `bonds(...)`
- `descriptors(...)`
- `coarse_grain(...)`, `mapping_report()`
- `to_graph()`, `to_networkx()`, `to_pyg_data()`, `to_dgl_graph()`
- `plot(...)`, `view(...)`, `spin_gif(...)`

## Other modules

- `molscope.ensemble`: RMSD matrices, alignment, average structures, RMSF, clustering.
- `molscope.contactmap`: contact map construction and plotting.
- `molscope.coarsegrain`: coarse-graining and mapping report classes.
- `molscope.descriptors`: descriptor helpers and batch featurization.
- `molscope.graph`: graph container and backend exporters.
