# API Reference

## Top-level functions

- `molecule3d.read(path)`: read a molecule by extension.
- `molecule3d.fetch(pdb_id, fmt="pdb")`: download from RCSB and read.
- `molecule3d.read_pdb(path)`, `read_pdb_models(path)`, `read_xyz(path)`, `read_xyz_frames(path)`, `read_cif(path)`, `read_sdf(path)`.
- `molecule3d.write_pdb(molecule, path)`, `write_xyz(molecule, path)`.
- `molecule3d.featurize_many(paths, return_names=False)`: build an ML feature matrix.

## Molecule

Construction:

```python
mol = m3d.Molecule(coords, elements, name="example")
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

- `molecule3d.ensemble`: RMSD matrices, alignment, average structures, RMSF, clustering.
- `molecule3d.contactmap`: contact map construction and plotting.
- `molecule3d.coarsegrain`: coarse-graining and mapping report classes.
- `molecule3d.descriptors`: descriptor helpers and batch featurization.
- `molecule3d.graph`: graph container and backend exporters.
