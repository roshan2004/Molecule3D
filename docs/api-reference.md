# API Reference

## Top-level functions

- `molscope.read(path)`: read a molecule by extension.
- `molscope.fetch(pdb_id, fmt="pdb")`: download from RCSB and read.
- `molscope.read_pdb(path)`, `read_pdb_models(path)`, `read_xyz(path)`, `read_xyz_frames(path)`, `read_cif(path)`, `read_sdf(path)`.
- `molscope.validate_cif(path)`: optional Gemmi-backed CIF/mmCIF validation.
- `molscope.write_pdb(molecule, path)`, `write_xyz(molecule, path)`.
- `molscope.featurize_many(paths, return_names=False)`: build an ML feature matrix.
- `molscope.descriptor_feature_names(preset)`: stable flattened descriptor columns.
- `molscope.node_feature_names(preset)`, `edge_feature_names(preset)`: graph preset columns.

## Molecule

Construction:

```python
mol = ms.Molecule(coords, elements, name="example")
```

Common methods:

- `select(...)`, `backbone()`, `alpha_carbons()`
- `translate(...)`, `centered(...)`, `rotate(...)`, `superpose(...)`
- `distance(...)`, `angle(...)`, `dihedral(...)`
- `distance_matrix(backend="numpy")`, `contacts(...)`, `contact_count(...)`, `contact_map(...)`
- `bonds(...)`, `bond_order_array(...)`
- `descriptors(...)`, `rdkit_descriptors(...)`
- `chemical_features(...)`
- `coarse_grain(...)`, `mapping_report()`
- `to_graph()`, `to_networkx()`, `to_pyg_data()`, `to_dgl_graph()`
- `plot(...)`, `view(...)`, `spin_gif(...)`

## Other modules

- `molscope.ensemble`: RMSD matrices, alignment, average structures, RMSF, clustering.
- `molscope.contactmap`: contact map construction and plotting.
- `molscope.distance`: optional NumPy, PyTorch, and CuPy dense distance backends.
- `molscope.coarsegrain`: coarse-graining and mapping report classes.
- `molscope.descriptors`: descriptor helpers and batch featurization.
- `molscope.graph`: graph container and backend exporters.
- `molscope.chem`: optional RDKit-backed chemical perception and descriptors.
