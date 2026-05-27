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
- `molscope.interface_residues(mol, chain_a, chain_b, cutoff=5.0)`, `chain_contact_matrix(mol, cutoff=5.0)`: chain interfaces.
- `molscope.ligands(mol, ...)`, `binding_site(mol, ligand=None, cutoff=4.5)`: ligand detection and binding-site residues.
- `molscope.backbone_torsions(mol)`: per-residue phi/psi/omega.

## Molecule

Construction:

```python
mol = ms.Molecule(coords, elements, name="example")
```

Common methods:

- `select(...)`, `backbone()`, `alpha_carbons()`, `protein()`, `hetero_atoms()`, `chain_ids()`
- `translate(...)`, `centered(...)`, `rotate(...)`, `superpose(...)`
- `distance(...)`, `angle(...)`, `dihedral(...)`
- `centroid`, `center_of_mass`, `radius_of_gyration`, `dimensions`
- `inertia_tensor()`, `principal_moments()`, `principal_axes()`
- `distance_matrix(backend="numpy")`, `contacts(...)`, `contact_count(...)`, `contact_map(...)`
- `secondary_structure()`, `backbone_torsions()`, `interface(...)`, `chain_contacts(...)`, `ligands(...)`, `binding_site(...)`
- `bonds(...)`, `bond_order_array(...)`
- `descriptors(...)`, `rdkit_descriptors(...)`
- `chemical_features(...)`
- `coarse_grain(...)`, `mapping_report()`
- `to_graph()`, `to_networkx()`, `to_pyg_data()`, `to_dgl_graph()`
- `plot(...)`, `view(...)`, `spin_gif(...)`

## Other modules

- `molscope.ensemble`: RMSD matrices, alignment, average structures, RMSF, clustering.
- `molscope.contactmap`: contact map construction, metrics, and plotting.
- `molscope.contacts`: chain interfaces and ligand-binding-site analysis.
- `molscope.dssp`: simplified DSSP-style secondary-structure assignment, segments, and backbone torsions.
- `molscope.distance`: optional NumPy, PyTorch, and CuPy dense distance backends.
- `molscope.coarsegrain`: coarse-graining and mapping report classes.
- `molscope.descriptors`: descriptor helpers and batch featurization.
- `molscope.graph`: graph container and backend exporters.
- `molscope.chem`: optional RDKit-backed chemical perception and descriptors.
