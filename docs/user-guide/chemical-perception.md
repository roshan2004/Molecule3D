# Chemical Perception

MolScope keeps cheminformatics out of the core install. For aromaticity,
valence and sanitized charge features, install the optional RDKit backend:

```bash
pip install "molscope[chem]"
```

Then call:

```python
import molscope as ms

mol = ms.read("ligand.sdf")
chem = mol.chemical_features()

chem.formal_charges
chem.total_valences
chem.aromatic_atoms
chem.bond_orders
chem.aromatic_bonds
```

RDKit descriptors are also available through the same optional backend:

```python
rdkit_features = mol.rdkit_descriptors(names=["MolWt", "TPSA"])
features = mol.descriptors(include_rdkit=True, rdkit_descriptor_names=["MolWt"])
```

SDF/MOL V2000 formal charges and bond orders are preserved by the built-in
reader before RDKit is involved. If a structure only has coordinates, MolScope
can pass geometrically inferred single bonds to RDKit, but it does not infer
general bond orders from raw coordinates.
