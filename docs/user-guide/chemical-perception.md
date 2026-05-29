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

## Proteins: residue-template bonds

A bare protein PDB has no bond records, so geometric inference gives single bonds
only: aromaticity, backbone carbonyls, and side-chain double bonds are all lost,
and `chemical_features` comes back essentially empty (zero aromatic atoms).

For **standard residues** the chemistry is known, so you don't need to guess it.
Read with `bond_perception="template"` to have RDKit's residue-aware PDB reader
assign the bonds, Kekule bond orders, and formal charges from its built-in
residue templates (plus peptide bonds and disulfides):

```python
mol = ms.read("protein.pdb", bond_perception="template")   # needs the chem extra
chem = mol.chemical_features()
chem.aromatic_atoms.sum()    # now counts the Phe/Tyr/His/Trp rings
```

`read`, `read_pdb`, and `fetch` all accept the option (PDB input only). It needs
RDKit, and it only helps for standard residues: modified residues, non-standard
ligands, and exotic chemistry still fall back to best-effort perception. This is
**not** a force field, just template-based connectivity and bond orders; for
energies and parameters use a dedicated force-field tool. The default stays
`"geometric"`, so nothing changes unless you opt in.
