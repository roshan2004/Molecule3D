# Reading Molecular Files

Use `m3d.read()` to dispatch by file extension:

```python
import molecule3d as m3d

mol = m3d.read("structure.pdb")
mol = m3d.read("trajectory.xyz")
mol = m3d.read("small_molecule.sdf")
mol = m3d.read("structure.cif")
```

Supported formats:

| Format | Notes |
| --- | --- |
| PDB | Fixed-column parser for `ATOM` and `HETATM` records. |
| XYZ | Single-frame and multi-frame XYZ files. |
| SDF/MOL | Basic V2000 atom block reader. |
| CIF/mmCIF | Basic reader for standard `_atom_site` coordinate loops. |

The CIF reader is not a full mmCIF syntax implementation. It handles standard
atom-site coordinate loops but does not aim to support all quoted, multiline, or
complex loop constructs.

Download a structure from RCSB:

```python
mol = m3d.fetch("1fqy")
```

Write structures:

```python
m3d.write_xyz(mol, "out.xyz")
m3d.write_pdb(mol, "out.pdb")
```
