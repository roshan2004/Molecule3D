# Reading Molecular Files

Use `ms.read()` to dispatch by file extension:

```python
import molscope as ms

mol = ms.read("structure.pdb")
mol = ms.read("trajectory.xyz")
mol = ms.read("small_molecule.sdf")
mol = ms.read("structure.cif")
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
mol = ms.fetch("1fqy")
```

Write structures:

```python
ms.write_xyz(mol, "out.xyz")
ms.write_pdb(mol, "out.pdb")
```
