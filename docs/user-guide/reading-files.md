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
| PDB | Fixed-column parser for `ATOM`/`HETATM`; preserves `CONECT` bonds. |
| XYZ | Single-frame and multi-frame XYZ files. |
| SDF/MOL | V2000 atom and bond block reader; preserves bond orders and formal charges. |
| CIF/mmCIF | Reader for standard `_atom_site` coordinate loops, including quoted values. |

For what each format stores, which metadata is reliable, and why PDB and mmCIF
differ, see [Coordinate formats compared](coordinate-formats.md).

PDB alternate conformations can be selected explicitly:

```python
mol = ms.read_pdb("structure.pdb", altloc="primary")
mol = ms.read_pdb("structure.pdb", altloc="highest_occupancy")
```

Supported policies are `primary`, `first`, `highest_occupancy`, and `all`.

The built-in CIF reader handles standard atom-site coordinate loops with quoted
values and semicolon-delimited text fields. Install `molscope[cif]` to use the
optional Gemmi parser and validation helpers:

```python
mol = ms.read_cif("structure.cif", parser="gemmi")
report = ms.validate_cif("structure.cif")
report.raise_for_errors()
```

Dictionary-aware validation is available when you provide local dictionary
files:

```python
report = ms.validate_cif("structure.cif", dictionaries=["mmcif_pdbx_v50.dic"])
```

Download a structure from RCSB:

```python
mol = ms.fetch("1fqy")
```

Write structures:

```python
ms.write_xyz(mol, "out.xyz")
ms.write_pdb(mol, "out.pdb")
```
