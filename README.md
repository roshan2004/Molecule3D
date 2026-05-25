# Molecule3D

Read molecular coordinate files (`.xyz`, `.pdb`) and plot the atoms in 3D.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                     # creates .venv, installs deps + dev tools from the lockfile
uv run molecule3d 1fqy.pdb  # run the CLI
uv run pytest               # run the tests
```

`uv sync` pins the interpreter from `.python-version` and resolves against
`uv.lock` for reproducible installs. Use `uv sync --no-dev` to skip the test tools.

With plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"    # or: pip install -r requirements.txt
```

## Library

```python
import molecule3d as m3d

mol = m3d.read("1fqy.pdb")          # parser chosen from the extension
print(len(mol), "atoms")

mol = mol.centered().rotate("z", 90).translate((1, 2, -1))
mol.plot()                          # CPK colours, inferred bonds, equal aspect
```

`Molecule` is immutable: `translate`, `centered` and `rotate` each return a new
molecule, so transformations chain cleanly without aliasing.

## Command line

```bash
molecule3d helix_201.xyz --translate 1 2 -1
molecule3d 1fqy.pdb --center --rotate z 90 --save aquaporin.png
python -m molecule3d 1aml.pdb          # equivalent if not pip-installed
```

## Sample structures

| File | Contents |
|------|----------|
| `helix_201.xyz` | a helix (bare coordinates) |
| `1fqy.pdb` | Aquaporin-1, single model (1661 atoms) |
| `1aml.pdb` | Alzheimer amyloid A4 peptide, 20-model NMR ensemble |

## Notes

- PDB files are parsed by **fixed columns**, not whitespace splitting, so atoms
  with touching coordinate fields (large or negative values) read correctly.
- Multi-model PDB files return a single model (`model=1` by default).
- Bond inference is `O(n^2)`; it is skipped automatically for large structures.

## Tests

```bash
uv run pytest      # or, with pip: pip install pytest && pytest
```
