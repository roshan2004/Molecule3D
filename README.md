# Molecule3D

Read molecular coordinate files (`.xyz`, `.pdb`), analyse them, and plot the
atoms in 3D.

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

## Quickstart

A runnable end-to-end tour over the bundled sample structures lives in
[`example.py`](example.py):

```bash
uv run python example.py                  # opens 3D plot windows
MPLBACKEND=Agg uv run python example.py   # headless: saves PNGs instead
```

It reads an `.xyz` and a `.pdb`, prints derived properties, compares the NMR
models of `1aml`, writes a transformed structure back out, and renders a plot.

## Library

```python
import molecule3d as m3d

mol = m3d.read("1fqy.pdb")          # parser chosen from the extension
print(len(mol), "atoms")

mol = mol.centered().rotate("z", 90).translate((1, 2, -1))
mol.plot()                          # CPK colours, inferred bonds, equal aspect
```

`Molecule` is immutable: `translate`, `centered` and `rotate` each return a new
molecule, so transformations chain cleanly without aliasing. Equality is by
value (`np.array_equal` on coordinates).

### Analysis

```python
mol.centroid             # geometric centre
mol.center_of_mass       # mass-weighted centre
mol.radius_of_gyration   # compactness (angstrom)
mol.bonds()              # inferred bond index pairs (KD-tree if scipy installed)

# Compare structures (matched by atom index)
a.rmsd(b)                # root-mean-square deviation
a.rmsd(b, align=True)    # minimum RMSD after Kabsch superposition
a.superpose(b)           # a rigidly fitted onto b
```

### NMR ensembles and writing

```python
models = m3d.read_pdb_models("1aml.pdb")     # all 20 models as a list
models[0].rmsd(models[1], align=True)

m3d.write_xyz(mol.centered(), "out.xyz")      # write transformed coordinates back
m3d.write_pdb(mol, "out.pdb")
```

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
- Alternate conformations (altLoc) other than the primary one are skipped.
- `read_pdb` returns a single model (`model=1` by default); use `read_pdb_models`
  for the whole ensemble.
- Bond inference uses a `scipy.spatial.cKDTree` when available; without scipy it
  falls back to a dense `O(n^2)` search that is refused above ~8000 atoms.
  Install the accelerator with `pip install "molecule3d[fast]"`.

## Tests and linting

```bash
uv run pytest          # 28 tests
uv run ruff check .    # lint
```

CI (GitHub Actions) runs both across Python 3.9 / 3.11 / 3.13 on every push and PR.

## License

[MIT](LICENSE)
