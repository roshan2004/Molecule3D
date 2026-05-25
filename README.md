# Molecule3D

Read molecular structure files (`.xyz`, `.pdb`, `.cif`, `.sdf`, optionally
gzip-compressed), select and analyse atoms, and visualise them in 3D.

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
mol = m3d.fetch("1fqy")             # ...or download straight from RCSB by id
print(mol.summary())                # atoms, formula, chains, bounding box

mol = mol.centered().rotate("z", 90).translate((1, 2, -1))
mol.plot()                          # CPK colours, inferred bonds, equal aspect
```

`Molecule` is immutable: `translate`, `centered` and `rotate` each return a new
molecule, so transformations chain cleanly without aliasing. Equality is by
value (`np.array_equal` on coordinates).

### Selections

PDB/mmCIF files carry per-atom metadata (atom name, residue, chain), so you can
slice a structure:

```python
mol.select(chain="A")               # one chain
mol.select(element="C")             # all carbons
mol.select(resname="HOH")           # waters
mol.select(resid=(10, 20))          # an inclusive residue range
mol.alpha_carbons()                 # CA atoms (the usual basis for protein RMSD)
mol.backbone()                      # N, CA, C, O
mol[mask_or_indices]                # subset by numpy mask / index array
```

### Analysis and measurements

```python
mol.centroid, mol.center_of_mass    # geometric / mass-weighted centre
mol.radius_of_gyration              # compactness (angstrom)
mol.dimensions, mol.formula         # bounding box, Hill-order formula
mol.bonds()                         # inferred bond index pairs (KD-tree if scipy)
mol.contacts(cutoff=5.0)            # atom pairs within a distance

mol.distance(i, j)                  # bond length
mol.angle(i, j, k)                  # bond angle (degrees)
mol.dihedral(a, b, c, d)            # torsion angle (degrees)

a.alpha_carbons().rmsd(b.alpha_carbons(), align=True)   # CA-RMSD after Kabsch fit
```

### NMR ensembles

```python
from molecule3d import ensemble

models = m3d.read_pdb_models("1aml.pdb")     # all 20 models
ensemble.rmsd_matrix(models)                 # pairwise RMSD matrix
ensemble.rmsf(models)                        # per-atom fluctuation
ensemble.average(models)                     # mean structure
ensemble.align_all(models)                   # superpose every model onto the first
```

### Writing and viewing

```python
m3d.write_xyz(mol.centered(), "out.xyz")     # write transformed coordinates back
m3d.write_pdb(mol, "out.pdb")

mol.plot(color_by="chain")                   # colour by element / chain / residue
mol.view(style="cartoon")                    # interactive py3Dmol viewer (notebooks)
from molecule3d.plotting import spin_gif
spin_gif(mol, "spin.gif")                    # rotating animation
```

### Molecular graphs (for machine learning)

Turn 3D coordinates plus inferred bonds into a graph, then export to the common
ML frameworks. The base `to_graph()` needs no extra dependencies; each exporter
imports its backend lazily.

```python
mol = m3d.read("1fqy.pdb")

g = mol.to_graph()                  # MolecularGraph: nodes + edges, no deps
g.n_atoms, g.n_bonds                # counts
g.atomic_numbers, g.masses          # per-node arrays
g.node_features()                   # (N, 2) default features [atomic_number, mass]

G = mol.to_networkx()               # networkx.Graph with node/edge attributes
data = mol.to_pyg_data()            # torch_geometric.data.Data (x, pos, edge_index, edge_attr, z)
dglg = mol.to_dgl_graph()           # dgl.DGLGraph with ndata/edata tensors
```

Nodes carry element, atomic number, mass, coordinates and (from PDB/mmCIF) atom
name, residue and chain. Edges carry the bonded pair, interatomic distance, and
bond order (`1.0` for geometrically inferred bonds). Install backends as needed:
`pip install "molecule3d[graph]"` (networkx), `pip install torch torch_geometric`,
or `pip install dgl`.

## Command line

```bash
molecule3d helix_201.xyz --translate 1 2 -1
molecule3d 1fqy.pdb --select atom_name=CA --color-by residue --save ca.png
molecule3d --fetch 1aml --center --gif amyloid.gif
python -m molecule3d 1fqy.pdb          # equivalent if not pip-installed
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
- Optional extras: `pip install "molecule3d[fast]"` (scipy, faster bonds/contacts)
  and `"molecule3d[viz]"` (py3Dmol, for `Molecule.view`).

## Tests and linting

```bash
uv run pytest          # 46 tests
uv run ruff check .    # lint
```

CI (GitHub Actions) runs both across Python 3.9 / 3.11 / 3.13 on every push and PR.

## License

[MIT](LICENSE)
