# Architecture

This document maps the structure of MolScope for contributors. It describes how
the package is organised, the dependency flow between layers, and where to find
(or add) a given piece of functionality.

MolScope is a NumPy-first toolkit for turning static molecular structures into
descriptor tables, machine-learning graphs, and coarse-grained representations.
The design goal is that a bare `pip install molscope` works with NumPy alone,
while heavier scientific libraries (SciPy, RDKit, gemmi, PyTorch, ...) stay
behind optional extras and are imported lazily only when a feature needs them.

## Layered design

The package is organised as a one-directional stack: a small reference layer at
the bottom, a central data model, a ring of single-responsibility analysis
modules, and three interchangeable front-ends on top.

```
                 +-------------------------------------------+
   interfaces    |  cli.py / __main__.py   mcp_server.py     |
                 |  plotting.py            (Python API)       |
                 +-------------------------------------------+
                                    ^
                 +-------------------------------------------+
   analysis      |  distance  contacts  contactmap  dssp     |
                 |  descriptors  ensemble  coarsegrain  graph |
                 |  chem        library     prepare           |
                 +-------------------------------------------+
                                    ^
                 +-------------------------------------------+
   I/O           |  io.py        cif.py                        |
                 +-------------------------------------------+
                                    ^
                 +-------------------------------------------+
   core model    |  molecule.py  (Molecule, ResidueId, ...)   |
                 +-------------------------------------------+
                                    ^
                 +-------------------------------------------+
   reference     |  elements.py  (periodic-table data)        |
                 +-------------------------------------------+
```

The flow is strictly upward: the core model never imports the interface layer,
and the reference layer depends on nothing. This is what keeps the bare install
light and lets every external library be quarantined behind an optional extra.

## The package: `molscope/`

### Reference layer

- **`elements.py`** — periodic-table reference data (covalent radii, symbols).
  Pure lookups, no dependencies. The bottom of the stack.

### Core data model

- **`molecule.py`** — the heart of the package. Defines the frozen `Molecule`
  dataclass plus `ResidueId`, `ResidueGroup` and `UnitCell`. Coordinates,
  elements, chains and charges are held as NumPy arrays. Provides the selection
  algebra (`__getitem__`, `&`, `|`, `-`, `~`), geometric bond/contact perception
  (`bonds()`, `contacts()` with a SciPy KD-tree fast path and a pure-NumPy
  cell-list fallback), `superpose()`/`rmsd()`, and the human-readable
  `summary()`/`__repr__`. Almost everything else depends on this module.

### I/O layer

- **`io.py`** — readers and writers. `read()` dispatches on file extension and
  handles PDB, builtin mmCIF, SDF/MOL (V2000), and xyz (optionally gzipped),
  plus RCSB `fetch()` and SMILES. `write_pdb` and `write_xyz` live here.
- **`cif.py`** — mmCIF *validation* (`validate_cif`, `CifValidationReport`) via
  the optional gemmi backend.
- **`chem.py`** — the RDKit bridge: `to_rdkit`, chemical-feature perception,
  template-based bond perception, and the idealised pH-7 protonation table for
  standard residues. (Documented as a static textbook model, not a pKa
  predictor.)
- **`library.py`** — tabular molecule libraries. The one place MolScope works on
  a *table of molecules* (CSV/XLSX rows with an id, a SMILES string and numeric
  properties) rather than a single 3D structure. Computes descriptors and picks
  a diverse subset via a pure-NumPy MaxMin farthest-first traversal.
- **`prepare.py`** — dataset preparation on top of `library`. Turns a table or
  multi-record SDF into ML-ready `train`/`validation`/`test` CSVs plus an
  optional `descriptors.csv`, a markdown report and a summary figure. Random and
  diversity splits and exact dedup are pure-NumPy; scaffold splits, canonical
  dedup, RDKit descriptors, Morgan fingerprints and SDF input are gated behind
  the `chem` extra. The plumbing for the `molscope prepare` command.

### Analysis layer

- **`distance.py`** — dense pairwise distance and contact matrices plus the
  pure-NumPy O(N) cell-list neighbour search, with numpy/torch/cupy/scipy
  backends and optional minimum-image (PBC) support. The performance core.
- **`contacts.py`** — binding sites, atom- and residue-level contacts, chain
  interfaces, and pocket descriptors.
- **`contactmap.py`** — the `ContactMap` result object, with filtering by
  sequence separation and intra/inter-chain mode.
- **`dssp.py`** — simplified DSSP secondary-structure assignment and backbone
  torsions (phi/psi).
- **`descriptors.py`** — structural descriptors and featurisation, the inertia
  tensor, and batch `featurize_many`.
- **`ensemble.py`** — multi-model workflows: `align_all`, `rmsd_matrix`,
  clustering, and contact frequency across an ensemble.
- **`coarsegrain.py`** — the coarse-graining engine: residue-COM, simplified
  Martini-style, custom and virtual-site bead mappings, plus index/mapping I/O.
- **`graph.py`** — graph export to NetworkX, PyTorch Geometric and DGL, with
  named node/edge feature presets. The GNN-preprocessing bridge.

### Interface layer

- **`plotting.py`** — Matplotlib renderers (distance matrix, RMSD heatmap, CG
  mapping).
- **`cli.py`** / **`__main__.py`** — the `molscope` command-line interface.
- **`mcp_server.py`** — a Model Context Protocol server exposing the analysis
  tools (render structure, contact map, RMSD, descriptors, ...) to AI agents.
- **`__init__.py`** — the public Python API; re-exports the names users import
  as `molscope.<name>` and defines `__version__`.

## Optional dependencies

Every heavy backend is an extra declared in `pyproject.toml` and imported lazily
at the point of use, so importing `molscope` never pulls them in:

| Extra | Brings in | Enables |
|-------|-----------|---------|
| `fast` | SciPy | KD-tree bond/contact search for large structures |
| `chem` | RDKit | chemical perception, SMILES descriptors |
| `cif` | gemmi | robust mmCIF parsing and validation |
| `graph` | NetworkX | NetworkX graph export |
| `gpu` | PyTorch | dense distance/contact-map backend |
| `pyg` | PyTorch + PyG | PyTorch Geometric graph export |
| `dgl` | PyTorch + DGL | DGL graph export |
| `viz` | py3Dmol | interactive notebook viewer |
| `mcp` | mcp | the MCP server |
| `xlsx` | openpyxl | reading XLSX molecule libraries |

When a backend is absent, the relevant code path either falls back to a
pure-NumPy implementation (e.g. the cell-list neighbour search) or raises a
clear error pointing at the extra to install.

## Supporting directories

- **`tests/`** — one test module per package module, plus `tests/validation/`
  for tier-2 cross-checks against reference tools (DSSP, MDAnalysis, RDKit).
  Tests use `pytest.importorskip` so the suite passes on a bare install and
  exercises the backends when the extras are present.
- **`docs/`** — MkDocs site source: quickstart, user guide, tutorials, API
  reference, benchmarks, limitations and roadmap. Built output lives in
  `docs/_build/` and `site/`.
- **`examples/`** — runnable example scripts (`protein_analysis.py`,
  `graph_to_gnn.py`, `binding_site.py`, `tour.py`, ...) with `data/` fixtures.
- **`notebooks/`** — tutorial notebooks, generated from `scripts/build_*.py` so
  they stay in sync with the example scripts.
- **`scripts/`** — developer tooling: benchmarks and the image/notebook/PDF
  builders that produce documentation assets.
- **`paper/`** — the JOSS submission (`paper.md` + `paper.bib`), built by the
  `draft-paper.yml` workflow.

## Continuous integration

`.github/workflows/ci.yml` defines the pipeline:

- **`test`** — runs the suite on Python 3.9 / 3.11 / 3.13 with the bare install
  (NumPy only), proving the package works without any extra. Also runs `ruff`.
- **`extras-smoke`** — installs each extra in isolation and checks the package
  still imports, catching extra-specific packaging breakage.
- **`coverage`** — the authoritative run: installs the optional backends
  (SciPy, RDKit, gemmi, NetworkX, PyTorch, PyG, ...) so the backend code paths
  actually execute, then uploads coverage to Codecov. `dgl` and `cupy` are
  omitted (no Linux wheel / no GPU runner).
- **`validation`** — tier-2 cross-checks against reference scientific tools,
  Linux-only because the reference binaries install cleanly via apt there.

Two further workflows handle releases: `draft-paper.yml` builds the JOSS PDF on
changes under `paper/`, and `publish.yml` publishes to PyPI via OIDC trusted
publishing when a GitHub Release is created.
