# Changelog

All notable changes to MolScope are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the package is pre-1.0, minor versions may include backwards-incompatible
API changes; these are called out under **Changed** where they occur.

## [Unreleased]

### Added

- Molecule-table workflow: `molscope select` and the `molscope.library` module
  read a CSV/XLSX of molecules and pick a diverse subset by MaxMin
  (farthest-first) selection over descriptors. Select on existing numeric columns
  (e.g. `MW`, `ALogP`) or compute RDKit descriptors from a SMILES column with
  `--compute-descriptors --smiles-col`. Adds an `xlsx` extra (`openpyxl`) for
  spreadsheet I/O; CSV input and selection need no optional backend.
- Optional MCP (Model Context Protocol) server, `molscope.mcp_server`, exposing
  MolScope's analyses as tools for AI assistants such as Claude Code and Claude
  Desktop. Adds a `molscope-mcp` console script, an `mcp` extra
  (`pip install "molscope[mcp]"`, Python >= 3.10), and nine read-only tools that
  wrap the existing API: summarise, descriptors, secondary structure, contact
  map, binding site, molecular graph, coarse-grain, and two PNG render tools.
- Broadened the DSSP reference cross-check to three fold classes instead of one:
  helix-dominated Aquaporin-1 (`1fqy`), mixed alpha/beta ubiquitin (`1ubq`), and
  the all-beta SH3 domain (`1shg`). The test is parametrised and prints
  per-fold 3-state agreement so results read as a range. Measured against
  `mkdssp` 4.5.8: 99.1% (`1fqy`), 100% (`1ubq`), 98.2% (`1shg`).
- Bundled `1ubq.pdb` and `1shg.pdb` in `examples/data` as the new validation
  structures.

### Changed

- Documentation and the JOSS paper now report DSSP agreement as a measured
  range across fold classes rather than a single helical figure, and no longer
  imply that strand-rich folds agree markedly less well.

## [0.8.3] - 2026-05-28

### Added

- JOSS paper draft under `paper/` and Zenodo deposition metadata
  (`.zenodo.json`) for an archival DOI.

### Changed

- Bumped GitHub Actions to Node 24-ready versions.

## [0.8.2] - 2026-05-28

### Added

- Read the Docs hosting for the documentation site.
- Coverage reporting via `pytest-cov` and Codecov, measured with the optional
  backends installed so the RDKit, gemmi, NetworkX, SciPy and Torch paths are
  exercised rather than skipped.
- Coarse-grained virtual-site support, preserved as derived coordinate metadata.
- PDB workflow tutorials and a protein-analysis-from-scratch walkthrough.
- CLI batch analysis and graph-export subcommands, with improved selection
  handling.
- `O(n)` cell-list neighbour search and opt-in periodic-boundary support for the
  distance and contact methods.
- Residue contact graphs for ML and an educational coarse-graining workflow.
- Scientific validation tables, references, and a reproducible benchmarks page.

### Changed

- Reorganised the limitations page by workflow and added a Graph ML section.
- Refocused the documentation around the three core workflows.
- Completed the geometry API and added a visual geometry guide validated against
  MDAnalysis.
- Polished the coarse-graining workflow, contact maps, and dense distance
  backends; improved coordinate-format parser errors and added edge-case
  fixtures.
- Stopped tracking generated graph exports and added `graphs/` to `.gitignore`.

### Fixed

- Batch CLI crash with `--jobs > 1` on spawn-based platforms.
- gemmi-backed mmCIF unit-cell read.
- Lint errors, and made periodic boundaries opt-in for distance and contact
  methods.

## [0.8.1] - 2026-05-26

### Added

- Tier-2 validation suite cross-checking against reference scientific tools,
  covering DSSP (`mkdssp`), geometry and RMSD (MDAnalysis), bond perception and
  chemical features (RDKit), and contact maps.
- PyTorch Geometric ML tutorial and `CITATION.cff` citation metadata.

### Changed

- DSSP validation now invokes `mkdssp` directly instead of going through
  Biopython, and the 3-state agreement floor was tightened to 0.95 after
  observing 99.1% on CI.
- CI runs the validation job with the required extras and fails loudly if the
  reference tools cannot be imported.
- Polished repository layout and documentation.

### Fixed

- README Mermaid diagram syntax.

## [0.8.0] - 2026-05-26

### Added

- Expanded molecular parsing and machine-learning feature support.

## [0.7.0] - 2026-05-25

### Added

- Simplified, dependency-free DSSP-style secondary-structure assignment based on
  the Kabsch-Sander hydrogen-bond model.

### Changed

- README: added a "Why MolScope" section, a tool comparison, and a CLI output
  example; the secondary-structure render replaced the earlier hero animation.

## [0.6.2] - 2026-05-25

### Fixed

- README images now render on PyPI, and the publish workflow was hardened.

## [0.6.1] - 2026-05-25

### Added

- PyPI trusted-publishing workflow.

### Changed

- Expanded the package docstring to cover the full feature scope.

## [0.6.0] - 2026-05-25

Initial public release under the **MolScope** name, renamed from the earlier
`molecule3d` prototype. This release consolidated the core toolkit:

### Added

- `Molecule` object on a NumPy core, with fixed-column PDB parsing and readers
  and writers for XYZ, PDB, mmCIF and SDF.
- Per-atom metadata, metadata-based selections, geometry measurements, and RMSD.
- Molecular graph construction with NetworkX, PyTorch Geometric and DGL
  exporters.
- Coarse-graining tools: residue and custom mappings, explicit-bond support,
  index-based mappings, and a dropped-atom warning.
- Contact maps and ensemble contact-frequency analysis, plus ensemble RMSD
  clustering and an RMSD heatmap.
- Native structural descriptors and a MkDocs documentation site, including a
  user-guide PDF builder.
- `uv` support (lockfile, dev dependency group, `.python-version`), continuous
  integration, and README visual examples.

[Unreleased]: https://github.com/roshan2004/molscope/compare/v0.8.3...HEAD
[0.8.3]: https://github.com/roshan2004/molscope/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/roshan2004/molscope/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/roshan2004/molscope/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/roshan2004/molscope/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/roshan2004/molscope/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/roshan2004/molscope/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/roshan2004/molscope/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/roshan2004/molscope/releases/tag/v0.6.0
