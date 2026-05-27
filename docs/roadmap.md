# Roadmap

This roadmap is intentionally conservative. MolScope is useful today as a
lightweight teaching, prototyping, and ML-representation toolkit; future work
should strengthen that identity rather than turn it into a full simulation
framework.

## Product focus

New work should make at least one of these workflows noticeably better:

- PDB to descriptors.
- PDB to graph/GNN.
- PDB to coarse-grained beads.

Features outside those paths should stay experimental, optional, or documented
as supporting capabilities until they earn a clearer role.

## v0.9

- Improve the documentation site structure and API reference.
- Polish the three core tutorials and keep examples aligned with them.
- Expand benchmark coverage for parsing, contact maps, graph export, and
  descriptor generation.
- Grow scientific validation from the current `1fqy`/small-molecule checks into
  a curated mini-panel for DSSP, geometry, ensembles, and bond perception.
- Make validation results easier to inspect from CI logs and docs.

## v1.0

- Declare a stable core API for `Molecule`, readers/writers, descriptors,
  contact maps, and graph export.
- Freeze descriptor and graph feature preset names where practical.
- Clarify deprecation policy for old helper APIs and compatibility shims.
- Publish a concise migration guide from pre-1.0 versions.

## Future

- Trajectory-lite support for small multi-frame XYZ/PDB workflows.
- Better CIF/mmCIF coverage while keeping Gemmi optional.
- More configurable graph edge construction for ML workflows.
- More explicit coarse-grained topology objects and export formats for
  prototyping.
- Optional generated API documentation from docstrings.

## Recently resolved

- PyPI publishing with trusted release workflow.
- `CITATION.cff` citation metadata.
- ML tutorial from PDB ensemble to PyTorch Geometric graph-level learning.
- Optional Gemmi-backed mmCIF syntax, atom-site, and dictionary validation hooks.
- Stable descriptor presets for native structural and RDKit-backed feature sets.
- Graph node/edge featurization presets for ML workflows.
- Residue-level contact graphs with NetworkX, PyG and DGL exporters.
- Convenience extras for NetworkX, PyTorch Geometric, DGL, RDKit, and Gemmi.
- Chunked distance histograms and contact-count paths for larger structures.
