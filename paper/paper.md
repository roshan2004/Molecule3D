---
title: 'MolScope: lightweight molecular structure analysis, graph export, and coarse-graining in Python'
tags:
  - Python
  - molecular structure
  - protein structure
  - graph neural networks
  - coarse-graining
  - cheminformatics
authors:
  - name: Roshan Shrestha
    orcid: 0000-0002-9356-5136
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 28 May 2026
bibliography: paper.bib
---

# Summary

`MolScope` is a lightweight Python toolkit for static molecular structure
analysis, graph export, and coarse-graining. It reads common coordinate
formats (XYZ, PDB, mmCIF, SDF), provides atom and residue level selections,
computes geometry, contacts, and a dependency-free secondary structure
assignment, and exposes three workflow-shaped outputs: fixed-width structural
descriptor tables, atom-and-bond or residue-contact graphs (with NetworkX,
PyTorch Geometric, and DGL exporters), and coarse-grained bead
representations with simple connectivity.

The package is built on `NumPy` [@harris2020array] and `matplotlib`
[@hunter2007matplotlib] only. All chemistry, file-format, and graph-backend
dependencies (`RDKit` [@rdkit], `Gemmi` [@wojdyr2022gemmi], `NetworkX`
[@hagberg2008networkx], `SciPy` [@virtanen2020scipy], `PyTorch`
[@paszke2019pytorch], `PyTorch Geometric` [@fey2019pytorch_geometric], `DGL`
[@wang2019dgl]) are optional extras that are installed only when the relevant
feature is used. This keeps the base install small and the dependency graph
predictable.

# Statement of need

Existing tooling for molecular structure work in Python sits at two
extremes. Specialist packages such as `MDAnalysis` [@michaud2011mdanalysis],
`MDTraj` [@mcgibbon2015mdtraj], `RDKit`, and `PyMOL` [@pymol] are powerful
and battle-tested, but each carries a substantial dependency footprint and
an API surface aimed at production scientific workflows. At the other
extreme, building from raw `NumPy` quickly grows into a bespoke pile of
parsers, selection logic, and graph builders that are easy to get wrong and
hard to maintain.

For teaching exercises, quick structural quality-control checks, and small
machine-learning-for-molecules prototypes, neither extreme fits well.
Students need readable Python that demonstrates the structure of the
underlying data without disappearing into a framework. Researchers
prototyping graph neural networks for molecules need a deterministic way to
turn a PDB file into a `torch_geometric.data.Data` object without first
standing up a full featurisation pipeline. Modellers want a fast way to
inspect a coarse-grained mapping before committing to a production Martini
[@marrink2007martini] setup.

`MolScope` sits in this gap. It is intentionally narrower than `MDAnalysis`
or `RDKit`, and intentionally more opinionated than a `NumPy`-only approach:
every workflow ends in a concrete, ready-to-use object (a descriptor table,
a graph, or a bead model). The package documents what it deliberately is
not, so users can identify when their work has outgrown it and reach for a
heavier specialist tool.

# Functionality

`MolScope` is organised around three core workflows, each documented end to
end in the project's tutorials:

1. **PDB to descriptors.** Geometry, contacts, contact maps, residue-level
   analyses, a dependency-free DSSP-style secondary structure assignment,
   and optional `RDKit`-backed scalar descriptors when the `chem` extra is
   installed. After collapsing assignments to helix, strand, and coil, the
   DSSP-style implementation reaches 98 to 99% per-residue agreement with
   reference `mkdssp` [@kabsch1983dssp] across three fold classes in the
   validation suite: helix-dominated Aquaporin-1 (99.1%), mixed alpha/beta
   ubiquitin (100%), and the all-beta SH3 domain (98.2%). This is a targeted
   regression check rather than an exhaustive benchmark panel; the package's
   per-workflow limitations page records its scope explicitly.

2. **PDB to graph or GNN.** Atom-and-bond graphs and residue contact graphs
   with named node and edge feature presets, Laplacian and random-walk
   positional encodings, and exporters to `NetworkX`, `PyTorch Geometric`,
   and `DGL`. When explicit bond records are absent, bonds are inferred
   geometrically from covalent radii.

3. **PDB to coarse-grained beads.** Residue centroids, simplified
   Martini-style backbone and side-chain mappings, custom user mappings,
   and virtual sites preserved as derived coordinate metadata. The output
   is a regular `Molecule` so plotting and graph export remain available
   on the coarse-grained representation.

The package also ships with a command-line interface (`molscope`) for the
most common batch operations, an interactive 3D viewer via the optional
`viz` extra, and a two-tier validation suite that combines dependency-free
invariants with cross-checks against `MDAnalysis`, `RDKit`, and `mkdssp`
where the reference tools are available.

# Quality, documentation, and continuous integration

`MolScope` is tested on Python 3.9, 3.11, and 3.13. Continuous integration
runs the full test suite on each version, measures coverage with the
optional backends installed, and runs a separate scientific validation job
that cross-checks geometry, contacts, and secondary structure against
reference scientific tools. Documentation is built with `MkDocs Material`
and hosted at <https://molscope.readthedocs.io/>, and a per-workflow
limitations page makes explicit what each workflow does and does not cover.

# Availability and archival

`MolScope` is distributed on PyPI as `molscope` and developed openly on
GitHub at <https://github.com/roshan2004/molscope> under the MIT licence.
Each release is automatically archived on Zenodo: the concept DOI
[10.5281/zenodo.20433850](https://doi.org/10.5281/zenodo.20433850)
resolves to the latest archived version, and the version under review
here (v0.8.3) is archived at
[10.5281/zenodo.20433851](https://doi.org/10.5281/zenodo.20433851).

# Acknowledgements

The author thanks the maintainers of `NumPy`, `matplotlib`, `RDKit`,
`MDAnalysis`, `PyTorch Geometric`, `DGL`, and `Gemmi`, on whose work
`MolScope`'s optional integrations depend.

# References
