# Introducing MolScope: lightweight molecular structure analysis in Python

*28 May 2026*

Working with molecular structures in Python usually means committing to a heavy
ecosystem before you have written your first line of analysis code. MDAnalysis
for trajectories and selections. RDKit for chemistry. PyMOL or VMD for
visualisation. Each is brilliant at what it does, and each carries substantial
conceptual and dependency weight.

For teaching exercises, quick structural checks, and small machine-learning-
for-molecules prototypes, that weight is often more than you need. You want to
read a PDB file, select some atoms, compute a contact map, export a graph to
PyTorch Geometric, and move on.

That is the gap MolScope is built for.

## What MolScope is

MolScope is a lightweight Python toolkit for molecular structure analysis,
graph export, and coarse-graining. It reads `.xyz`, `.pdb`, `.cif` and `.sdf`
files, selects and analyses atoms, visualises structures in 3D, and turns
coordinates into the three things people most often actually want:

- A fixed-width descriptor table for screening, QC, and classical ML.
- A graph (NetworkX, PyTorch Geometric, or DGL) for message-passing
  experiments.
- A coarse-grained bead model for inspection or graph prototyping.

The base install is just NumPy and matplotlib. Optional extras (RDKit, Gemmi,
NetworkX, SciPy, PyTorch, PyTorch Geometric) light up the features that need
them, but nothing is pulled in until you ask for it.

## Three core workflows

The library is deliberately organised around three workflows that each end in
something useful, not around an open-ended toolkit you have to assemble
yourself.

**PDB to descriptors.** Read a structure, compute geometry, contacts,
secondary structure, and optionally RDKit-backed chemical descriptors. The
output is a fixed-width table that drops cleanly into pandas, scikit-learn, or
a notebook.

**PDB to graph or GNN.** Build atom-and-bond graphs or residue contact graphs,
attach node and edge features (atomic number, mass, formal charge, aromatic
flag, distance, optional bond order), compute Laplacian or random-walk
positional encodings, and export to NetworkX, PyTorch Geometric, or DGL.

**PDB to coarse-grained beads.** Map an atomistic structure to residue
centroids, simplified Martini-style backbone and side-chain beads, or a custom
mapping. You get bead coordinates, simple connectivity, a report of assigned
and dropped atoms, and virtual sites preserved as metadata.

A small example: turning Aquaporin-1 into a PyTorch Geometric graph in three
lines.

```python
import molscope as ms

aqp = ms.read_pdb("1fqy.pdb")
data = aqp.to_pyg_data(node_preset="ml", edge_preset="basic")
```

That is the whole story for the graph-ML workflow. `data` is a standard
`torch_geometric.data.Data` object with `x`, `pos`, `edge_index`, and
`edge_attr` populated and ready for a message-passing model. The tutorials in
the docs walk each workflow end to end on a real PDB.

## Who it is for, and who it is not

MolScope is for students learning molecular coordinate formats and structural
analysis from readable Python, for modellers who want quick static-structure
checks, selections, and lightweight coarse-grained mapping prototypes, and for
ML-for-molecules learners who need deterministic descriptors and graph exports
before moving to bigger frameworks.

It is deliberately not a few things, and the docs say so plainly. It is not a
trajectory engine: reach for MDAnalysis or MDTraj if you need to iterate over
thousands of frames. It is not a Martini force-field generator: production
coarse-grained model preparation belongs in Martini tooling. It is not a
replacement for RDKit when you need full chemistry perception, or for PyMOL,
VMD, or ChimeraX when you want deep interactive visualisation. The
[limitations page](https://molscope.readthedocs.io/en/latest/limitations/)
lists what each workflow does and does not cover.

By not trying to replace any of those tools, MolScope stays lightweight,
approachable, and easy to drop into a teaching repo or a quick prototype.

## Try it

```bash
pip install molscope
```

Optional extras for the backends you want:

```bash
pip install "molscope[chem]"    # rdkit-backed chemistry
pip install "molscope[cif]"     # gemmi-backed mmCIF validation
pip install "molscope[graph]"   # networkx exporter
pip install "molscope[pyg]"     # torch + torch-geometric
```

* Docs: [molscope.readthedocs.io](https://molscope.readthedocs.io/)
* Source: [github.com/roshan2004/molscope](https://github.com/roshan2004/molscope)
* PyPI: [pypi.org/project/molscope](https://pypi.org/project/molscope/)

The project is MIT-licensed and the current release is `0.8.2`. CI runs the
test suite on Python 3.9, 3.11 and 3.13, with separate jobs measuring coverage
across the optional backends and cross-checking secondary structure against
`mkdssp` where the reference binary is available.

If MolScope helps with your teaching, prototyping or ML work, I would be glad
to hear about it. Bug reports, feature ideas and small contributions are all
welcome on the
[issue tracker](https://github.com/roshan2004/molscope/issues).
