# Examples

The examples here are small scripts that can be copied into notebooks or Python
files. For more guided workflows, see the [Tutorials](../tutorials/index.md):
PDB to descriptors, PDB to graph/GNN, and PDB to coarse-grained beads.

The repository also includes:

- `examples/tour.py`: an end-to-end tour over the bundled sample structures.
- `examples/geometry.py`: a tour of every geometry quantity (see [Molecular geometry tour](geometry-tour.md)).
- `examples/protein_analysis.py`: protein metadata, contacts, simplified DSSP, NMR ensemble contacts, and binding sites.
- `examples/coarse_graining.py`: residue COM, centroid, and simplified BB/SC coarse-graining with a visual mapping comparison.
- `examples/data/`: small bundled structures used by the examples and tests.
- `notebooks/molscope_tour.ipynb`: a notebook version of the tour.
- `notebooks/protein_analysis_from_scratch.ipynb`: a tutorial notebook over `1fqy`, `1aml`, and `3ptb`.
- `docs/examples/protein-analysis-from-scratch.md`: the short doc version of the same workflow.
- `docs/examples/pdb-to-graph-cg.md`: a focused PDB to graph and coarse-grain walkthrough.
- `docs/examples/residue-contact-graphs.md`: residue nodes plus spatial contact edges.
- `docs/examples/pdb-to-pyg-ml.md`: PDB to PyTorch Geometric classifier/regressor.
- `examples/graph_to_gnn.py`: graph export and a PyTorch Geometric GNN forward pass.
- `examples/pdb_to_pyg_ml.py`: runnable graph-level PyG toy ML example.
- `examples/residue_contact_graph.py`: runnable residue-contact graph drawing example.

Suggested notebook examples for future work:

- `01_read_plot_structure.ipynb`
- `02_selections_geometry.ipynb`
- `03_nmr_ensemble_analysis.ipynb`
- `04_molecular_graphs_for_ml.ipynb`
- `05_coarse_graining.ipynb`
