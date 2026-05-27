# Tutorials

These tutorials are guided, copy-pasteable workflows over the bundled example
structures. They start from ordinary PDB files and end with the three outputs
MolScope is most often used for: descriptor tables, graph ML inputs, and
coarse-grained bead models.

## Choose a workflow

| Tutorial | Output | Use when |
| --- | --- | --- |
| [PDB to descriptors](pdb-to-descriptors.md) | A fixed-width CSV feature table | You want classical ML features, quick structure summaries, or batch descriptors. |
| [PDB to graph/GNN](pdb-to-graph-gnn.md) | Atom/residue graphs and a PyTorch Geometric toy dataset | You want message-passing inputs or a minimal graph neural network workflow. |
| [PDB to coarse-grained beads](pdb-to-coarse-grained-beads.md) | Residue and backbone/sidechain bead models | You want an interpretable reduced representation for inspection or graph prototyping. |

Run commands from the repository root so paths like `examples/data/1fqy.pdb`
resolve correctly.
